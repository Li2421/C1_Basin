"""Execute frozen matched Safety-versus-recovery continuations.

This runner never changes either controller.  Branch N executes Safety for the
entire remaining absolute horizon.  Branch R forces the one legal recovery
entry on the queried step, then delegates to the authoritative one-segment
structured-eta state machine and its frozen learned exit head.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from branch_common import atomic_json, canonical_hash, file_hash, verify_semantic_hash


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/recovery_entry_identifiability_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SINGLE = ROOT / "diagnostics/single_segment_recovery_training_v1"
REFERENCE = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"
ETA_CHECKPOINT = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
FUTURE_ROOT_SEED = 2026102603
ETA_SHA256 = "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095"
EXIT_SHA256 = "7b3bd0d9caff3bfa4baed95d227a6484b5f6f176d7c5b21a33715acb151db694"
EXIT_THRESHOLD_LOGIT = -1.0986122886681098
HORIZON = 850

for value in (str(SYSROOT), str(ROOT), str(PILOT), str(SINGLE), str(HERE)):
    while value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(SINGLE), str(HERE)]


class ForcedEntry:
    """A single-use true decision; a second entry query is an audit failure."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, feature: np.ndarray) -> bool:
        value = np.asarray(feature)
        if value.shape != (214,):
            raise ValueError(("forced-entry feature mismatch", value.shape))
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("one-segment recovery attempted a second entry query")
        return True


def classify(env: Any, error: Mapping[str, Any] | None) -> str:
    if error is not None:
        return "other"
    summary = env.summary()
    if summary["wall_collision"] or summary["agent_collision"]:
        return "collision"
    if summary["success"]:
        return "success"
    if summary["deadlock"]:
        return "deadlock"
    return "timeout"


def make_key_schedule(current_key: np.ndarray, future_episode_key: Any,
                      decision_step: int) -> Callable[[Any, int], Any]:
    """Reuse the saved current draw, then the pair's common future stream."""

    import jax

    current = np.asarray(current_key, dtype=np.uint32)
    if current.shape != (2,):
        raise ValueError(("current Flow key must be uint32[2]", current.shape))

    def key_for_step(unused_episode_key: Any, step: int) -> Any:
        if int(step) == int(decision_step):
            return current
        return jax.random.fold_in(future_episode_key, int(step))

    return key_for_step


def _verify_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    verify_semantic_hash(manifest, path)
    if manifest.get("schema") != "recovery_entry_paired_branch_manifest_v1":
        raise RuntimeError((path, "branch manifest schema mismatch"))
    if manifest.get("status") != "FROZEN_READY_FOR_EXECUTION":
        raise RuntimeError((path, "branch manifest is not frozen/executable"))
    if int(manifest["matched_future_semantics"].get("future_root_seed", -1)) != FUTURE_ROOT_SEED:
        raise RuntimeError((path, "future root seed mismatch"))
    policy = manifest["recovery_policy"]
    if policy.get("system") != "PERSISTENT_STRUCTURED_ETA":
        raise RuntimeError((path, "only structured eta is legal"))
    if policy.get("one_segment_only") is not True or policy.get("no_reentry") is not True:
        raise RuntimeError((path, "single-segment/no-reentry guardrail missing"))
    eta = policy["eta_checkpoint"]
    exit_head = policy["exit_head"]
    if eta.get("sha256") != ETA_SHA256 or file_hash(Path(eta["path"])) != ETA_SHA256:
        raise RuntimeError("structured-eta checkpoint mismatch")
    if (exit_head.get("sha256") != EXIT_SHA256
            or file_hash(Path(exit_head["path"])) != EXIT_SHA256
            or float(exit_head.get("threshold_logit")) != EXIT_THRESHOLD_LOGIT
            or int(exit_head.get("input_dimension", -1)) != 217
            or exit_head.get("checkpoint_local_threshold_ignored") is not True):
        raise RuntimeError("authoritative learned-exit checkpoint/override mismatch")
    state_machine = policy["state_machine"]
    if file_hash(Path(state_machine["path"])) != state_machine["sha256"]:
        raise RuntimeError("authoritative state machine changed after manifest freeze")
    hashes_path = Path(policy["controller_hash_manifest"]["path"])
    if file_hash(hashes_path) != policy["controller_hash_manifest"]["sha256"]:
        raise RuntimeError("controller hash manifest changed after branch freeze")
    hashes = json.loads(hashes_path.read_text())
    for name in ("flowbc", "environment", "projection_constraints", "projection_retry",
                 "event_priority_and_monitor", "startup_feature_builder", "eta_basis",
                 "structured_eta_recovery"):
        record = hashes[name]
        if file_hash(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((name, "frozen asset mismatch"))
    tasks = manifest.get("tasks", [])
    if len(tasks) != int(manifest.get("task_count", -1)):
        raise RuntimeError((path, "task count mismatch"))
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError((path, "duplicate task id"))
    pairs: dict[tuple[str, str], set[str]] = {}
    for task in tasks:
        pairs.setdefault((task["state_id"], task["future_stream_id"]), set()).add(task["branch"])
    if any(branches != {"N", "R"} for branches in pairs.values()):
        raise RuntimeError((path, "N/R pair coverage mismatch"))


def _assert_replay(context: Any, task: Mapping[str, Any]) -> dict[str, float]:
    maxima = {
        "current_u_flow": float(np.max(np.abs(
            np.asarray(context.u_flow) - np.asarray(task["current_u_flow"], dtype=np.float64)
        ))),
        "current_u_safe": float(np.max(np.abs(
            np.asarray(context.u_safe) - np.asarray(task["current_u_safe"], dtype=np.float64)
        ))),
        "current_feature": float(np.max(np.abs(
            np.asarray(context.feature) - np.asarray(task["feature"], dtype=np.float64)
        ))),
        "current_flow_key": float(np.max(np.abs(
            np.asarray(context.flow_key, dtype=np.uint32).astype(np.int64)
            - np.asarray(task["current_flow_key_data"], dtype=np.uint32).astype(np.int64)
        ))),
    }
    if max(maxima.values()) > 1e-12:
        raise RuntimeError((task["task_id"], "decision-state replay mismatch", maxima))
    return maxima


def _run_safety(kernel: Any, env: Any, task: Mapping[str, Any]) -> tuple[list[Any], dict[str, float]]:
    records: list[Any] = []
    first = kernel.prepare(need_feature=True)
    maxima = _assert_replay(first, task)
    done, info = kernel.execute(first.u_safe)
    records.append((first, info))
    while not done:
        context = kernel.prepare(need_feature=False)
        done, info = kernel.execute(context.u_safe)
        records.append((context, info))
    return records, maxima


def run_task(task: Mapping[str, Any], manifest: Mapping[str, Any], *, config: Any,
             cbf: Any, sample_action: Callable[..., Any], eta_model: Any,
             exit_head: Callable[[np.ndarray], bool]) -> dict[str, Any]:
    import jax
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from full_state_io import restore_full_history
    from state_machine import AuthoritativeStepKernel, RecoverySystem, SingleSegmentRecoveryMachine

    state_path = Path(task["state_file"])
    if file_hash(state_path) != task["state_sha256"]:
        raise RuntimeError((task["task_id"], "state hash mismatch"))
    env = restore_full_history(state_path, config)
    if int(env.step_count) != int(task["absolute_step"]):
        raise RuntimeError((task["task_id"], "absolute step mismatch"))
    future_key = jax.random.fold_in(
        jax.random.PRNGKey(np.uint32(FUTURE_ROOT_SEED)),
        int(task["future_rollout_id"]),
    )
    kernel = AuthoritativeStepKernel(
        env=env, config=config, cbf=cbf, episode_key=future_key,
        sample_action=sample_action, feature_builder=StartupAwareFeatureBuilder(),
        project=project_velocity_with_retry,
        key_for_step=make_key_schedule(
            np.asarray(task["current_flow_key_data"], dtype=np.uint32),
            future_key, int(task["absolute_step"]),
        ),
    )
    began = time.monotonic()
    error: dict[str, Any] | None = None
    branch = str(task["branch"])
    records: list[Any] = []
    machine = None
    replay_maxima: dict[str, float] = {}
    try:
        if branch == "N":
            records, replay_maxima = _run_safety(kernel, env, task)
        elif branch == "R":
            entry = ForcedEntry()
            machine = SingleSegmentRecoveryMachine(
                kernel=kernel, system=RecoverySystem.STRUCTURED_ETA,
                entry_head=entry, exit_head=exit_head, eta_model=eta_model,
            )
            first = machine.step()
            records.append(first)
            replay_maxima = _assert_replay(first, task)
            if (first.decision != "ENTER" or first.exit_queried
                    or not first.recovery_action or first.entry_step != int(task["absolute_step"])):
                raise RuntimeError((task["task_id"], "forced entry semantics violated"))
            while not env.done:
                records.append(machine.step())
            if entry.calls != 1 or machine.eta_query_count != 1:
                raise RuntimeError((task["task_id"], "eta/entry was not queried exactly once"))
        else:
            raise RuntimeError((task["task_id"], "unknown branch", branch))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    terminal = classify(env, error)
    if branch == "N":
        first_retries = sum(int(context.first_projection_retry) for context, _ in records)
        second_retries = 0
        monitor_rows = [info for _, info in records]
        entry_step = exit_step = None
        recovery_transitions = eta_queries = entry_queries = exit_queries = 0
        returned_to_safety = False
        eta_hash = None
    else:
        first_retries = sum(int(record.first_projection_retry) for record in records)
        second_retries = sum(int(record.second_projection_retry) for record in records)
        monitor_rows = [record.monitor_info for record in records]
        entry_step = machine.memory.entry_step
        exit_step = machine.memory.exit_step
        recovery_transitions = int(machine.memory.recovery_transitions)
        eta_queries = int(machine.eta_query_count)
        entry_queries = int(machine.entry_query_count)
        exit_queries = int(machine.exit_query_count)
        returned_to_safety = exit_step is not None
        eta_hash = canonical_hash(np.asarray(machine.memory.eta_latched).tolist())
    wall_events = sum(int(info.get("wall_collision", False)) for info in monitor_rows)
    agent_events = sum(int(info.get("agent_collision", False)) for info in monitor_rows)
    transitions = int(kernel.physical_transition_count)
    return {
        "schema": "recovery_entry_paired_branch_result_v1",
        "record_complete": error is None,
        "manifest_content_sha256": manifest["content_sha256"],
        "wave": manifest["wave"],
        "task_id": task["task_id"], "task_index": int(task["task_index"]),
        "state_id": task["state_id"], "root_source_id": task["root_source_id"],
        "state_sha256": task["state_sha256"], "absolute_step": int(task["absolute_step"]),
        "branch": branch, "future_index": int(task["future_index"]),
        "future_stream_id": task["future_stream_id"],
        "future_rollout_id": int(task["future_rollout_id"]),
        "future_rng_state_hash": task["future_rng_state_hash"],
        "current_flow_realization_id": task["current_flow_realization_id"],
        "current_replay_max_abs": replay_maxima,
        "success": terminal == "success", "outcome": terminal,
        "deadlock": terminal == "deadlock", "timeout": terminal == "timeout",
        "collision": terminal == "collision", "other_failure": terminal == "other",
        "terminal_global_step": int(env.step_count),
        "remaining_physical_steps": transitions,
        "flow_sample_count": int(kernel.flow_sample_count),
        "physical_transition_count": transitions,
        "entry_step": entry_step, "exit_step": exit_step,
        "recovery_transitions": recovery_transitions,
        "returned_to_safety": returned_to_safety,
        "entry_query_count": entry_queries, "exit_query_count": exit_queries,
        "eta_query_count": eta_queries, "eta_latched_hash": eta_hash,
        "first_step_exit_queried": bool(records[0].exit_queried) if branch == "R" and records else False,
        "first_projection_retry_count": first_retries,
        "second_projection_retry_count": second_retries,
        "agent_collision_events": agent_events, "wall_collision_events": wall_events,
        "invalid_actions": int(error is not None and "invalid" in error["message"].lower()),
        "nan_inf_events": int(error is not None and "finite" in error["message"].lower()),
        "projection_failures": int(error is not None and "projection" in error["message"].lower()),
        "execution_error": error,
        "runtime_seconds": time.monotonic() - began,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    import jax.numpy as jnp
    from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor
    from run_full_episode_system import ThresholdedFrozenHead
    from single_integrator.cbf import CBFConfig
    from single_integrator.environment import Config
    from single_integrator.evaluate import load_policy

    manifest = json.loads(args.manifest.read_text())
    _verify_manifest(manifest, args.manifest)
    reference = json.loads(REFERENCE.read_text())
    config = Config(**reference["environment"])
    cbf = CBFConfig(**reference["cbf"])
    if int(config.max_steps) != HORIZON:
        raise RuntimeError(("official horizon mismatch", config.max_steps))
    hashes_path = Path(manifest["recovery_policy"]["controller_hash_manifest"]["path"])
    hashes = json.loads(hashes_path.read_text())
    policy, provenance = load_policy(Path(hashes["flowbc"]["path"]))
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC/environment mismatch")
    sample_action = jax.jit(
        lambda observation, key: policy.sample_actions(observation[None], seed=key)[0]
    )
    eta_model = FixedDEtaPredictor(ETA_CHECKPOINT)
    exit_head = ThresholdedFrozenHead(manifest["recovery_policy"]["exit_head"], 217)
    assigned = [task for task in manifest["tasks"]
                if int(task["task_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    new_tasks = new_steps = 0
    for task in assigned:
        path = args.result_directory / f"task_{int(task['task_index']):05d}.json"
        if path.is_file():
            prior = json.loads(path.read_text())
            if (prior.get("record_complete") is True and prior.get("task_id") == task["task_id"]
                    and prior.get("manifest_content_sha256") == manifest["content_sha256"]):
                continue
            raise RuntimeError((path, "existing branch result is not exact/reusable"))
        row = run_task(
            task, manifest, config=config, cbf=cbf, sample_action=sample_action,
            eta_model=eta_model, exit_head=exit_head,
        )
        atomic_json(path, row)
        if not row["record_complete"]:
            raise RuntimeError((task["task_id"], "branch failed", row["execution_error"]))
        new_tasks += 1
        new_steps += int(row["remaining_physical_steps"])
        print(json.dumps({"task_index": task["task_index"], "branch": task["branch"],
                          "outcome": row["outcome"]}), flush=True)
    atomic_json(args.result_directory / f"runtime_shard{args.shard_index}.json", {
        "schema": "recovery_entry_branch_runtime_v1", "wave": manifest["wave"],
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "new_branch_continuations": new_tasks, "new_physical_steps": new_steps,
        "device": args.device, "jax_devices": [str(item) for item in jax.devices()],
        "manifest_path": str(args.manifest.resolve()), "manifest_sha256": file_hash(args.manifest),
        "cpu_thread_cap": 6,
    })


if __name__ == "__main__":
    main()

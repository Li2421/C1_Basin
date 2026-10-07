"""Execute matched Stage-2 S-now/L-now/R-now continuation branches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1"
STAGE1 = ROOT / "diagnostics/semantic_unified_controller_v1/stage1_state_driven_local"
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage2_recovery_entry"
for value in (str(SYSROOT), str(ROOT), str(PILOT), str(BASE), str(STAGE1), str(HERE)):
    while value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(BASE), str(STAGE1), str(HERE)]

from pilot_common import DeterministicGphi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from full_state_io import restore_full_history  # noqa: E402
from local_common import FirstLocalOverride, FrozenLocalHead, StateDrivenLocalMachine  # noqa: E402
from mode_common import (  # noqa: E402
    ACTION_NAMES, DIRECT_CHECKPOINT, DIRECT_SHA256, ETA_CHECKPOINT, ETA_SHA256,
    FUTURE_ROOT_SEED, HERE, content_hash, sha256, verify_content_hash,
)
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from state_machine import AuthoritativeStepKernel, RecoverySystem, SingleSegmentRecoveryMachine  # noqa: E402


FROZEN = BASE / "controller_and_projection_hashes.json"
ENVIRONMENT = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"


class ConstantHead:
    def __init__(self, value: bool):
        self.value = bool(value)

    def __call__(self, feature: np.ndarray) -> bool:
        return self.value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def key_schedule(current_key: Any, future_key: Any, decision_step: int):
    import jax

    current = np.asarray(current_key, dtype=np.uint32)
    if current.shape != (2,):
        raise ValueError(("current Flow key shape", current.shape))

    def key_for_step(unused_episode_key: Any, step: int):
        return current if int(step) == int(decision_step) else jax.random.fold_in(future_key, int(step))

    return key_for_step


def classify(env: Any, error: dict[str, Any] | None) -> str:
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


def verify_assets(manifest: dict[str, Any]) -> tuple[Config, CBFConfig, Path]:
    if sha256(DIRECT_CHECKPOINT) != DIRECT_SHA256 or sha256(ETA_CHECKPOINT) != ETA_SHA256:
        raise RuntimeError("frozen learned checkpoint mismatch")
    frozen = json.loads(FROZEN.read_text())
    for name, row in frozen.items():
        if isinstance(row, dict) and "path" in row and sha256(row["path"]) != row["sha256"]:
            raise RuntimeError((name, "frozen asset mismatch"))
    reference = json.loads(ENVIRONMENT.read_text())
    config, cbf = Config(**reference["environment"]), CBFConfig(**reference["cbf"])
    if config.max_steps != 850 or config.dt != 0.05:
        raise RuntimeError("official horizon/dt mismatch")
    expected_assets = {
        "direct_g": (DIRECT_CHECKPOINT, DIRECT_SHA256),
        "structured_eta": (ETA_CHECKPOINT, ETA_SHA256),
    }
    for name, (path, digest) in expected_assets.items():
        if manifest[name] != {"path": str(path), "sha256": digest}:
            raise RuntimeError((name, "manifest asset mismatch"))
    expected_impl = {"mode_common": HERE / "mode_common.py", "branch_runner": HERE / "run_mode_branches.py"}
    if set(manifest.get("implementation_hashes", {})) != set(expected_impl):
        raise RuntimeError("implementation hash manifest incomplete")
    for name, path in expected_impl.items():
        row = manifest["implementation_hashes"][name]
        if Path(row["path"]).resolve() != path.resolve() or sha256(path) != row["sha256"]:
            raise RuntimeError((name, "implementation changed after manifest freeze"))
    return config, cbf, Path(frozen["flowbc"]["path"])


def run_task(task: dict[str, Any], manifest: dict[str, Any], *, sample_action: Any,
             direct_model: Any, eta_model: Any, config: Config, cbf: CBFConfig) -> dict[str, Any]:
    import jax

    state_path = Path(task["state_file"])
    if sha256(state_path) != task["state_sha256"]:
        raise RuntimeError((task["task_id"], "state hash mismatch"))
    env = restore_full_history(state_path, config)
    if int(env.step_count) != int(task["absolute_step"]):
        raise RuntimeError((task["task_id"], "absolute step mismatch"))
    future_key = jax.random.fold_in(
        jax.random.PRNGKey(np.uint32(FUTURE_ROOT_SEED)), int(task["future_rollout_id"])
    )
    kernel = AuthoritativeStepKernel(
        env=env, config=config, cbf=cbf, episode_key=future_key,
        sample_action=sample_action, feature_builder=StartupAwareFeatureBuilder(),
        project=project_velocity_with_retry,
        key_for_step=key_schedule(task["current_flow_key_data"], future_key, int(task["absolute_step"])),
    )
    local_record = manifest["downstream_policy"]
    local_path = Path(local_record["checkpoint"])
    if sha256(local_path) != local_record["sha256"]:
        raise RuntimeError("frozen local-policy head hash mismatch")
    local_head = FrozenLocalHead(local_path, threshold_logit=float(local_record["threshold_logit"]))
    action = int(task["action"])
    if action not in ACTION_NAMES:
        raise RuntimeError((task["task_id"], "invalid action"))
    records: list[Any] = []
    error = None
    replay_max = None
    eta_latched = None
    began = time.monotonic()
    try:
        if action in (0, 1):
            override = FirstLocalOverride(action, local_head)
            machine = StateDrivenLocalMachine(kernel=kernel, local_head=override, direct_model=direct_model)
            first = machine.step()
            records.append(first)
            replay_max = max(
                float(np.max(np.abs(first.u_flow - np.asarray(task["current_u_flow"])))),
                float(np.max(np.abs(first.u_safe - np.asarray(task["current_u_safe"])))),
                float(np.max(np.abs(first.feature - np.asarray(task["feature"])))),
            )
            if replay_max > 1e-12:
                raise RuntimeError((task["task_id"], "decision replay mismatch", replay_max))
            while not env.done:
                records.append(machine.step())
            local_queries = int(machine.head_query_count)
            direct_queries = int(machine.direct_query_count)
            eta_queries = recovery_queries = 0
            local_events = sum(row.decision == "LOCAL" for row in records)
            recovery_steps = 0
        else:
            recovery = SingleSegmentRecoveryMachine(
                kernel=kernel, system=RecoverySystem.STRUCTURED_ETA,
                entry_head=ConstantHead(True), exit_head=ConstantHead(False), eta_model=eta_model,
            )
            first = recovery.step()
            records.append(first)
            replay_max = max(
                float(np.max(np.abs(first.u_flow - np.asarray(task["current_u_flow"])))),
                float(np.max(np.abs(first.u_safe - np.asarray(task["current_u_safe"])))),
                float(np.max(np.abs(first.feature - np.asarray(task["feature"])))),
            )
            if replay_max > 1e-12:
                raise RuntimeError((task["task_id"], "decision replay mismatch", replay_max))
            while not env.done:
                records.append(recovery.step())
            eta_latched = recovery.memory.eta_latched.copy()
            local_queries = local_events = direct_queries = 0
            eta_queries = int(recovery.eta_query_count)
            recovery_queries = int(recovery.exit_query_count)
            recovery_steps = int(recovery.memory.recovery_transitions)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
        local_queries = direct_queries = eta_queries = recovery_queries = local_events = recovery_steps = -1
    outcome = classify(env, error)
    eta_hash = "NOT_APPLICABLE" if eta_latched is None else content_hash(np.asarray(eta_latched).tolist())
    return {
        "schema": "semantic_mode_paired_branch_result_v1", "record_complete": error is None,
        "task_id": task["task_id"], "task_index": int(task["task_index"]),
        "decision_id": task["decision_id"], "pass_id": task["pass_id"],
        "action": action, "action_name": ACTION_NAMES[action],
        "root_source_id": task["root_source_id"], "split": task["split"],
        "state_id": task["state_id"], "state_hash": task["state_sha256"],
        "absolute_step": int(task["absolute_step"]), "mode": "NORMAL",
        "current_flow_realization_id": task["current_flow_realization_id"],
        "future_stream_id": task["future_stream_id"],
        "future_rollout_id": int(task["future_rollout_id"]),
        "future_rng_state_hash": task["future_rng_state_hash"],
        "downstream_policy_hash": task["downstream_policy_hash"],
        "eta_latched": None if eta_latched is None else eta_latched.tolist(),
        "eta_latched_hash": eta_hash,
        "success": outcome == "success", "outcome": outcome,
        "deadlock": outcome == "deadlock", "timeout": outcome == "timeout",
        "collision": outcome == "collision", "other_failure": outcome == "other",
        "remaining_jdef": config.dt * float(sum(np.sum((row.u_exec-row.u_safe) ** 2) for row in records)),
        "terminal_global_step": int(env.step_count), "remaining_physical_steps": len(records),
        "physical_transition_count": int(kernel.physical_transition_count),
        "flow_sample_count": int(kernel.flow_sample_count),
        "local_head_query_count": local_queries, "direct_query_count": direct_queries,
        "eta_query_count": eta_queries, "recovery_exit_query_count": recovery_queries,
        "local_action_count": local_events, "recovery_transition_count": recovery_steps,
        "current_replay_max_abs": replay_max,
        "first_projection_retry_count": sum(bool(row.first_projection_retry) for row in records),
        "second_projection_retry_count": sum(bool(row.second_projection_retry) for row in records),
        "agent_collision_events": sum(int(row.monitor_info.get("agent_collision", False)) for row in records),
        "wall_collision_events": sum(int(row.monitor_info.get("wall_collision", False)) for row in records),
        "invalid_actions": int(error is not None and "invalid" in error["message"].lower()),
        "nan_inf_events": int(error is not None and "finite" in error["message"].lower()),
        "projection_failures": int(error is not None and "projection" in error["message"].lower()),
        "execution_error": error, "manifest_content_sha256": manifest["content_sha256"],
        "runtime_seconds": time.monotonic() - began,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    manifest = json.loads(args.manifest.read_text())
    verify_content_hash(manifest, args.manifest)
    if manifest.get("schema") != "semantic_mode_paired_branch_manifest_v1" or manifest.get("status") != "FROZEN_READY_FOR_EXECUTION":
        raise RuntimeError("invalid Stage-2 manifest")
    if manifest.get("contains_periodic_timing") is not False or manifest.get("recovery_exit_enabled") is not False:
        raise RuntimeError("Stage-2 semantic mismatch")
    if int(manifest["matched_future_semantics"]["future_root_seed"]) != FUTURE_ROOT_SEED:
        raise RuntimeError("future root mismatch")
    config, cbf, flow_path = verify_assets(manifest)
    policy, provenance = load_policy(flow_path)
    reference = json.loads(ENVIRONMENT.read_text())
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    direct_model = DeterministicGphi(DIRECT_CHECKPOINT)
    eta_model = FixedDEtaPredictor(ETA_CHECKPOINT)
    output_dir = HERE / "runs/paired_branches" / manifest["stage"]
    assigned = [task for task in manifest["tasks"] if int(task["task_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    new_tasks = new_steps = 0
    for task in assigned:
        path = output_dir / f"task_{int(task['task_index']):05d}.json"
        if path.is_file():
            prior = json.loads(path.read_text())
            if prior.get("record_complete") and prior.get("manifest_content_sha256") == manifest["content_sha256"]:
                continue
            raise RuntimeError((path, "non-reusable prior result"))
        row = run_task(task, manifest, sample_action=sample_action, direct_model=direct_model,
                       eta_model=eta_model, config=config, cbf=cbf)
        atomic_json(path, row)
        if not row["record_complete"]:
            raise RuntimeError((task["task_id"], row["execution_error"]))
        new_tasks += 1
        new_steps += int(row["physical_transition_count"])
        print(json.dumps({"task": task["task_index"], "outcome": row["outcome"]}), flush=True)
    atomic_json(output_dir / f"runtime_shard{args.shard_index}.json", {
        "schema": "semantic_mode_branch_runtime_v1", "stage": manifest["stage"],
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "new_continuations": new_tasks, "new_physical_steps": new_steps,
        "device": args.device, "jax_devices": [str(item) for item in jax.devices()],
        "manifest_sha256": sha256(args.manifest),
    })


if __name__ == "__main__":
    main()

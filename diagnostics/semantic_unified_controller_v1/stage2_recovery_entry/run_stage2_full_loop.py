"""Run frozen Stage-2 Safety/local/mode policies on development sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
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
from local_common import FrozenLocalHead, StateDrivenLocalMachine  # noqa: E402
from mode_common import (  # noqa: E402
    DIRECT_CHECKPOINT, DIRECT_SHA256, ETA_CHECKPOINT, ETA_SHA256, FrozenModeHead,
    content_hash, sha256, verify_content_hash,
)
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from stage2_full_loop_common import Stage2ModeMachine  # noqa: E402
from state_machine import AuthoritativeStepKernel  # noqa: E402


FROZEN = BASE / "controller_and_projection_hashes.json"
ENVIRONMENT = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"
SOURCE_MANIFEST = HERE / "source_split_manifest.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


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
    expected_policy_assets = {
        "direct_g": {"path": str(DIRECT_CHECKPOINT), "sha256": DIRECT_SHA256},
        "structured_eta": {"path": str(ETA_CHECKPOINT), "sha256": ETA_SHA256},
    }
    for name, expected_row in expected_policy_assets.items():
        if manifest.get("policy", {}).get(name) != expected_row:
            raise RuntimeError((name, "manifest policy asset mismatch"))
    expected = {
        "mode_common": HERE / "mode_common.py",
        "full_loop_common": HERE / "stage2_full_loop_common.py",
        "full_loop_runner": HERE / "run_stage2_full_loop.py",
    }
    if set(manifest.get("implementation_hashes", {})) != set(expected):
        raise RuntimeError("implementation hash manifest incomplete")
    for name, path in expected.items():
        row = manifest["implementation_hashes"][name]
        if Path(row["path"]).resolve() != path.resolve() or sha256(path) != row["sha256"]:
            raise RuntimeError((name, "implementation changed after manifest freeze"))
    if manifest.get("split") not in {"validation", "calibration"}:
        raise RuntimeError("only frozen development splits may be run")
    source = json.loads(SOURCE_MANIFEST.read_text())
    verify_content_hash(source, SOURCE_MANIFEST)
    if (manifest.get("source_manifest") != str(SOURCE_MANIFEST)
            or manifest.get("source_manifest_sha256") != sha256(SOURCE_MANIFEST)
            or manifest.get("source_manifest_content_sha256") != source["content_sha256"]):
        raise RuntimeError("unauthenticated Stage-2 source manifest")
    expected_tasks = [{
        "task_index": index, "source_id": row["source_id"], "root_source_id": row["source_id"],
        "split": manifest["split"], "episode_index": int(row["episode_index"]),
        "rollout_id": int(row["rollout_id"]), "flow_root_seed": int(row["flow_root_seed"]),
        "initial_positions": row["initial_positions"],
    } for index, row in enumerate(source["sources"][manifest["split"]])]
    if manifest.get("tasks") != expected_tasks:
        raise RuntimeError("full-loop tasks do not equal frozen source split")
    return config, cbf, Path(frozen["flowbc"]["path"])


def run_task(task: dict[str, Any], manifest: dict[str, Any], *, sample_action: Any,
             direct_model: Any, eta_model: Any, config: Config, cbf: CBFConfig) -> dict[str, Any]:
    import jax

    env = GiveWayEnv(config)
    env.reset(np.asarray(task["initial_positions"], dtype=np.float64))
    episode_key = jax.random.fold_in(
        jax.random.PRNGKey(np.uint32(task["flow_root_seed"])), int(task["rollout_id"])
    )
    kernel = AuthoritativeStepKernel(
        env=env, config=config, cbf=cbf, episode_key=episode_key,
        sample_action=sample_action, feature_builder=StartupAwareFeatureBuilder(),
        project=project_velocity_with_retry,
    )
    controller = manifest["controller"]
    records: list[Any] = []
    machine: Any | None = None
    error = None
    began = time.monotonic()
    try:
        if controller == "SAFETY":
            while not env.done:
                context = kernel.prepare(need_feature=False)
                _, info = kernel.execute(context.u_safe)
                records.append(SimpleNamespace(
                    step=context.step, decision="SAFETY", u_safe=context.u_safe,
                    u_exec=context.u_safe, first_projection_retry=context.first_projection_retry,
                    second_projection_retry=False, monitor_info=info,
                ))
        elif controller == "SEMANTIC_LOCAL":
            record = manifest["policy"]["local_head"]
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise RuntimeError("local head hash mismatch")
            machine = StateDrivenLocalMachine(
                kernel=kernel,
                local_head=FrozenLocalHead(path, threshold_logit=float(record["threshold_logit"])),
                direct_model=direct_model,
            )
            records = list(machine.run_to_terminal())
        elif controller == "SEMANTIC_MODE":
            record = manifest["policy"]["mode_head"]
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise RuntimeError("mode head hash mismatch")
            machine = Stage2ModeMachine(
                kernel=kernel, mode_head=FrozenModeHead(path), direct_model=direct_model,
                eta_model=eta_model,
            )
            records = list(machine.run_to_terminal())
        else:
            raise RuntimeError(("unsupported controller", controller))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    outcome = classify(env, error)
    local_steps = [int(row.step) for row in records if row.decision == "LOCAL"]
    recovery_rows = [row for row in records if row.decision in {"ENTER_RECOVERY", "CONTINUE_RECOVERY"}]
    eta_latched = getattr(machine, "eta_latched", None)
    if controller == "SEMANTIC_LOCAL":
        mode_queries = int(machine.head_query_count)
        direct_queries = int(machine.direct_query_count)
        eta_queries = recovery_transitions = 0
        entry_step = None
    elif controller == "SEMANTIC_MODE":
        mode_queries = int(machine.mode_query_count)
        direct_queries = int(machine.direct_query_count)
        eta_queries = int(machine.eta_query_count)
        recovery_transitions = int(machine.recovery_transition_count)
        entry_step = machine.entry_step
    else:
        mode_queries = direct_queries = eta_queries = recovery_transitions = 0
        entry_step = None
    return {
        "schema": "semantic_stage2_full_loop_result_v1", "record_complete": error is None,
        "task_index": int(task["task_index"]), "source_id": task["source_id"],
        "root_source_id": task["root_source_id"], "split": task["split"],
        "controller": controller, "policy_hash": manifest["policy_hash"],
        "outcome": outcome, "success": outcome == "success", "deadlock": outcome == "deadlock",
        "timeout": outcome == "timeout", "collision": outcome == "collision",
        "other_failure": outcome == "other", "terminal_step": int(env.step_count),
        "jdef": config.dt * float(sum(np.sum((row.u_exec-row.u_safe) ** 2) for row in records)),
        "local_action_count": len(local_steps), "local_action_steps": local_steps,
        "mode_query_count": mode_queries, "direct_query_count": direct_queries,
        "eta_query_count": eta_queries, "entry_step": entry_step,
        "eta_latched": None if eta_latched is None else np.asarray(eta_latched).tolist(),
        "eta_latched_hash": None if eta_latched is None else content_hash(np.asarray(eta_latched).tolist()),
        "recovery_transition_count": recovery_transitions,
        "flow_sample_count": int(kernel.flow_sample_count),
        "physical_transition_count": int(kernel.physical_transition_count),
        "agent_collision_events": sum(int(row.monitor_info.get("agent_collision", False)) for row in records),
        "wall_collision_events": sum(int(row.monitor_info.get("wall_collision", False)) for row in records),
        "first_projection_retry_count": sum(bool(row.first_projection_retry) for row in records),
        "second_projection_retry_count": sum(bool(row.second_projection_retry) for row in records),
        "invalid_actions": int(error is not None and "invalid" in error["message"].lower()),
        "nan_inf_events": int(error is not None and "finite" in error["message"].lower()),
        "projection_failures": int(error is not None and "projection" in error["message"].lower()),
        "execution_error": error, "runtime_seconds": time.monotonic()-began,
        "manifest_content_sha256": manifest["content_sha256"],
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
    if (manifest.get("schema") != "semantic_stage2_full_loop_manifest_v1"
            or manifest.get("status") != "FROZEN_READY_FOR_EXECUTION"
            or manifest.get("final_test_forbidden") is not True
            or manifest.get("contains_periodic_timing") is not False):
        raise RuntimeError("invalid or deployment-unsafe Stage-2 full-loop manifest")
    config, cbf, flow_path = verify_assets(manifest)
    policy, provenance = load_policy(flow_path)
    reference = json.loads(ENVIRONMENT.read_text())
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    direct_model = DeterministicGphi(DIRECT_CHECKPOINT)
    eta_model = FixedDEtaPredictor(ETA_CHECKPOINT)
    output = HERE / "runs/full_loop" / manifest["split"] / manifest["policy_hash"]
    assigned = [task for task in manifest["tasks"] if int(task["task_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    new_tasks = new_steps = 0
    for task in assigned:
        path = output / f"task_{int(task['task_index']):04d}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("record_complete") and prior.get("manifest_content_sha256") == manifest["content_sha256"]:
                continue
            raise RuntimeError((path, "non-reusable prior result"))
        row = run_task(task, manifest, sample_action=sample_action, direct_model=direct_model,
                       eta_model=eta_model, config=config, cbf=cbf)
        atomic_json(path, row)
        if not row["record_complete"]:
            raise RuntimeError((task["source_id"], row["execution_error"]))
        new_tasks += 1
        new_steps += int(row["physical_transition_count"])
    atomic_json(output / f"runtime_shard{args.shard_index}.json", {
        "schema": "semantic_stage2_full_loop_runtime_v1",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "device": args.device, "new_continuations": new_tasks, "new_physical_steps": new_steps,
        "manifest_sha256": sha256(args.manifest), "jax_devices": [str(item) for item in jax.devices()],
    })


if __name__ == "__main__":
    main()

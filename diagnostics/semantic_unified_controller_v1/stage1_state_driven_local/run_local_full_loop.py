"""Run a frozen semantic S/L policy (or external H8 reference) on development WIDE."""

from __future__ import annotations

import argparse
import csv
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
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage1_state_driven_local"
for value in (str(SYSROOT), str(ROOT), str(PILOT), str(BASE), str(HERE)):
    while value in sys.path: sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(BASE), str(HERE)]

from pilot_common import DeterministicGphi  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
from local_common import (  # noqa: E402
    BASE, DIRECT_CHECKPOINT, DIRECT_SHA256, HERE, FrozenLocalHead,
    StateDrivenLocalMachine, sha256, verify_content_hash,
)
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402
from state_machine import AuthoritativeStepKernel  # noqa: E402


REFERENCE = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"
FROZEN = BASE / "controller_and_projection_hashes.json"
EXTERNAL_RESULTS = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1/per_episode_results.csv"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def classify(env: Any, error: dict[str, Any] | None) -> str:
    if error is not None: return "other"
    summary = env.summary()
    if summary["wall_collision"] or summary["agent_collision"]: return "collision"
    if summary["success"]: return "success"
    if summary["deadlock"]: return "deadlock"
    return "timeout"


def safety_reference(task: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if task["split"] == "development_external":
        with EXTERNAL_RESULTS.open(newline="") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row["condition"] == "Safety" and row["source_id"] == task["source_id"]
            ]
        if len(rows) != 1:
            raise RuntimeError((task["source_id"], "external Safety reference mismatch"))
        return {
            "outcome": rows[0]["outcome"],
            "success": rows[0]["success"].strip().lower() == "true",
        }
    path = BASE / "runs/safety_raw" / task["split"] / f"{task['source_id']}.json"
    row = json.loads(path.read_text())
    if not row.get("record_complete") or row.get("source_manifest_sha256") != manifest["source_manifest_sha256"]:
        raise RuntimeError((task["source_id"], "frozen Safety baseline mismatch"))
    return row


def run_task(task: dict[str, Any], manifest: dict[str, Any], *, sample_action: Any,
             direct_model: Any, config: Config, cbf: CBFConfig) -> dict[str, Any]:
    import jax

    env = GiveWayEnv(config)
    env.reset(np.asarray(task["initial_positions"], dtype=np.float64))
    episode_key = jax.random.fold_in(jax.random.PRNGKey(np.uint32(task["flow_root_seed"])), int(task["rollout_id"]))
    kernel = AuthoritativeStepKernel(
        env=env, config=config, cbf=cbf, episode_key=episode_key, sample_action=sample_action,
        feature_builder=StartupAwareFeatureBuilder(), project=project_velocity_with_retry,
    )
    controller = manifest["controller"]
    head = None
    if controller == "SEMANTIC_LOCAL":
        record = manifest["policy"]["local_head"]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]: raise RuntimeError("local head hash mismatch")
        head = FrozenLocalHead(path, threshold_logit=float(record["threshold_logit"]))
        machine = StateDrivenLocalMachine(kernel=kernel, local_head=head, direct_model=direct_model)
    records, error = [], None
    began = time.monotonic()
    try:
        if controller == "SEMANTIC_LOCAL":
            records = list(machine.run_to_terminal())
        else:
            while not env.done:
                context = kernel.prepare(need_feature=int(env.step_count) % 8 == 0)
                correction = np.zeros((2, 2), dtype=np.float64)
                second_retry = False
                decision = "SAFETY"
                if int(env.step_count) % 8 == 0:
                    predicted = np.asarray(direct_model(np.asarray(context.feature)[None]), dtype=np.float64)
                    correction = predicted[0].reshape(2, 2)
                    u_exec, second_retry = kernel.second_projection(context, correction)
                    decision = "LOCAL"
                else:
                    u_exec = context.u_safe
                done, info = kernel.execute(u_exec)
                records.append(type("Record", (), {
                    "step": context.step, "decision": decision, "u_safe": context.u_safe,
                    "u_exec": np.asarray(u_exec), "first_projection_retry": context.first_projection_retry,
                    "second_projection_retry": second_retry, "monitor_info": info,
                })())
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
    outcome = classify(env, error)
    event_steps = [int(row.step) for row in records if row.decision == "LOCAL"]
    spacings = [right-left for left, right in zip(event_steps, event_steps[1:])]
    safety = safety_reference(task, manifest)
    return {
        "schema": "semantic_local_full_loop_result_v1", "record_complete": error is None,
        "task_index": int(task["task_index"]), "source_id": task["source_id"],
        "root_source_id": task["root_source_id"], "split": task["split"],
        "controller": controller, "policy_hash": manifest["policy_hash"],
        "safety_outcome": safety["outcome"], "safety_success": bool(safety["success"]),
        "learned_outcome": outcome, "learned_success": outcome == "success",
        "learned_deadlock": outcome == "deadlock", "learned_timeout": outcome == "timeout",
        "learned_collision": outcome == "collision", "learned_other_failure": outcome == "other",
        "terminal_step": int(env.step_count),
        "jdef": config.dt * float(sum(np.sum((row.u_exec-row.u_safe) ** 2) for row in records)),
        "local_event_count": len(event_steps), "local_event_steps": event_steps,
        "local_event_spacings": spacings,
        "flow_sample_count": int(kernel.flow_sample_count),
        "physical_transition_count": int(kernel.physical_transition_count),
        "head_query_count": int(machine.head_query_count) if controller == "SEMANTIC_LOCAL" else 0,
        "direct_query_count": len(event_steps),
        "agent_collision_events": sum(int(row.monitor_info.get("agent_collision", False)) for row in records),
        "wall_collision_events": sum(int(row.monitor_info.get("wall_collision", False)) for row in records),
        "first_projection_retry_count": sum(row.first_projection_retry for row in records),
        "second_projection_retry_count": sum(row.second_projection_retry for row in records),
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
    import jax

    jax.config.update("jax_enable_x64", True); jax.config.update("jax_platform_name", args.device)
    manifest = json.loads(args.manifest.read_text()); verify_content_hash(manifest, args.manifest)
    if manifest.get("status") != "FROZEN_READY_FOR_EXECUTION" or manifest.get("final_test_forbidden") is not True:
        raise RuntimeError("invalid/deployment-unsafe full-loop manifest")
    if sha256(DIRECT_CHECKPOINT) != DIRECT_SHA256: raise RuntimeError("Direct-g hash mismatch")
    expected = {"local_common": HERE / "local_common.py", "full_loop_runner": HERE / "run_local_full_loop.py"}
    if set(manifest.get("implementation_hashes", {})) != set(expected):
        raise RuntimeError("implementation hash manifest incomplete")
    for name, path in expected.items():
        record = manifest["implementation_hashes"][name]
        if Path(record["path"]).resolve() != path.resolve() or sha256(path) != record["sha256"]:
            raise RuntimeError((name, "implementation changed after manifest freeze"))
    reference = json.loads(REFERENCE.read_text())
    config, cbf = Config(**reference["environment"]), CBFConfig(**reference["cbf"])
    frozen = json.loads(FROZEN.read_text()); flow_path = Path(frozen["flowbc"]["path"])
    if sha256(flow_path) != frozen["flowbc"]["sha256"]: raise RuntimeError("FlowBC hash mismatch")
    policy, provenance = load_policy(flow_path)
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    direct_model = DeterministicGphi(DIRECT_CHECKPOINT)
    output = HERE / "runs/full_loop" / manifest["split"] / manifest["policy_hash"]
    assigned = [task for task in manifest["tasks"] if int(task["task_index"]) % args.shard_count == args.shard_index]
    began = datetime.now(timezone.utc).isoformat(); new_tasks = new_steps = 0
    for task in assigned:
        path = output / f"task_{int(task['task_index']):04d}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("record_complete") and prior.get("manifest_content_sha256") == manifest["content_sha256"]: continue
            raise RuntimeError((path, "non-reusable prior result"))
        row = run_task(task, manifest, sample_action=sample_action, direct_model=direct_model, config=config, cbf=cbf)
        atomic_json(path, row)
        if not row["record_complete"]: raise RuntimeError((task["source_id"], row["execution_error"]))
        new_tasks += 1; new_steps += int(row["physical_transition_count"])
    atomic_json(output / f"runtime_shard{args.shard_index}.json", {
        "schema": "semantic_local_full_loop_runtime_v1", "started_utc": began,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "shard_index": args.shard_index,
        "shard_count": args.shard_count, "device": args.device,
        "new_continuations": new_tasks, "new_physical_steps": new_steps,
        "manifest_sha256": sha256(args.manifest), "jax_devices": [str(item) for item in jax.devices()],
    })


if __name__ == "__main__":
    main()

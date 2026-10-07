"""Collect outcome-blind ETA recovery-visited states for Stage-B exit labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
PLAN = HERE / "eta_recovery_state_collection_plan.json"
HASHES = HERE / "controller_and_projection_hashes.json"
REFERENCE = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"

sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT)]
from pilot_common import canonical_json_hash, sha256  # noqa: E402
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi  # noqa: E402
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import (  # noqa: E402
    StartupAwareFeatureBuilder,
)
from diagnostics.single_segment_recovery_training_v1.full_state_io import (  # noqa: E402
    audit_round_trip,
    restore_full_history,
    save_full_history,
)
from diagnostics.single_segment_recovery_training_v1.state_machine import (  # noqa: E402
    AuthoritativeStepKernel,
)
from diagnostics.success_basin_multimodality.exact_projector import (  # noqa: E402
    project_velocity_with_retry,
)
from single_integrator.cbf import CBFConfig  # noqa: E402
from single_integrator.environment import Config  # noqa: E402
from single_integrator.evaluate import load_policy  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def verify_semantic(payload: dict[str, Any], label: str) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_json_hash(body):
        raise RuntimeError((label, "semantic hash mismatch"))


def assert_action(context: Any, action: np.ndarray, max_speed: float, cbf: CBFConfig, label: str) -> tuple[float, float]:
    action = np.asarray(action, dtype=np.float64)
    linear_min = float(np.min(context.constraint_matrix @ action.reshape(4) - context.constraint_lower))
    speed_excess = float(np.max(np.linalg.norm(action, axis=-1) - max_speed))
    if not np.isfinite(action).all() or linear_min < -cbf.feasibility_tol or speed_excess > cbf.speed_tol:
        raise RuntimeError((label, "invalid action", linear_min, speed_excess))
    return linear_min, speed_excess


def arrays_exact(actual: np.ndarray, expected: Any, label: str) -> None:
    expected_array = np.asarray(expected, dtype=actual.dtype)
    if not np.array_equal(np.asarray(actual), expected_array):
        difference = float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))
        raise RuntimeError((label, "frozen entry quantity mismatch", difference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.shard_count not in (1, 2) or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard configuration")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    plan = json.loads(PLAN.read_text())
    verify_semantic(plan, "collection plan")
    if plan.get("status") != "FROZEN_BEFORE_RECOVERY_OUTCOMES" or plan.get("task_count") != 24:
        raise RuntimeError("collection plan is not frozen/complete")
    hashes = json.loads(HASHES.read_text())
    for label in ("flowbc", "environment", "projection_constraints", "projection_retry",
                  "event_priority_and_monitor", "startup_feature_builder", "eta_basis", "structured_eta_recovery"):
        record = hashes[label]
        if sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((label, "frozen hash mismatch"))
    reference = json.loads(REFERENCE.read_text())
    config, cbf = Config(**reference["environment"]), CBFConfig(**reference["cbf"])
    if config.max_steps != 850 or config.dt != 0.05:
        raise RuntimeError("official horizon/dt mismatch")
    policy, provenance = load_policy(Path(hashes["flowbc"]["path"]))
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    policy_sample = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    sample_action = lambda observation, key: np.asarray(policy_sample(jnp.asarray(observation), key), dtype=np.float64)
    eta_model = FixedDEtaPredictor(Path(hashes["structured_eta_recovery"]["path"]))
    builder = StartupAwareFeatureBuilder()
    tasks = plan["tasks"]
    assigned = [task for task in tasks if int(task["task_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    counters = {"new_continuations": 0, "new_physical_steps": 0, "available_states": 0,
                "unavailable_terminal_before_depth": 0, "eta_queries": 0}

    for task in assigned:
        task_index = int(task["task_index"])
        result_path = HERE / "runs/eta_recovery_state_rows" / f"task_{task_index:03d}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get("record_complete") and existing.get("collection_plan_sha256") == sha256(PLAN):
                continue
            raise RuntimeError((task["task_id"], "partial/mismatched cached record"))
        if sha256(Path(task["entry_state_file"])) != task["entry_state_sha256"]:
            raise RuntimeError((task["task_id"], "entry state hash mismatch"))
        env = restore_full_history(Path(task["entry_state_file"]), config)
        if env.step_count != int(task["entry_absolute_step"]):
            raise RuntimeError((task["task_id"], "entry absolute step mismatch"))
        episode_key = jax.random.fold_in(
            jax.random.PRNGKey(np.uint32(task["flow_root_seed"])), int(task["flow_rollout_id"])
        )
        kernel = AuthoritativeStepKernel(
            env=env, config=config, cbf=cbf, episode_key=episode_key,
            sample_action=sample_action, feature_builder=builder, project=project_velocity_with_retry,
        )
        began = time.monotonic()
        entry = kernel.prepare(need_feature=True)
        arrays_exact(entry.flow_key.astype(np.uint32), task["entry_flow_key"], "entry Flow key")
        arrays_exact(entry.u_flow, task["entry_u_flow"], "entry u_flow")
        arrays_exact(entry.u_safe, task["entry_u_safe"], "entry u_safe")
        arrays_exact(entry.feature, task["entry_feature"], "entry feature")
        assert_action(entry, entry.u_safe, config.max_speed, cbf, "entry first projection")
        eta_hat, eta_raw, eta_clipped_norm = eta_model.predict(entry.feature[None])
        eta_hat = np.asarray(eta_hat[0], dtype=np.float64)
        eta_raw = np.asarray(eta_raw[0], dtype=np.float64)
        eta_clipped_norm = np.asarray(eta_clipped_norm[0], dtype=np.float64)
        eta_controller = DiagnosticCorrector(DiagnosticPhi(*eta_hat.tolist()))
        counters["eta_queries"] += 1
        depth = int(task["recovery_depth_transitions"])
        first_retries = second_retries = 0
        max_linear_violation = max_speed_excess = 0.0
        jdef = 0.0
        terminal_info: dict[str, Any] | None = None
        executed = 0
        for recovery_index in range(depth):
            context = entry if recovery_index == 0 else kernel.prepare(need_feature=False)
            first_retries += int(context.first_projection_retry)
            linear_min, speed_excess = assert_action(context, context.u_safe, config.max_speed, cbf, "first projection")
            max_linear_violation = max(max_linear_violation, max(0.0, -linear_min))
            max_speed_excess = max(max_speed_excess, max(0.0, speed_excess))
            correction = np.asarray(
                eta_controller(np.asarray(context.observation, dtype=np.float64), context.u_safe, config.max_speed),
                dtype=np.float64,
            )
            u_exec, retry2 = kernel.second_projection(context, correction)
            second_retries += int(retry2)
            linear_min, speed_excess = assert_action(context, u_exec, config.max_speed, cbf, "second projection")
            max_linear_violation = max(max_linear_violation, max(0.0, -linear_min))
            max_speed_excess = max(max_speed_excess, max(0.0, speed_excess))
            jdef += config.dt * float(np.sum((u_exec - context.u_safe) ** 2))
            done, info = kernel.execute(u_exec)
            executed += 1
            if done:
                terminal_info = dict(info)
                break
        counters["new_continuations"] += 1
        counters["new_physical_steps"] += executed
        common = {
            "schema": "single_segment_eta_recovery_state_row_v1", "record_complete": True,
            "task_index": task_index, "task_id": task["task_id"],
            "entry_state_id": task["entry_state_id"], "root_source_id": task["root_source_id"],
            "split": task["split"], "entry_step": int(task["entry_absolute_step"]),
            "requested_recovery_transitions": depth, "executed_recovery_transitions": executed,
            "eta_latched": eta_hat.tolist(), "eta_raw_normalized": eta_raw.tolist(),
            "eta_clipped_normalized": eta_clipped_norm.tolist(),
            "eta_clipped": bool(np.any(eta_raw < 0.0) or np.any(eta_raw > 1.0)),
            "eta_query_count": 1, "eta_unchanged_while_active": True,
            "first_projection_retry_count": first_retries, "second_projection_retry_count": second_retries,
            "maximum_linear_violation": max_linear_violation, "maximum_speed_excess": max_speed_excess,
            "J_def_collection_prefix": jdef, "flow_sample_count": kernel.flow_sample_count,
            "physical_transition_count": kernel.physical_transition_count,
            "collection_plan_sha256": sha256(PLAN), "collection_plan_content_sha256": plan["content_sha256"],
            "runtime_seconds": time.monotonic() - began,
        }
        if terminal_info is not None or env.done:
            counters["unavailable_terminal_before_depth"] += 1
            common.update({
                "available": False, "reason": "terminal_at_or_before_predeclared_recovery_depth",
                "terminal_global_step": int(env.step_count),
                "terminal_event": terminal_info.get("termination") if terminal_info else "terminal",
                "no_replacement": True,
            })
            atomic_json(result_path, common)
            print(json.dumps({"task": task_index, "available": False, "step": env.step_count}), flush=True)
            continue

        # Exact recovery-visited pre-action state for the exit decision.
        decision = kernel.prepare(need_feature=True)
        assert_action(decision, decision.u_safe, config.max_speed, cbf, "exit-state first projection")
        stem = f"eta_exit_{task['entry_state_id']}_d{depth:02d}"
        state_path = HERE / "states/recovery_anchors/ETA" / task["split"] / f"{stem}.npz"
        input_path = HERE / "decision_inputs/recovery_anchors/ETA" / task["split"] / f"{stem}.npz"
        save_full_history(state_path, env)
        round_trip = audit_round_trip(state_path, env, config)
        if not round_trip["passed"] or round_trip["history_start_step"] != 0:
            raise RuntimeError((task["task_id"], "derived-state round-trip failed", round_trip))
        atomic_npz(
            input_path,
            feature=np.asarray(decision.feature, dtype=np.float64),
            u_flow=np.asarray(decision.u_flow, dtype=np.float64),
            u_safe=np.asarray(decision.u_safe, dtype=np.float64),
            flow_step_key=np.asarray(decision.flow_key, dtype=np.uint32),
            observation=np.asarray(decision.observation, dtype=np.float64),
            eta_latched=eta_hat,
            entry_step=np.asarray(task["entry_absolute_step"], dtype=np.int64),
            recovery_transitions=np.asarray(depth, dtype=np.int64),
            absolute_step=np.asarray(env.step_count, dtype=np.int64),
        )
        row = {
            **common, "available": True, "state_id": stem,
            "state_file": str(state_path), "state_sha256": sha256(state_path),
            "decision_input_file": str(input_path), "decision_input_sha256": sha256(input_path),
            "absolute_step": int(env.step_count), "feature": np.asarray(decision.feature).tolist(),
            "current_u_flow": np.asarray(decision.u_flow).tolist(),
            "current_u_safe": np.asarray(decision.u_safe).tolist(),
            "current_flow_key_data": np.asarray(decision.flow_key, dtype=np.uint32).tolist(),
            "current_flow_realization_id": (
                f"root={task['flow_root_seed']}|rollout={task['flow_rollout_id']}|"
                f"absolute_step={env.step_count}|key={np.asarray(decision.flow_key, dtype=np.uint32).tolist()}"
            ),
            "flow_root_seed": int(task["flow_root_seed"]), "flow_rollout_id": int(task["flow_rollout_id"]),
            "entry_current_flow_reused": True, "current_flow_prepared_once_not_executed": True,
            "mode": "RECOVERY", "recovery_used": True, "exit_step": None,
            "recovery_transitions": depth, "round_trip_audit": round_trip,
        }
        atomic_json(result_path, row)
        counters["available_states"] += 1
        print(json.dumps({"task": task_index, "available": True, "step": env.step_count, "depth": depth}), flush=True)

    atomic_json(HERE / "runs" / f"eta_recovery_state_runtime_shard{args.shard_index}.json", {
        "schema": "single_segment_eta_recovery_state_runtime_v1",
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        **counters, "device": args.device, "jax_devices": [str(device) for device in jax.devices()],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "collection_plan_sha256": sha256(PLAN),
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XLA_FLAGS"
        )},
    })


if __name__ == "__main__":
    main()

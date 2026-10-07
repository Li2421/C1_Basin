"""Collect frozen Safety trajectories and predeclared generic decision anchors.

Only train/validation/calibration source groups are evaluated.  The reserved
final-test sources are deliberately unreachable from this executable.
"""

from __future__ import annotations

import argparse
import hashlib
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
SOURCE_MANIFEST = HERE / "source_split_manifest.json"
PROTOCOL = HERE / "protocol.json"
HASHES = HERE / "controller_and_projection_hashes.json"
REFERENCE = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"
ALLOWED_SPLITS = ("train", "validation", "calibration")

# Frozen import order: the authoritative Toy Give-Way package must win.
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT)]
from pilot_common import canonical_json_hash, sha256  # noqa: E402
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import (  # noqa: E402
    StartupAwareFeatureBuilder,
)
from diagnostics.gphi_training_dataset_v2.build_states import restore_full  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import (  # noqa: E402
    project_velocity_with_retry,
)
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal  # noqa: E402
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


def atomic_full_state(path: Path, env: GiveWayEnv) -> None:
    """Persist the complete real monitor history, not the tail-41 feature view."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    history = np.asarray(env.distance_history, dtype=np.float64)
    if history.shape != (env.step_count + 1, 2) or not np.isfinite(history).all():
        raise RuntimeError(("invalid complete monitor history", history.shape, env.step_count))
    np.savez_compressed(
        temporary,
        positions=np.asarray(env.positions, dtype=np.float64),
        velocities=np.asarray(env.velocities, dtype=np.float64),
        step=np.asarray(env.step_count),
        error_history=history,
        history_start_step=np.asarray(0),
        candidate_since=np.asarray(-1 if env.candidate_since is None else env.candidate_since),
        stuck_timer=np.asarray(env.stuck_timer),
        max_stuck_timer=np.asarray(env.max_stuck_timer),
        ever_candidate_deadlock=np.asarray(env.ever_candidate_deadlock),
        first_success_step=np.asarray(-1 if env.first_success_step is None else env.first_success_step),
        first_deadlock_step=np.asarray(-1 if env.first_deadlock_step is None else env.first_deadlock_step),
        first_wall_collision_step=np.asarray(-1 if env.first_wall_collision_step is None else env.first_wall_collision_step),
        first_agent_collision_step=np.asarray(-1 if env.first_agent_collision_step is None else env.first_agent_collision_step),
        done=np.asarray(env.done),
        complete_real_history=np.asarray(True),
    )
    os.replace(temporary, path)


def assert_file(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError((label, "hash mismatch", str(path), expected, actual))


def assert_manifests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = json.loads(SOURCE_MANIFEST.read_text())
    body = {key: value for key, value in source.items() if key != "content_sha256"}
    if canonical_json_hash(body) != source["content_sha256"]:
        raise RuntimeError("source split manifest semantic hash mismatch")
    if source.get("outcomes_observed_by_this_script") is not False:
        raise RuntimeError("source manifest was not frozen outcome-blind")
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(PROTOCOL) != source["protocol_sha256"]:
        raise RuntimeError("source/protocol hash mismatch")
    if protocol["official_horizon_steps"] != 850 or protocol["dt_seconds"] != 0.05:
        raise RuntimeError("official horizon/dt mismatch")
    hashes = json.loads(HASHES.read_text())
    if hashes.get("status") != "PASS":
        raise RuntimeError("frozen implementation audit not PASS")
    for label, record in hashes.items():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            assert_file(Path(record["path"]), record["sha256"], label)
    reference = json.loads(REFERENCE.read_text())
    if reference["environment"]["max_steps"] != 850 or reference["environment"]["dt"] != 0.05:
        raise RuntimeError("reference environment horizon/dt mismatch")
    return source, protocol, hashes, reference


def classify(env: GiveWayEnv, error: dict[str, Any] | None) -> str:
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


def rows_for_collection(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        for row in source["sources"][split]:
            rows.append(row)
    if len(rows) != 80:
        raise RuntimeError(("expected 80 non-test sources", len(rows)))
    final_ids = {row["source_id"] for row in source["sources"]["final_test"]}
    if any(row["source_id"] in final_ids for row in rows):
        raise RuntimeError("final-test source leakage")
    return rows


def save_anchor(
    *, env: GiveWayEnv, source_row: dict[str, Any], slot: int, requested_step: int,
    step_key: np.ndarray, u_flow: np.ndarray, u_safe: np.ndarray,
    feature: np.ndarray, feature_metadata: dict[str, Any], config: Config,
) -> dict[str, Any]:
    split = source_row["split"]
    source_id = source_row["source_id"]
    stem = f"{source_id}_a{slot}_t{requested_step:04d}"
    state_path = HERE / "states/safety_anchors" / split / f"{stem}.npz"
    input_path = HERE / "decision_inputs/safety_anchors" / split / f"{stem}.npz"
    atomic_full_state(state_path, env)
    restored = restore_full(state_path, config)
    source_history = np.asarray(env.distance_history, dtype=np.float64)
    restored_history = np.asarray(restored.distance_history, dtype=np.float64)
    if source_history.shape != (env.step_count + 1, 2) or not np.isfinite(source_history).all():
        raise RuntimeError((stem, "source monitor history incomplete", source_history.shape))
    if restored_history.shape != source_history.shape or not np.isfinite(restored_history).all():
        raise RuntimeError((stem, "restored monitor history incomplete", restored_history.shape, source_history.shape))
    numeric_diffs = {
        "positions": float(np.max(np.abs(restored.positions - env.positions))),
        "velocities": float(np.max(np.abs(restored.velocities - env.velocities))),
        "history": float(np.max(np.abs(restored_history - source_history))),
        "stuck_timer": float(abs(restored.stuck_timer - env.stuck_timer)),
        "max_stuck_timer": float(abs(restored.max_stuck_timer - env.max_stuck_timer)),
    }
    scalar_equal = (
        restored.step_count == env.step_count
        and restored.candidate_since == env.candidate_since
        and restored.ever_candidate_deadlock == env.ever_candidate_deadlock
        and restored.first_success_step == env.first_success_step
        and restored.first_deadlock_step == env.first_deadlock_step
        and restored.first_wall_collision_step == env.first_wall_collision_step
        and restored.first_agent_collision_step == env.first_agent_collision_step
        and restored.done == env.done
    )
    if max(numeric_diffs.values()) > 1e-12 or not scalar_equal or restored.done:
        raise RuntimeError((stem, "augmented-state restore mismatch", numeric_diffs, scalar_equal, restored.done))
    atomic_npz(
        input_path,
        feature=np.asarray(feature, dtype=np.float64),
        u_flow=np.asarray(u_flow, dtype=np.float64),
        u_safe=np.asarray(u_safe, dtype=np.float64),
        flow_step_key=np.asarray(step_key, dtype=np.uint32),
        observation=np.asarray(env.observation(), dtype=np.float64),
        requested_global_step=np.asarray(requested_step, dtype=np.int64),
    )
    return {
        "schema": "single_segment_generic_safety_anchor_v1",
        "state_id": stem,
        "root_source_id": source_id,
        "split": split,
        "episode_index": int(source_row["episode_index"]),
        "rollout_id": int(source_row["rollout_id"]),
        "anchor_slot": int(slot),
        "requested_global_step": int(requested_step),
        "actual_global_step": int(env.step_count),
        "available": True,
        "mode": "SAFETY_BEFORE",
        "state_file": str(state_path),
        "state_sha256": sha256(state_path),
        "decision_input_file": str(input_path),
        "decision_input_sha256": sha256(input_path),
        "feature_dimension": int(feature.shape[0]),
        "feature_metadata": feature_metadata,
        "flow_root_seed": int(source_row["flow_root_seed"]),
        "flow_rollout_id": int(source_row["rollout_id"]),
        "flow_global_step": int(requested_step),
        "flow_step_key": np.asarray(step_key, dtype=np.uint32).tolist(),
        "flow_rng_semantics": "episode_key=fold_in(PRNGKey(flow_root_seed),rollout_id); step_key=fold_in(episode_key,absolute_step)",
        "single_flow_sample_at_decision": True,
        "state_restore_max_abs_difference": max(numeric_diffs.values()),
        "state_restore_scalar_equal": scalar_equal,
        "complete_real_monitor_history": True,
        "monitor_history_start_step": 0,
        "monitor_history_length": int(len(source_history)),
        "selection_outcome_blind": True,
        "selection_location_independent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.shard_count not in (1, 2):
        raise ValueError("source collection may use at most two shards")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")

    # Import JAX only after all process-level thread limits are inherited.
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    source, _, hashes, reference = assert_manifests()
    config = Config(**reference["environment"])
    cbf = CBFConfig(**reference["cbf"])
    if config.max_steps != 850 or config.dt != 0.05:
        raise RuntimeError("runtime horizon changed")
    flow_path = Path(hashes["flowbc"]["path"])
    policy, provenance = load_policy(flow_path)
    if not provenance or provenance["evaluation_environment"] != reference["environment"]:
        raise RuntimeError("FlowBC evaluation environment mismatch")
    sample = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    builder = StartupAwareFeatureBuilder()
    feature_schema_sha = hashlib.sha256(
        json.dumps(builder.schema, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    manifest_sha = sha256(SOURCE_MANIFEST)
    all_rows = rows_for_collection(source)
    assigned = [row for index, row in enumerate(all_rows) if index % args.shard_count == args.shard_index]
    started_utc = datetime.now(timezone.utc).isoformat()
    counters = {"new_source_rollouts": 0, "new_physical_steps": 0, "materialized_anchors": 0,
                "unavailable_anchors": 0, "reused_complete_sources": 0}

    for source_row in assigned:
        split = source_row["split"]
        source_id = source_row["source_id"]
        result_path = HERE / "runs/safety_raw" / split / f"{source_id}.json"
        trajectory_path = HERE / "runs/safety_trajectories" / split / f"{source_id}.npz"
        anchors_path = HERE / "runs/safety_anchor_rows" / split / f"{source_id}.json"
        if result_path.is_file() and trajectory_path.is_file() and anchors_path.is_file():
            existing = json.loads(result_path.read_text())
            if existing.get("record_complete") and existing.get("source_manifest_sha256") == manifest_sha:
                counters["reused_complete_sources"] += 1
                continue
            raise RuntimeError((source_id, "partial or mismatched cached source"))
        if any(path.exists() for path in (result_path, trajectory_path, anchors_path)):
            raise RuntimeError((source_id, "partial cached source artifacts; refuse overwrite"))

        env = GiveWayEnv(config)
        initial = np.asarray(source_row["initial_positions"], dtype=np.float64)
        env.reset(initial)
        episode_key = jax.random.fold_in(
            jax.random.PRNGKey(np.uint32(source_row["flow_root_seed"])), int(source_row["rollout_id"])
        )
        requests = {int(step): slot for slot, step in enumerate(source_row["requested_anchor_steps"])}
        anchor_rows: list[dict[str, Any]] = []
        names = (
            "step", "flow_step_key", "positions_before", "velocities_before", "positions_after",
            "u_flow", "u_safe", "goal_errors_after", "stuck_timer_after", "max_stuck_timer_after",
            "candidate_since_after", "ever_candidate_deadlock_after", "first_success_step_after",
            "first_deadlock_step_after", "first_wall_collision_step_after", "first_agent_collision_step_after",
            "event", "wall_collision", "agent_collision", "first_projection_retry", "linear_min", "speed_excess",
        )
        buffers: dict[str, list[Any]] = {name: [] for name in names}
        error: dict[str, Any] | None = None
        projection_failures = invalid_actions = nan_inf_events = 0
        wall_events = agent_events = 0
        began = time.monotonic()
        while not env.done:
            try:
                step = int(env.step_count)
                observation = np.asarray(env.observation(), dtype=np.float32)
                step_key = jax.random.fold_in(episode_key, step)
                raw = np.asarray(sample(jnp.asarray(observation), step_key), dtype=np.float64)
                u_flow = bounded_nominal(raw, config.max_speed)
                A, lower, _ = barrier_constraints(env.snapshot(), cbf)
                u_safe, _, retry, _ = project_velocity_with_retry(u_flow, A, lower, config.max_speed, cbf)
                linear_min = float(np.min(A @ u_safe.reshape(4) - lower))
                speed_excess = float(np.max(np.linalg.norm(u_safe, axis=-1) - config.max_speed))
                finite = bool(np.isfinite(u_safe).all())
                if not finite:
                    nan_inf_events += 1
                if not finite or linear_min < -cbf.feasibility_tol or speed_excess > cbf.speed_tol:
                    invalid_actions += 1
                    raise RuntimeError(("invalid Safety action", finite, linear_min, speed_excess))

                # Materialize the exact pre-action state and decision quantities.
                # The already-computed sample is reused for the physical transition.
                if step in requests:
                    feature, _ = builder.build(env, {"u_flow": u_flow, "u_safe": u_safe}, config, cbf)
                    anchor_rows.append(save_anchor(
                        env=env, source_row=source_row, slot=requests[step], requested_step=step,
                        step_key=np.asarray(step_key), u_flow=u_flow, u_safe=u_safe, feature=feature,
                        feature_metadata=builder.history_metadata(env), config=config,
                    ))
                    counters["materialized_anchors"] += 1

                before = env.positions.copy()
                velocity_before = env.velocities.copy()
                _, _, _, info = env.step(u_safe)
                wall_events += int(info["wall_collision"])
                agent_events += int(info["agent_collision"])
                values = {
                    "step": step, "flow_step_key": np.asarray(step_key, dtype=np.uint32),
                    "positions_before": before, "velocities_before": velocity_before,
                    "positions_after": env.positions.copy(), "u_flow": u_flow, "u_safe": u_safe,
                    "goal_errors_after": np.asarray(info["goal_errors"], dtype=np.float64),
                    "stuck_timer_after": env.stuck_timer, "max_stuck_timer_after": env.max_stuck_timer,
                    "candidate_since_after": -1 if env.candidate_since is None else env.candidate_since,
                    "ever_candidate_deadlock_after": env.ever_candidate_deadlock,
                    "first_success_step_after": -1 if env.first_success_step is None else env.first_success_step,
                    "first_deadlock_step_after": -1 if env.first_deadlock_step is None else env.first_deadlock_step,
                    "first_wall_collision_step_after": -1 if env.first_wall_collision_step is None else env.first_wall_collision_step,
                    "first_agent_collision_step_after": -1 if env.first_agent_collision_step is None else env.first_agent_collision_step,
                    "event": info["termination"], "wall_collision": info["wall_collision"],
                    "agent_collision": info["agent_collision"], "first_projection_retry": retry,
                    "linear_min": linear_min, "speed_excess": speed_excess,
                }
                for name in names:
                    buffers[name].append(values[name])
            except Exception as exc:
                projection_failures += int("projection" in str(exc).lower() or "solver" in type(exc).__name__.lower())
                error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
                break

        materialized_steps = {int(row["requested_global_step"]) for row in anchor_rows}
        for slot, requested_step in enumerate(source_row["requested_anchor_steps"]):
            if int(requested_step) not in materialized_steps:
                anchor_rows.append({
                    "schema": "single_segment_generic_safety_anchor_v1",
                    "state_id": f"{source_id}_a{slot}_t{int(requested_step):04d}",
                    "root_source_id": source_id, "split": split,
                    "episode_index": int(source_row["episode_index"]),
                    "rollout_id": int(source_row["rollout_id"]), "anchor_slot": int(slot),
                    "requested_global_step": int(requested_step), "actual_global_step": None,
                    "available": False, "reason": "Safety_terminated_before_predeclared_absolute_step",
                    "terminal_step": int(env.step_count), "no_replacement": True,
                    "selection_outcome_blind": True, "selection_location_independent": True,
                })
                counters["unavailable_anchors"] += 1
        anchor_rows.sort(key=lambda row: row["anchor_slot"])
        atomic_json(anchors_path, {
            "schema": "single_segment_source_anchor_rows_v1", "record_complete": True,
            "root_source_id": source_id, "split": split, "anchors": anchor_rows,
            "source_manifest_sha256": manifest_sha,
        })

        arrays = {key: np.asarray(value) for key, value in buffers.items()}
        arrays["initial_positions"] = initial
        arrays["initial_goal_errors"] = np.linalg.norm(env.goals - initial, axis=-1)
        arrays["flow_root_seed"] = np.asarray(source_row["flow_root_seed"], dtype=np.int64)
        arrays["flow_rollout_id"] = np.asarray(source_row["rollout_id"], dtype=np.int64)
        atomic_npz(trajectory_path, **arrays)
        result = classify(env, error)
        row = {
            "schema": "single_segment_safety_source_result_v1", "record_complete": True,
            "source_id": source_id, "split": split, "episode_index": int(source_row["episode_index"]),
            "rollout_id": int(source_row["rollout_id"]), "initial_positions": source_row["initial_positions"],
            "outcome": result, "success": result == "success", "timeout": result == "timeout",
            "deadlock": result == "deadlock", "collision": result == "collision", "other_failure": result == "other",
            "terminal_step": int(env.step_count), "terminal_time_sec": env.step_count * config.dt,
            "execution_error": error, "wall_collision_events": wall_events, "agent_collision_events": agent_events,
            "projection_failures": projection_failures, "invalid_actions": invalid_actions,
            "nan_inf_events": nan_inf_events, "first_projection_retry_count": int(np.sum(buffers["first_projection_retry"])),
            "requested_anchor_count": len(source_row["requested_anchor_steps"]),
            "materialized_anchor_count": sum(bool(item["available"]) for item in anchor_rows),
            "unavailable_anchor_count": sum(not bool(item["available"]) for item in anchor_rows),
            "source_manifest_sha256": manifest_sha, "source_manifest_content_sha256": source["content_sha256"],
            "protocol_sha256": sha256(PROTOCOL), "hash_audit_sha256": sha256(HASHES),
            "flow_root_seed": int(source_row["flow_root_seed"]),
            "flow_semantics": "episode/absolute-step stateless fold_in; no policy bookkeeping consumes Flow RNG",
            "trajectory_file": str(trajectory_path), "trajectory_sha256": sha256(trajectory_path),
            "anchor_rows_file": str(anchors_path), "anchor_rows_sha256": sha256(anchors_path),
            "feature_schema_sha256": feature_schema_sha,
            "rollout_started_after_manifest_freeze": started_utc > source["frozen_utc"],
            "runtime_seconds": time.monotonic() - began,
        }
        atomic_json(result_path, row)
        counters["new_source_rollouts"] += 1
        counters["new_physical_steps"] += int(env.step_count)
        print(json.dumps({"source": source_id, "split": split, "outcome": result,
                          "steps": env.step_count, "anchors": row["materialized_anchor_count"]}), flush=True)

    atomic_json(HERE / "runs" / f"source_collection_runtime_shard{args.shard_index}.json", {
        "schema": "single_segment_source_collection_runtime_shard_v1",
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        **counters, "device": args.device, "jax_devices": [str(device) for device in jax.devices()],
        "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_manifest_sha256": manifest_sha,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XLA_FLAGS"
        )},
    })


if __name__ == "__main__":
    main()

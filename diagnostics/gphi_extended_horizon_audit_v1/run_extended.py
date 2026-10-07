"""Replay only frozen WIDE timeouts to 1700 steps, with exact prefix audit."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_extended_horizon_audit_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
TRAINING = ROOT / "diagnostics/gphi_startup_warm_pareto_v1"
BASE_TRAINING = ROOT / "diagnostics/gphi_training_startup_complete_v1"
CHECKPOINT = TRAINING / "best_balanced_checkpoint.npz"
NORMALIZATION = BASE_TRAINING / "artifacts/normalization.json"
SAMPLES = DATASET / "samples.npz"
BENCHMARK = WIDE / "frozen_benchmark_manifest.json"
WIDE_CONFIG = WIDE / "wide_cadence_config.json"
EXPECTED_CHECKPOINT = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
ORDER = ("Safety", "H=1", "H=4", "H=8", "H=16")
SLUG = {"Safety": "safety", "H=1": "h1", "H=4": "h4", "H=8": "h8", "H=16": "h16"}
EXPECTED_TIMEOUTS = {"Safety": 51, "H=1": 48, "H=4": 18, "H=8": 13, "H=16": 27}
ORIGINAL_HORIZON = 850
EXTENDED_HORIZON = 1700

# Keep both helper directories importable, but the WIDE runner must win the
# generic ``run_evaluation`` module name.
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(WIDE))

import run_evaluation as wide_runner  # noqa: E402
from pilot_common import (  # noqa: E402
    DeterministicGphi,
    TrainingReference,
    assert_frozen_sources,
    audit_startup_training_artifacts,
    canonical_json_hash,
    load_environment_config,
    sha256,
    write_json,
)


PREFIX_FIELDS = (
    "step", "flow_step_key", "positions_before", "positions_after",
    "u_flow", "u_safe", "g_hat", "raw_second_target", "u_exec",
    "cadence_scheduled", "raw_correction_norm",
    "executed_correction_norm", "projection_rewrite_norm",
    "first_linear_min", "executed_linear_min", "executed_speed_excess",
    "first_status", "second_status", "first_retry", "second_retry",
    "feature", "real_history_length", "candidate_since", "stuck_timer",
    "max_stuck_timer", "wall_collision", "agent_collision",
)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_tasks(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for condition in ORDER:
        slug = SLUG[condition]
        count = 0
        for episode in benchmark["episodes"]:
            index = int(episode["episode_index"])
            raw_path = WIDE / "runs/production/raw" / slug / f"episode_{index:04d}.json"
            raw = load_json(raw_path)
            if raw.get("outcome") != "timeout":
                continue
            trajectory_path = WIDE / raw["trajectory_file"]
            if not trajectory_path.is_file() or sha256(trajectory_path) != raw["trajectory_sha256"]:
                raise RuntimeError(("invalid original trajectory", raw_path))
            if int(raw["episode_steps"]) != ORIGINAL_HORIZON:
                raise RuntimeError(("timeout length mismatch", raw_path))
            tasks.append({
                "task_index": len(tasks),
                "condition": condition,
                "slug": slug,
                "episode": episode,
                "original_record_path": raw_path,
                "original_record_sha256": sha256(raw_path),
                "original_trajectory_path": trajectory_path,
                "original_trajectory_sha256": raw["trajectory_sha256"],
                "original_row": raw,
            })
            count += 1
        counts[condition] = count
    if counts != EXPECTED_TIMEOUTS:
        raise RuntimeError(("original timeout counts changed", counts, EXPECTED_TIMEOUTS))
    if len(tasks) != sum(EXPECTED_TIMEOUTS.values()):
        raise RuntimeError("unexpected total timeout count")
    return tasks


def prefix_audit(original_path: Path, extended: dict[str, np.ndarray]) -> dict[str, Any]:
    exact: dict[str, bool] = {}
    max_error: dict[str, float] = {}
    with np.load(original_path, allow_pickle=False) as old:
        for field in PREFIX_FIELDS:
            if field not in old or field not in extended:
                raise RuntimeError(("prefix field missing", field, original_path))
            old_value = np.asarray(old[field])
            new_value = np.asarray(extended[field])
            if len(old_value) != ORIGINAL_HORIZON or len(new_value) < ORIGINAL_HORIZON:
                raise RuntimeError(("prefix length mismatch", field, old_value.shape, new_value.shape))
            new_prefix = new_value[:ORIGINAL_HORIZON]
            exact[field] = bool(np.array_equal(old_value, new_prefix))
            if np.issubdtype(old_value.dtype, np.number):
                max_error[field] = float(np.max(np.abs(old_value.astype(np.float64) - new_prefix.astype(np.float64))))
        old_event = np.asarray(old["event"]).astype(str)
        new_event = np.asarray(extended["event"][:ORIGINAL_HORIZON]).astype(str)
        event_preterminal_exact = bool(np.array_equal(old_event[:-1], new_event[:-1]))
        expected_horizon_event_change = bool(old_event[-1] == "timeout" and new_event[-1] == "running")
        if not all(exact.values()) or not event_preterminal_exact or not expected_horizon_event_change:
            raise RuntimeError({
                "prefix_exact": exact,
                "event_preterminal_exact": event_preterminal_exact,
                "old_last_event": old_event[-1],
                "new_last_event": new_event[-1],
            })
    return {
        "status": "PASS",
        "steps_0_through_849_controller_state_monitor_exact": True,
        "exact_fields": exact,
        "max_absolute_error": max_error,
        "event_steps_0_through_848_exact": event_preterminal_exact,
        "step_849_expected_timeout_to_running_only": expected_horizon_event_change,
        "maximum_numeric_prefix_error": max(max_error.values(), default=0.0),
    }


def progress_diagnostics(arrays: dict[str, np.ndarray], dt: float) -> dict[str, Any]:
    start = ORIGINAL_HORIZON
    positions_before = np.asarray(arrays["positions_before"], dtype=np.float64)
    positions_after = np.asarray(arrays["positions_after"], dtype=np.float64)
    u_exec = np.asarray(arrays["u_exec"], dtype=np.float64)
    executed_norm = np.asarray(arrays["executed_correction_norm"], dtype=np.float64)
    if len(positions_before) <= start:
        raise RuntimeError("extended trajectory contains no post-850 steps")
    goals = np.asarray([[2.29, 0.0], [-2.29, 0.0]], dtype=np.float64)
    p0 = positions_before[start]
    p1 = positions_after[-1]
    error0 = np.linalg.norm(goals - p0, axis=1)
    error1 = np.linalg.norm(goals - p1, axis=1)
    displacement = np.linalg.norm(p1 - p0, axis=1)
    inter = np.linalg.norm(positions_after[start:, 0] - positions_after[start:, 1], axis=1)
    speed = np.linalg.norm(u_exec[start:], axis=2)
    candidate_since = np.asarray(arrays["candidate_since"])[start:]
    stuck = np.asarray(arrays["stuck_timer"], dtype=np.float64)[start:]
    return {
        "extension_steps_observed": int(len(positions_before) - start),
        "goal_error_start_each": error0.tolist(),
        "goal_error_final_each": error1.tolist(),
        "goal_error_reduction_each": (error0 - error1).tolist(),
        "goal_error_reduction_sum": float(np.sum(error0 - error1)),
        "displacement_each": displacement.tolist(),
        "displacement_sum": float(np.sum(displacement)),
        "agent_speed_mean": float(np.mean(speed)),
        "agent_speed_median": float(np.median(speed)),
        "agent_speed_max": float(np.max(speed)),
        "inter_agent_distance_start": float(inter[0]),
        "inter_agent_distance_final": float(inter[-1]),
        "inter_agent_distance_min": float(np.min(inter)),
        "inter_agent_distance_max": float(np.max(inter)),
        "candidate_deadlock_fraction": float(np.mean(candidate_since >= 0)),
        "maximum_stuck_timer": float(np.max(stuck)),
        "effective_correction_fraction": float(np.mean(executed_norm[start:] > 1e-6)),
        "additional_J_def": float(dt * np.sum(executed_norm[start:] ** 2)),
    }


def existing_valid(record_path: Path, expected: dict[str, Any]) -> bool:
    if not record_path.is_file():
        return False
    try:
        row = load_json(record_path)
    except Exception:
        return False
    if row.get("record_complete") is not True or row.get("prefix_audit", {}).get("status") != "PASS":
        return False
    if any(row.get(key) != value for key, value in expected.items()):
        return False
    trajectory = HERE / row.get("trajectory_file", "")
    return trajectory.is_file() and sha256(trajectory) == row.get("trajectory_sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--stop-task", type=int, default=sum(EXPECTED_TIMEOUTS.values()))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("checkpoint hash mismatch")
    frozen_sources = assert_frozen_sources()
    extra_sources = wide_runner._audit_extra_sources()
    benchmark = wide_runner._load_frozen_benchmark(BENCHMARK, True)
    original_config_payload = load_json(WIDE_CONFIG)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv
    from single_integrator.evaluate import load_policy
    from single_integrator.outcomes import first_event

    resolved = {
        "projection_constraints": str(Path(inspect.getsourcefile(barrier_constraints)).resolve()),
        "environment": str(Path(inspect.getsourcefile(GiveWayEnv)).resolve()),
        "retry_projector": str(Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()),
        "feature_builder": str(Path(inspect.getsourcefile(StartupAwareFeatureBuilder)).resolve()),
        "outcome_classifier": str(Path(inspect.getsourcefile(first_event)).resolve()),
    }
    if any(not Path(path).is_file() for path in resolved.values()):
        raise RuntimeError(("resolved source missing", resolved))

    audit_startup_training_artifacts(CHECKPOINT, NORMALIZATION, require_startup_complete=True)
    model = DeterministicGphi(CHECKPOINT, NORMALIZATION)
    reference = TrainingReference(SAMPLES, model)
    original_environment = load_environment_config(DATASET)
    if original_environment != benchmark["environment"] or original_environment["max_steps"] != ORIGINAL_HORIZON:
        raise RuntimeError("original environment/benchmark mismatch")
    extended_environment = dict(original_environment)
    extended_environment["max_steps"] = EXTENDED_HORIZON
    config = Config(**extended_environment)
    feature_config = Config(**original_environment)

    class Frozen850FeatureBuilder(StartupAwareFeatureBuilder):
        """Keep the deployed 850-step feature coordinate while extending termination."""

        def build(self, env: Any, first: dict, _extended_config: Any, cbf_config: Any):
            return super().build(env, first, feature_config, cbf_config)
    cbf = CBFConfig()
    if cbf.to_dict() != benchmark["cbf"]:
        raise RuntimeError("CBF config mismatch")
    policy, provenance = load_policy(wide_runner.FLOW_CHECKPOINT)
    if not provenance or provenance.get("evaluation_environment") != original_environment:
        raise RuntimeError("Flow checkpoint environment mismatch")
    sample_action = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])

    tasks = build_tasks(benchmark)
    if not (0 <= args.start_task < len(tasks)) or not (args.start_task < args.stop_task <= len(tasks)):
        raise ValueError(("invalid task slice", args.start_task, args.stop_task, len(tasks)))
    config_payload = {
        "schema": "gphi_extended_horizon_config_v1",
        "only_change": {"max_steps": [ORIGINAL_HORIZON, EXTENDED_HORIZON]},
        "original_wide_config_path": str(WIDE_CONFIG),
        "original_wide_config_sha256": sha256(WIDE_CONFIG),
        "original_wide_config_semantic_sha256": original_config_payload["content_sha256"],
        "benchmark_manifest_path": str(BENCHMARK),
        "benchmark_manifest_sha256": sha256(BENCHMARK),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "normalization_sha256": sha256(NORMALIZATION),
        "original_environment": original_environment,
        "extended_environment": extended_environment,
        "feature_builder_environment": original_environment,
        "feature_time_coordinate_frozen_at_850": True,
        "cadence": wide_runner.CONTROLLER_CADENCE,
        "flow_key_semantics": wide_runner.FLOW_KEY_SEMANTICS,
        "frozen_sources": frozen_sources,
        "extra_frozen_sources": extra_sources,
        "resolved_sources": resolved,
        "timeout_counts": EXPECTED_TIMEOUTS,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    config_payload["content_sha256"] = canonical_json_hash(config_payload)
    config_path = HERE / "extended_config.json"
    if config_path.is_file():
        if load_json(config_path).get("content_sha256") != config_payload["content_sha256"]:
            raise RuntimeError("extended config changed")
    else:
        write_json(config_path, config_payload)

    completed = skipped = 0
    for task in tasks[args.start_task:args.stop_task]:
        condition = task["condition"]
        slug = task["slug"]
        episode = task["episode"]
        index = int(episode["episode_index"])
        record_path = HERE / "runs/production/raw" / slug / f"episode_{index:04d}.json"
        trajectory_path = HERE / "runs/production/trajectories" / slug / f"episode_{index:04d}.npz"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        expected = {
            "condition": condition,
            "episode_index": index,
            "extended_config_sha256": config_payload["content_sha256"],
            "original_record_sha256": task["original_record_sha256"],
            "original_trajectory_sha256": task["original_trajectory_sha256"],
        }
        if existing_valid(record_path, expected):
            skipped += 1
            continue
        row, arrays = wide_runner.rollout(
            controller=slug,
            episode=episode,
            policy=policy,
            sample_action=sample_action,
            model=model,
            training_reference=reference,
            config=config,
            cbf=cbf,
            feature_builder_cls=Frozen850FeatureBuilder,
            project=project_velocity_with_retry,
            first_event=first_event,
        )
        prefix = prefix_audit(task["original_trajectory_path"], arrays)
        if row["episode_steps"] <= ORIGINAL_HORIZON:
            raise RuntimeError(("extended timeout replay terminated in original prefix", condition, index, row))
        outcome_map = {
            "success": "LATE_SUCCESS",
            "deadlock": "LATE_DEADLOCK",
            "timeout": "PERSISTENT_TIMEOUT",
        }
        extended_class = outcome_map.get(row["outcome"], "OTHER_FAILURE")
        executed_norm = np.asarray(arrays["executed_correction_norm"], dtype=np.float64)
        dt = float(config.dt)
        j_prefix = float(dt * np.sum(executed_norm[:ORIGINAL_HORIZON] ** 2))
        j_extra = float(dt * np.sum(executed_norm[ORIGINAL_HORIZON:] ** 2))
        original_j = float(task["original_row"]["J_def"])
        if not np.isclose(j_prefix, original_j, atol=1e-14, rtol=0.0):
            raise RuntimeError(("prefix J_def mismatch", condition, index, j_prefix, original_j))
        progress = progress_diagnostics(arrays, dt)
        first_success_step = int(row["episode_steps"]) if extended_class == "LATE_SUCCESS" else None
        first_deadlock_step = int(row["episode_steps"]) if extended_class == "LATE_DEADLOCK" else None
        atomic_npz(trajectory_path, **arrays)
        row.update({
            "schema": "gphi_extended_horizon_episode_v1",
            "record_complete": True,
            "condition": condition,
            "original_outcome": "timeout",
            "original_horizon_steps": ORIGINAL_HORIZON,
            "extended_horizon_steps": EXTENDED_HORIZON,
            "extended_outcome_class": extended_class,
            "first_success_step": first_success_step,
            "first_deadlock_step": first_deadlock_step,
            "total_completion_time_sec": None if first_success_step is None else first_success_step * dt,
            "extra_time_after_original_horizon_sec": None if first_success_step is None else (first_success_step - ORIGINAL_HORIZON) * dt,
            "J_def_0_850": j_prefix,
            "J_def_additional_after_850": j_extra,
            "J_def_extended_total": j_prefix + j_extra,
            "progress_after_850": progress,
            "prefix_audit": prefix,
            "original_record_path": str(task["original_record_path"]),
            "original_record_sha256": task["original_record_sha256"],
            "original_trajectory_path": str(task["original_trajectory_path"]),
            "original_trajectory_sha256": task["original_trajectory_sha256"],
            "extended_config_sha256": config_payload["content_sha256"],
            "trajectory_file": str(trajectory_path.relative_to(HERE)),
            "trajectory_sha256": sha256(trajectory_path),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(record_path, row)
        completed += 1
        print(json.dumps({
            "task": task["task_index"], "condition": condition,
            "episode": index, "extended": extended_class,
            "steps": row["episode_steps"], "prefix": prefix["status"],
        }), flush=True)

    write_json(HERE / "runs/production" / f"runtime_{os.getpid()}.json", {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "task_slice": [args.start_task, args.stop_task],
        "completed_tuples": completed,
        "skipped_valid_tuples": skipped,
        "device": args.device,
        "jax_devices": [str(device) for device in jax.devices()],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "platform": platform.platform(),
        "python": sys.version,
        "extended_config_sha256": config_payload["content_sha256"],
    })
    print(json.dumps({"status": "PASS", "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()

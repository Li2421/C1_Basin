"""Run and cache the common Safety prefixes for the development cohort."""

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
HERE = ROOT / "diagnostics/recovery_takeover_primitive_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
MANIFEST = HERE / "development_manifest.json"

sys.path[:0] = [str(ROOT), str(SYSROOT), str(PILOT)]
from pilot_common import canonical_json_hash, sha256  # noqa: E402
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry  # noqa: E402
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


def outcome(env: GiveWayEnv, error: dict | None) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", args.device)
    manifest = json.loads(MANIFEST.read_text())
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_json_hash(body) != manifest["content_sha256"] or not manifest["frozen_before_any_outcome"]:
        raise RuntimeError("development manifest integrity failure")
    flow_path = Path(manifest["controllers"]["flowbc"]["path"])
    if sha256(flow_path) != manifest["controllers"]["flowbc"]["sha256"]:
        raise RuntimeError("FlowBC checkpoint mismatch")
    config = Config(**manifest["environment"])
    cbf = CBFConfig(**manifest["cbf"])
    policy, provenance = load_policy(flow_path)
    if not provenance or provenance["evaluation_environment"] != manifest["environment"]:
        raise RuntimeError("FlowBC environment mismatch")
    sample = jax.jit(lambda observation, key: policy.sample_actions(observation[None], seed=key)[0])
    flow_root = int(manifest["flow_randomness"]["root_seed"])
    manifest_sha = sha256(MANIFEST)
    assigned = [row for row in manifest["episodes"] if int(row["episode_index"]) % args.shard_count == args.shard_index]
    started = datetime.now(timezone.utc).isoformat()
    counters = {"episodes": 0, "steps": 0}
    for episode in assigned:
        index = int(episode["episode_index"])
        raw_path = HERE / "runs/safety_raw" / f"episode_{index:04d}.json"
        trajectory_path = HERE / "runs/safety_trajectories" / f"episode_{index:04d}.npz"
        if raw_path.is_file() and trajectory_path.is_file():
            continue
        env = GiveWayEnv(config)
        env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
        episode_key = jax.random.fold_in(jax.random.PRNGKey(np.uint32(flow_root)), int(episode["rollout_id"]))
        names = [
            "step", "flow_step_key", "positions_before", "velocities_before", "positions_after",
            "u_flow", "u_safe", "goal_errors_after", "stuck_timer_after", "max_stuck_timer_after",
            "candidate_since_after", "ever_candidate_deadlock_after", "first_success_step_after",
            "first_deadlock_step_after", "first_wall_collision_step_after", "first_agent_collision_step_after",
            "event", "wall_collision", "agent_collision", "first_retry", "linear_min", "speed_excess",
        ]
        buffers: dict[str, list] = {name: [] for name in names}
        error = None
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
                finite = np.isfinite(u_safe).all()
                if not finite:
                    nan_inf_events += 1
                if not finite or linear_min < -cbf.feasibility_tol or speed_excess > cbf.speed_tol:
                    invalid_actions += 1
                    raise RuntimeError(("invalid Safety action", finite, linear_min, speed_excess))
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
                    "agent_collision": info["agent_collision"], "first_retry": retry,
                    "linear_min": linear_min, "speed_excess": speed_excess,
                }
                for name in names:
                    buffers[name].append(values[name])
            except Exception as exc:
                projection_failures += int("projection" in str(exc).lower() or "solver" in type(exc).__name__.lower())
                error = {"type": type(exc).__name__, "message": str(exc), "step": int(env.step_count)}
                break
        arrays = {key: np.asarray(value) for key, value in buffers.items()}
        arrays["initial_goal_errors"] = np.linalg.norm(env.goals - np.asarray(episode["initial_positions"], dtype=np.float64), axis=-1)
        atomic_npz(trajectory_path, **arrays)
        result = outcome(env, error)
        row = {
            "schema": "recovery_takeover_safety_episode_v1", "record_complete": True,
            "episode_index": index, "source_id": episode["source_id"], "rollout_id": episode["rollout_id"],
            "initial_positions": episode["initial_positions"], "outcome": result,
            "success": result == "success", "timeout": result == "timeout", "deadlock": result == "deadlock",
            "collision": result == "collision", "other_failure": result == "other",
            "terminal_step": int(env.step_count), "terminal_time_sec": env.step_count * config.dt,
            "execution_error": error, "wall_collision_events": wall_events, "agent_collision_events": agent_events,
            "projection_failures": projection_failures, "invalid_actions": invalid_actions,
            "nan_inf_events": nan_inf_events, "first_projection_retry_count": int(np.sum(buffers["first_retry"])),
            "manifest_sha256": manifest_sha, "manifest_content_sha256": manifest["content_sha256"],
            "flow_root_seed": flow_root, "flow_semantics": manifest["flow_randomness"]["semantics"],
            "trajectory_file": str(trajectory_path.relative_to(HERE)), "trajectory_sha256": sha256(trajectory_path),
            "rollout_started_after_manifest_freeze": started > manifest["frozen_utc"],
            "runtime_seconds": time.monotonic() - began,
        }
        atomic_json(raw_path, row)
        counters["episodes"] += 1
        counters["steps"] += env.step_count
        print(json.dumps({"episode": index, "outcome": result, "steps": env.step_count}), flush=True)
    atomic_json(HERE / "runs" / f"safety_runtime_shard{args.shard_index}.json", {
        "shard_index": args.shard_index, "shard_count": args.shard_count, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), **counters,
        "device": args.device, "jax_devices": [str(device) for device in jax.devices()],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "manifest_sha256": manifest_sha,
    })


if __name__ == "__main__":
    main()

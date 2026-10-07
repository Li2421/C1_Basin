"""Reconstruct exact Safety takeover states and freeze nominal matching/tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_takeover_primitive_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
MANIFEST = HERE / "development_manifest.json"
OFFSETS = (160, 80, 40, 20)
CONTROLLERS = ("continue_safety", "dense_direct_g", "persistent_structured_eta")

sys.path[:0] = [str(ROOT), str(SYSROOT)]
from diagnostics.gphi_training_dataset_v2.build_states import restore_full, save_full  # noqa: E402
from single_integrator.environment import Config, GiveWayEnv  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def replay_state(episode: dict, trajectory: dict[str, np.ndarray], step: int, config: Config) -> GiveWayEnv:
    env = GiveWayEnv(config)
    env.reset(np.asarray(episode["initial_positions"], dtype=np.float64))
    for action in np.asarray(trajectory["u_safe"], dtype=np.float64)[:step]:
        _, _, done, _ = env.step(action)
        if done:
            raise RuntimeError((episode["episode_index"], "terminated before takeover", step, env.step_count))
    if env.step_count != step:
        raise RuntimeError("replay step mismatch")
    if step < len(trajectory["step"]):
        if not np.array_equal(env.positions, np.asarray(trajectory["positions_before"][step], dtype=np.float64)):
            raise RuntimeError((episode["episode_index"], step, "position replay mismatch"))
        if not np.array_equal(env.velocities, np.asarray(trajectory["velocities_before"][step], dtype=np.float64)):
            raise RuntimeError((episode["episode_index"], step, "velocity replay mismatch"))
    return env


def save_and_audit(path: Path, env: GiveWayEnv, config: Config) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_full(path, env)
    restored = restore_full(path, config)
    maxima = {
        "positions": float(np.max(np.abs(restored.positions - env.positions))),
        "velocities": float(np.max(np.abs(restored.velocities - env.velocities))),
        "last_goal_error": float(np.max(np.abs(np.asarray(restored.distance_history[-1]) - np.asarray(env.distance_history[-1])))),
        "stuck_timer": abs(restored.stuck_timer - env.stuck_timer),
    }
    scalar_equal = (
        restored.step_count == env.step_count and restored.candidate_since == env.candidate_since
        and restored.ever_candidate_deadlock == env.ever_candidate_deadlock and restored.done == env.done
    )
    if max(maxima.values()) > 1e-12 or not scalar_equal:
        raise RuntimeError((path, maxima, scalar_equal))
    return maxima


def choose_control(failure_episode: int, offset: int, step: int, candidates: list[int]) -> int:
    token = f"recovery_nominal_match_v1|{failure_episode}|{offset}|{step}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest(), "big")
    return candidates[value % len(candidates)]


def main() -> None:
    if (HERE / "branch_task_manifest.json").exists():
        raise RuntimeError("takeover task manifest already frozen")
    manifest = json.loads(MANIFEST.read_text())
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_hash(body) != manifest["content_sha256"]:
        raise RuntimeError("development manifest changed")
    safety_files = sorted((HERE / "runs/safety_raw").glob("episode_*.json"))
    if len(safety_files) != 200:
        raise RuntimeError(("Safety incomplete", len(safety_files)))
    config = Config(**manifest["environment"])
    episodes = {int(row["episode_index"]): row for row in manifest["episodes"]}
    safety = {index: json.loads((HERE / "runs/safety_raw" / f"episode_{index:04d}.json").read_text()) for index in range(200)}
    if any(row["manifest_sha256"] != sha256(MANIFEST) for row in safety.values()):
        raise RuntimeError("Safety record manifest mismatch")
    successes = sorted(index for index, row in safety.items() if row["outcome"] == "success")
    failures = sorted(index for index, row in safety.items() if row["outcome"] != "success")
    if not successes or not failures:
        raise RuntimeError("development cohort lacks success/failure mixture")
    trajectories: dict[int, dict[str, np.ndarray]] = {}

    def trajectory(index: int) -> dict[str, np.ndarray]:
        if index not in trajectories:
            path = HERE / safety[index]["trajectory_file"]
            if sha256(path) != safety[index]["trajectory_sha256"]:
                raise RuntimeError((index, "Safety trajectory hash mismatch"))
            with np.load(path, allow_pickle=False) as data:
                trajectories[index] = {key: np.asarray(data[key]) for key in data.files}
        return trajectories[index]

    state_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    max_restore_diff = 0.0
    nominal_cache: dict[tuple[int, int], tuple[Path, str]] = {}
    unavailable = 0
    for failure_index in failures:
        terminal = int(safety[failure_index]["terminal_step"])
        for offset in OFFSETS:
            takeover = terminal - offset
            state_id = f"failure_e{failure_index:04d}_minus{offset:03d}"
            if takeover < 0:
                state_rows.append({"state_id": state_id, "failure_episode_index": failure_index,
                                   "failure_type": safety[failure_index]["outcome"], "offset_steps": offset,
                                   "offset_seconds": offset * config.dt, "terminal_step": terminal,
                                   "takeover_step": takeover, "available": False, "reason": "episode_too_short"})
                unavailable += 1
                continue
            failure_env = replay_state(episodes[failure_index], trajectory(failure_index), takeover, config)
            failure_path = HERE / "states/failure" / f"{state_id}.npz"
            diffs = save_and_audit(failure_path, failure_env, config)
            max_restore_diff = max(max_restore_diff, *diffs.values())
            state_row = {
                "state_id": state_id, "failure_episode_index": failure_index,
                "failure_source_id": episodes[failure_index]["source_id"],
                "failure_type": safety[failure_index]["outcome"], "offset_steps": offset,
                "offset_seconds": offset * config.dt, "terminal_step": terminal, "takeover_step": takeover,
                "available": True, "state_file": str(failure_path), "state_sha256": sha256(failure_path),
                "flow_rollout_id": episodes[failure_index]["rollout_id"],
            }
            state_rows.append(state_row)
            for controller in CONTROLLERS:
                tasks.append({"task_id": f"failure|{state_id}|{controller}", "cohort": "failure",
                              "controller": controller, **state_row})
            candidates = [index for index in successes if int(safety[index]["terminal_step"]) > takeover]
            if not candidates:
                matching_rows.append({"state_id": state_id, "failure_episode_index": failure_index,
                                      "offset_steps": offset, "takeover_step": takeover,
                                      "available": False, "reason": "no_Safety_success_episode_active_at_takeover"})
                continue
            control_index = choose_control(failure_index, offset, takeover, candidates)
            key = (control_index, takeover)
            if key not in nominal_cache:
                control_env = replay_state(episodes[control_index], trajectory(control_index), takeover, config)
                control_path = HERE / "states/nominal" / f"success_e{control_index:04d}_step{takeover:04d}.npz"
                diffs = save_and_audit(control_path, control_env, config)
                max_restore_diff = max(max_restore_diff, *diffs.values())
                nominal_cache[key] = (control_path, sha256(control_path))
            control_path, control_sha = nominal_cache[key]
            control_state_id = f"nominal_for_{state_id}"
            match = {
                "state_id": state_id, "control_state_id": control_state_id,
                "failure_episode_index": failure_index, "failure_type": safety[failure_index]["outcome"],
                "offset_steps": offset, "offset_seconds": offset * config.dt, "takeover_step": takeover,
                "available": True, "eligible_success_candidates": len(candidates),
                "control_episode_index": control_index, "control_source_id": episodes[control_index]["source_id"],
                "control_terminal_step": safety[control_index]["terminal_step"],
                "state_file": str(control_path), "state_sha256": control_sha,
                "flow_rollout_id": episodes[control_index]["rollout_id"],
            }
            matching_rows.append(match)
            for controller in CONTROLLERS:
                tasks.append({"task_id": f"nominal|{state_id}|{controller}", "cohort": "nominal",
                              "controller": controller, **match})
    write_csv(HERE / "takeover_state_manifest.csv", state_rows)
    write_csv(HERE / "nominal_control_matching.csv", matching_rows)
    task_manifest = {
        "schema": "recovery_takeover_branch_tasks_v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "development_manifest": str(MANIFEST), "development_manifest_sha256": sha256(MANIFEST),
        "development_manifest_content_sha256": manifest["content_sha256"],
        "Safety_counts": dict(Counter(row["outcome"] for row in safety.values())),
        "failure_episode_count": len(failures), "failure_takeover_states": sum(row["available"] for row in state_rows),
        "unavailable_failure_states": unavailable,
        "nominal_matches": sum(row["available"] for row in matching_rows),
        "unique_nominal_source_episodes": len({row.get("control_episode_index") for row in matching_rows if row["available"]}),
        "unique_nominal_augmented_states": len(nominal_cache),
        "max_restore_difference": max_restore_diff, "controller_count": len(CONTROLLERS),
        "task_count": len(tasks), "tasks": tasks,
    }
    task_manifest["content_sha256"] = canonical_hash(task_manifest)
    atomic_json(HERE / "branch_task_manifest.json", task_manifest)
    print(json.dumps({key: value for key, value in task_manifest.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()

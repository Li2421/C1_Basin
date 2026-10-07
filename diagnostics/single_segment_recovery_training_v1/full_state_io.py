"""Lossless augmented-state I/O for single-segment continuation branches.

Unlike the older oracle dataset helper, this format stores the complete real
goal-error history from episode step zero.  Startup padding is never serialized
into monitor state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def save_full_history(path: Path, env: Any) -> None:
    if bool(env.done):
        raise ValueError("decision states must be nonterminal")
    history = np.asarray(env.distance_history, dtype=np.float64)
    if history.shape != (int(env.step_count) + 1, 2) or not np.isfinite(history).all():
        raise ValueError(("incomplete/nonfinite actual monitor history", history.shape, env.step_count))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        positions=np.asarray(env.positions, dtype=np.float64),
        velocities=np.asarray(env.velocities, dtype=np.float64),
        step=np.asarray(env.step_count, dtype=np.int64),
        error_history=history,
        history_start_step=np.asarray(0, dtype=np.int64),
        candidate_since=np.asarray(-1 if env.candidate_since is None else env.candidate_since),
        stuck_timer=np.asarray(env.stuck_timer, dtype=np.float64),
        max_stuck_timer=np.asarray(env.max_stuck_timer, dtype=np.float64),
        ever_candidate_deadlock=np.asarray(env.ever_candidate_deadlock),
        first_success_step=np.asarray(-1 if env.first_success_step is None else env.first_success_step),
        first_deadlock_step=np.asarray(-1 if env.first_deadlock_step is None else env.first_deadlock_step),
        first_wall_collision_step=np.asarray(-1 if env.first_wall_collision_step is None else env.first_wall_collision_step),
        first_agent_collision_step=np.asarray(-1 if env.first_agent_collision_step is None else env.first_agent_collision_step),
        done=np.asarray(env.done),
    )
    os.replace(temporary, path)


def restore_full_history(path: Path, config: Any) -> Any:
    from single_integrator.environment import GiveWayEnv

    with np.load(path, allow_pickle=False) as data:
        required = {
            "positions", "velocities", "step", "error_history", "history_start_step",
            "candidate_since", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock",
            "first_success_step", "first_deadlock_step", "first_wall_collision_step",
            "first_agent_collision_step", "done",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError((path, "augmented monitor fields missing", sorted(missing)))
        if "schema_version" in data.files:
            if int(data["schema_version"]) != SCHEMA_VERSION:
                raise RuntimeError((path, "state schema mismatch"))
        elif "complete_real_history" in data.files and bool(data["complete_real_history"]):
            # Compatibility with the outcome-blind Safety collector that was
            # frozen immediately before this shared helper landed.  Acceptance
            # still requires every strict full-history check below; old tail-41
            # snapshots have no marker and remain rejected.
            pass
        else:
            raise RuntimeError((path, "unmarked/legacy truncated state is forbidden"))
        step = int(data["step"])
        history = np.asarray(data["error_history"], dtype=np.float64)
        if int(data["history_start_step"]) != 0:
            raise RuntimeError((path, "truncated history is forbidden"))
        if history.shape != (step + 1, 2) or not np.isfinite(history).all():
            raise RuntimeError((path, "invalid complete history", history.shape, step))
        env = GiveWayEnv(config)
        env.positions = np.asarray(data["positions"], dtype=np.float64).copy()
        env.velocities = np.asarray(data["velocities"], dtype=np.float64).copy()
        env.step_count = step
        env.distance_history = [row.copy() for row in history]
        candidate = int(data["candidate_since"])
        env.candidate_since = None if candidate < 0 else candidate
        env.stuck_timer = float(data["stuck_timer"])
        env.max_stuck_timer = float(data["max_stuck_timer"])
        env.ever_candidate_deadlock = bool(data["ever_candidate_deadlock"])
        for field in (
            "first_success_step", "first_deadlock_step", "first_wall_collision_step",
            "first_agent_collision_step",
        ):
            value = int(data[field])
            setattr(env, field, None if value < 0 else value)
        env.done = bool(data["done"])
    if env.done or env.step_count >= int(config.max_steps):
        raise RuntimeError((path, "absorbing/expired state cannot be branched"))
    return env


def audit_round_trip(path: Path, source: Any, config: Any) -> dict[str, Any]:
    restored = restore_full_history(path, config)
    numeric = {
        "positions": float(np.max(np.abs(restored.positions - source.positions))),
        "velocities": float(np.max(np.abs(restored.velocities - source.velocities))),
        "history": float(np.max(np.abs(
            np.asarray(restored.distance_history) - np.asarray(source.distance_history)
        ))),
        "stuck_timer": abs(restored.stuck_timer - source.stuck_timer),
        "max_stuck_timer": abs(restored.max_stuck_timer - source.max_stuck_timer),
    }
    scalar_fields = (
        "step_count", "candidate_since", "ever_candidate_deadlock", "done",
        "first_success_step", "first_deadlock_step", "first_wall_collision_step",
        "first_agent_collision_step",
    )
    scalar_exact = all(getattr(restored, name) == getattr(source, name) for name in scalar_fields)
    return {
        "max_numeric_difference": max(numeric.values()),
        "numeric_differences": numeric,
        "scalar_fields_exact": scalar_exact,
        "full_history_points": len(restored.distance_history),
        "history_start_step": 0,
        "passed": max(numeric.values()) == 0.0 and scalar_exact,
    }


__all__ = ["SCHEMA_VERSION", "audit_round_trip", "restore_full_history", "save_full_history"]

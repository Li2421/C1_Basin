"""Reusable *development-only* closed-loop evaluation primitives.

This module never constructs a ``JointTransitionDataset`` and never opens a
test rollout archive.  It reads the manifest solely to select ``dev`` +
``nominal`` entries, then opens exactly those archive paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import numpy as np

from .dataset import DATASET_SCHEMA, TRAJECTORY_SCHEMA
from .macflow import sample_bounded_actions


@dataclass(frozen=True)
class DevNominalCase:
    rollout_id: str
    initial_state: dict[str, Any]
    archive_path: Path
    metadata: Mapping[str, Any]


def load_dev_nominal_cases(dataset_root: str | Path, *, expected_count: int | None = None):
    """Read initial states from dev nominal archives only, never test archives."""
    root = Path(dataset_root).resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != DATASET_SCHEMA or not manifest.get("complete", False):
        raise ValueError("dataset manifest is incomplete or has the wrong schema")
    selected = [
        row for row in manifest["files"]
        if row.get("split") == "dev" and row.get("source") == "nominal"
    ]
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(f"expected {expected_count} dev nominal rollouts, found {len(selected)}")
    cases = []
    opened = []
    for row in sorted(selected, key=lambda value: str(value["rollout_id"])):
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("rollouts", "dev"):
            raise ValueError(f"refusing non-dev archive {relative}")
        archive_path = root / relative
        with np.load(archive_path, allow_pickle=False) as archive:
            if str(archive["schema"].item()) != TRAJECTORY_SCHEMA:
                raise ValueError(f"wrong trajectory schema in {archive_path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            initial_state = json.loads(str(archive["initial_state_json"].item()))
        if metadata.get("split") != "dev" or metadata.get("source") != "nominal":
            raise ValueError(f"archive metadata contradicts dev-nominal selection: {archive_path}")
        if metadata.get("rollout_id") != row["rollout_id"]:
            raise ValueError(f"manifest/archive rollout id mismatch: {archive_path}")
        opened.append(str(relative))
        cases.append(DevNominalCase(str(row["rollout_id"]), initial_state, archive_path, metadata))
    audit = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_split": "dev", "selected_source": "nominal", "opened_archives": opened,
        "opened_test_archives": 0,
    }
    return manifest, tuple(cases), audit


def evaluate_dev_nominal(
    agent,
    cases: tuple[DevNominalCase, ...],
    *,
    make_env: Callable[[], Any],
    reset: Callable[[Any, Mapping[str, Any]], None],
    observation: Callable[[Any], np.ndarray],
    step: Callable[[Any, np.ndarray], tuple[bool, Mapping[str, Any]]],
    summary: Callable[[Any], Mapping[str, Any]],
    max_steps: int,
    max_speed: float,
    seed: int = 0,
):
    """Evaluate the supplied policy with ``sample_bounded_actions`` every step."""
    rows = []
    base = jax.random.PRNGKey(seed)
    for index, case in enumerate(cases):
        env = make_env()
        reset(env, case.initial_state)
        episode_key = jax.random.fold_in(base, index)
        executed_max_speed = 0.0
        termination = "timeout"
        final_info: Mapping[str, Any] = {}
        for timestep in range(max_steps):
            policy_key = jax.random.fold_in(episode_key, timestep)
            # MACFlow's JAX sampler supplies the protocol-required per-agent
            # radial bound.  Reapply that same bound in float64 at the plant
            # boundary to remove float32 round-up (e.g. 0.52000004 > 0.52).
            action = np.asarray(sample_bounded_actions(agent, observation(env)[None], policy_key)[0], dtype=np.float64)
            norms = np.linalg.norm(action, axis=-1, keepdims=True)
            action = action * np.minimum(1.0, float(max_speed) / np.maximum(norms, 1e-30))
            executed_max_speed = max(executed_max_speed, float(np.linalg.norm(action, axis=-1).max()))
            done, final_info = step(env, action)
            termination = str(final_info.get("termination", "running"))
            if done:
                break
        outcome = dict(summary(env))
        rows.append({
            "rollout_id": case.rollout_id,
            "split": "dev",
            "source": "nominal",
            "termination": termination,
            "success": bool(outcome.get("collision_free_success", outcome.get("success", False))),
            "obstacle_collision": bool(outcome.get("obstacle_collision", False)),
            "wall_collision": bool(outcome.get("wall_collision", False)),
            "agent_collision": bool(outcome.get("agent_collision", False)),
            "timeout": termination == "timeout",
            "episode_length": int(outcome.get("episode_steps", 0)),
            "max_executed_speed": executed_max_speed,
            "first_success_step": outcome.get("first_success_step"),
            "first_obstacle_collision_step": outcome.get("first_obstacle_collision_step"),
            "first_wall_collision_step": outcome.get("first_outer_collision_step"),
            "first_agent_collision_step": outcome.get("first_agent_collision_step"),
        })
    count = len(rows)
    rate = lambda name: (sum(bool(row[name]) for row in rows) / count) if count else 0.0
    aggregate = {
        "rollouts": count, "full_task_success": rate("success"),
        "obstacle_collision": rate("obstacle_collision"), "wall_collision": rate("wall_collision"),
        "agent_collision": rate("agent_collision"), "timeout": rate("timeout"),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])) if rows else 0.0,
        "max_executed_speed": max((row["max_executed_speed"] for row in rows), default=0.0),
        "controller": "MACFlow sample_bounded_actions (per-agent radial speed bound)",
    }
    return {"aggregate": aggregate, "rollouts": rows}

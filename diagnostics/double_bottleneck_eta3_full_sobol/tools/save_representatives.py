#!/usr/bin/env python3
"""Save pre-registered high/median/low P0 success trace pairs."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import numpy as np

from double_bottleneck.environment import Config
from double_bottleneck.evaluate_flowbc_4a import _initialize_env, _radial_bound64
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.visualization import save_trajectory_svg
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
CHECKPOINT = ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}


def trace(policy, dataset, episode, theta, seed, rollout_id):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = CertifiedHardSafetyFilter()
    rep = REPRESENTATIONS["P0-3D"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(rollout_id))
    positions = [env.positions.copy()]
    goals = env.goals.copy()
    values = {name: [] for name in ("raw_g_norm", "executable_g_norm", "second_projection_norm", "total_goal_error", "minimum_wall_clearance_step", "minimum_pair_clearance_step")}
    terminal = None
    for step in range(config.max_steps):
        step_key = jax.random.fold_in(key, step)
        raw = np.asarray(policy.sample_actions(env.observation()[None], step_key)[0], dtype=np.float64)
        u_flow = _radial_bound64(raw, config.max_speed)
        u_safe = np.asarray(projector(env.snapshot(), u_flow).velocity, dtype=np.float64)
        g = rep.correction(np.asarray(theta), env.positions, env.goals, u_safe, config.max_speed, step, config.max_steps)
        projected = projector(env.snapshot(), u_safe + g)
        u_exec = np.asarray(projected.velocity, dtype=np.float64)
        executed = u_exec - u_safe
        values["raw_g_norm"].append(float(np.linalg.norm(g)))
        values["executable_g_norm"].append(float(np.linalg.norm(executed)))
        values["second_projection_norm"].append(float(np.linalg.norm(u_exec - (u_safe + g))))
        _, _, done, info = env.step(u_exec)
        positions.append(env.positions.copy())
        values["total_goal_error"].append(float(np.linalg.norm(goals - env.positions, axis=-1).sum()))
        values["minimum_wall_clearance_step"].append(float(info["min_swept_wall_distance"]))
        values["minimum_pair_clearance_step"].append(float(info["min_swept_agent_distance"]))
        terminal = str(info["termination"])
        if done:
            break
    return env, {
        "positions": np.asarray(positions),
        "goals": goals,
        **{name: np.asarray(value) for name, value in values.items()},
        "theta": np.asarray(theta),
        "dt": np.asarray(config.dt),
        "goal_tolerance": np.asarray(config.goal_tolerance),
        "terminal": terminal,
    }


def summarize(data: dict) -> dict:
    positions = np.asarray(data["positions"], dtype=np.float64)
    goals = np.asarray(data["goals"], dtype=np.float64)
    dt = float(data["dt"])
    tolerance = float(data["goal_tolerance"])
    directions = -np.sign(positions[0, :, 0]).astype(int)
    first_planes = np.where(directions > 0, -0.79, 0.79)
    second_planes = np.where(directions > 0, 2.01, -2.01)

    def crossings(planes):
        crossed = directions[None] * (positions[:, :, 0] - planes[None]) >= 0
        return np.asarray([np.flatnonzero(crossed[:, i])[0] if np.any(crossed[:, i]) else -1 for i in range(4)], dtype=int)

    first, second = crossings(first_planes), crossings(second_planes)
    goal_distance = np.linalg.norm(positions - goals[None], axis=-1)
    entries = np.asarray([np.flatnonzero(goal_distance[:, i] <= tolerance)[0] if np.any(goal_distance[:, i] <= tolerance) else -1 for i in range(4)], dtype=int)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / dt
    valid = (first >= 0) & (second >= 0)
    as_times = lambda indices: [float(index * dt) if index >= 0 else None for index in indices]
    return {
        "steps": len(positions) - 1,
        "duration_seconds": (len(positions) - 1) * dt,
        "first_bottleneck_crossing_seconds_per_agent": as_times(first),
        "second_bottleneck_crossing_seconds_per_agent": as_times(second),
        "last_second_bottleneck_clearance_seconds": float(second[second >= 0].max() * dt) if np.any(second >= 0) else None,
        "mean_chamber_transit_seconds": float(np.mean((second[valid] - first[valid]) * dt)) if np.any(valid) else None,
        "waiting_seconds_per_agent": (np.sum(speeds < 0.025, axis=0) * dt).tolist(),
        "mean_waiting_seconds_per_agent": float(np.mean(np.sum(speeds < 0.025, axis=0) * dt)),
        "goal_entry_seconds_per_agent": as_times(entries),
        "initial_goal_error_per_agent": goal_distance[0].tolist(),
        "final_goal_error_per_agent": goal_distance[-1].tolist(),
        "progress_per_agent": (goal_distance[0] - goal_distance[-1]).tolist(),
        "mean_raw_g_norm": float(np.mean(data["raw_g_norm"])),
        "mean_executable_g_norm": float(np.mean(data["executable_g_norm"])),
        "mean_second_projection_norm": float(np.mean(data["second_projection_norm"])),
    }


def main() -> int:
    geometry = json.loads((STUDY / "geometry_representatives.json").read_text())
    selections = {row["episode_id"]: row for row in json.loads((STUDY / "selected_success_representatives.json").read_text())["episodes"]}
    catalog = {row["episode_id"]: row for row in json.loads((STUDY / "episode_catalog.json").read_text())["episodes"]}
    datasets = {name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46) for name, path in DATASETS.items()}
    policy, _ = load_checkpoint(CHECKPOINT, next(iter({dataset.environment_fingerprint for dataset in datasets.values()})))
    output = STUDY / "representatives"
    output.mkdir(exist_ok=True)
    metadata = []
    for geometry_role in ("high", "median", "low"):
        episode_id = geometry[geometry_role]
        item = selections[episode_id]
        representative = min(item["representatives"], key=lambda row: (-row["timeout_coverage"], -row["controls_preserved"], float(np.linalg.norm(row["theta"])), row["parameter_id"]))
        episode_meta = catalog[episode_id]
        dataset = datasets[episode_meta["set"]]
        episode = dataset.by_family[episode_meta["family_id"]][0]
        stem = episode_id.replace("|", "_")
        traces = {}
        for label, theta in (("eta0", (0.0, 0.0, 0.0)), ("eta_success", representative["theta"])):
            env, data = trace(policy, dataset, episode, theta, episode_meta["seed"], episode_meta["rollout_id"])
            terminal = data.pop("terminal")
            npz_path = output / f"{stem}_{label}.npz"
            np.savez_compressed(npz_path, **data)
            svg_path = output / f"{stem}_{label}.svg"
            save_trajectory_svg(env, data["positions"], theta, terminal, svg_path, title=f"{episode_id} — {label}")
            traces[label] = {
                "theta": list(map(float, theta)),
                "terminal": terminal,
                "npz": str(npz_path.relative_to(ROOT)),
                "svg": str(svg_path.relative_to(ROOT)),
                "summary": summarize(data),
            }
        metadata.append({
            "geometry_role": geometry_role,
            "episode_id": episode_id,
            "regime": episode_meta["regime"],
            "successful_parameter_id": representative["parameter_id"],
            "timeout_coverage": representative["timeout_coverage"],
            "controls_preserved": representative["controls_preserved"],
            "traces": traces,
        })
    result = {"schema": "double_bottleneck_eta3_representative_mechanism_v1", "episodes": metadata}
    (output / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (STUDY / "behavioral_mechanism.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representative_episodes": len(metadata)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

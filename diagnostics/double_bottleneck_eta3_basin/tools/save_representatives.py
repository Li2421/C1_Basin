#!/usr/bin/env python3
"""Re-run a frozen, deterministic subset and save full eta=0/success traces."""

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
from shared_control.diagnostic_corrector import DiagnosticCorrector
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"
CHECKPOINT = ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}


def trace(policy, dataset, episode, eta, seed, rollout_id):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = CertifiedHardSafetyFilter()
    corrector = DiagnosticCorrector(np.asarray(eta, dtype=np.float64))
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(rollout_id))
    positions = [env.positions.copy()]
    goals = env.goals.copy()
    g_values, second_values, progress_values = [], [], []
    wall_values, pair_values = [], []
    terminal = None
    for step in range(config.max_steps):
        step_key = jax.random.fold_in(key, step)
        raw = np.asarray(policy.sample_actions(env.observation()[None], step_key)[0], dtype=np.float64)
        u_flow = _radial_bound64(raw, config.max_speed)
        u_safe = np.asarray(projector(env.snapshot(), u_flow).velocity, dtype=np.float64)
        g, _, _ = corrector(env.positions, env.goals, u_safe, config.max_speed)
        result = projector(env.snapshot(), u_safe + g)
        u_exec = np.asarray(result.velocity, dtype=np.float64)
        second_values.append(float(np.linalg.norm(u_exec - (u_safe + g))))
        g_values.append(float(np.linalg.norm(g)))
        _, _, done, info = env.step(u_exec)
        positions.append(env.positions.copy())
        progress_values.append(float(np.linalg.norm(goals - env.positions, axis=-1).sum()))
        wall_values.append(float(info["min_swept_wall_distance"]))
        pair_values.append(float(info["min_swept_agent_distance"]))
        terminal = str(info["termination"])
        if done:
            break
    return env, {
        "positions": np.asarray(positions),
        "g_norm": np.asarray(g_values),
        "second_projection_norm": np.asarray(second_values),
        "total_goal_error": np.asarray(progress_values),
        "minimum_wall_clearance_step": np.asarray(wall_values),
        "minimum_pair_clearance_step": np.asarray(pair_values),
        "eta": np.asarray(eta),
        "terminal": terminal,
        "dt": config.dt,
    }


def main() -> int:
    results = json.loads((STUDY / "per_episode_basin.json").read_text())["episodes"]
    candidates = [row for row in results if row["representative"] is not None]
    selected = []
    for basin_type in sorted({row["basin_type"] for row in candidates}):
        group = [row for row in candidates if row["basin_type"] == basin_type]
        selected.append(max(group, key=lambda row: (row["stage_a_basin_fraction"], row["episode_id"])))
    for regime in ("clearly_asymmetric", "weakly_asymmetric", "near_symmetric"):
        group = [row for row in candidates if row["regime"] == regime and row not in selected]
        if group:
            selected.append(max(group, key=lambda row: (row["stage_a_basin_fraction"], row["episode_id"])))
    selected = selected[:6]
    datasets = {
        name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46)
        for name, path in DATASETS.items()
    }
    fingerprint = next(iter({dataset.environment_fingerprint for dataset in datasets.values()}))
    policy, _ = load_checkpoint(CHECKPOINT, fingerprint)
    output = STUDY / "representatives"
    output.mkdir(exist_ok=True)
    metadata = []
    for item in selected:
        dataset = datasets[item["set"]]
        episode = dataset.by_family[item["family_id"]][0]
        stem = item["episode_id"].replace("|", "_")
        pair = (("eta0", (0.0, 0.0, 0.0)), ("eta_success", item["representative"]["eta"]))
        meta = {"episode_id": item["episode_id"], "basin_type": item["basin_type"], "regime": item["regime"], "traces": {}}
        for label, eta in pair:
            env, values = trace(policy, dataset, episode, eta, item["seed"], item["rollout_id"])
            terminal = values.pop("terminal")
            dt = values.pop("dt")
            npz_path = output / f"{stem}_{label}.npz"
            np.savez_compressed(npz_path, **values, dt=np.asarray(dt))
            svg_path = output / f"{stem}_{label}.svg"
            save_trajectory_svg(env, values["positions"], eta, terminal, svg_path, title=f"{item['episode_id']} — {label}")
            meta["traces"][label] = {
                "eta": list(map(float, eta)),
                "terminal": terminal,
                "steps": int(len(values["g_norm"])),
                "npz": str(npz_path.relative_to(ROOT)),
                "svg": str(svg_path.relative_to(ROOT)),
                "mean_g_norm": float(values["g_norm"].mean()),
                "mean_second_projection_norm": float(values["second_projection_norm"].mean()),
                "final_total_goal_error": float(values["total_goal_error"][-1]),
            }
        metadata.append(meta)
    (output / "metadata.json").write_text(json.dumps({"episodes": metadata}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"representative_episodes": len(metadata)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

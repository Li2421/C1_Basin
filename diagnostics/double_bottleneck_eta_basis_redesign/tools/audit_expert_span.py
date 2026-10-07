#!/usr/bin/env python3
"""Uniform-state expert executable-action span diagnostic for P0 and P1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.expert import CentralizedExpert
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import P0_HIGH, P0_LOW, correction


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}


def evaluate(projector, snapshot, name, theta, positions, goals, u_safe, max_speed, scale):
    raw = correction(name, theta, positions, goals, u_safe, max_speed, scale)
    return np.asarray(projector(snapshot, u_safe + raw).velocity, dtype=float)


def fit(projector, snapshot, name, candidates, positions, goals, u_safe, target, max_speed, scale):
    def objective(theta):
        return float(np.linalg.norm(evaluate(projector, snapshot, name, theta, positions, goals, u_safe, max_speed, scale) - target))
    values = np.asarray([objective(theta) for theta in candidates], dtype=float)
    index = int(np.argmin(values))
    start = candidates[index]
    optimized = minimize(
        objective,
        start,
        method="Powell",
        bounds=list(zip(P0_LOW, P0_HIGH, strict=True)),
        options={"maxiter": 100, "maxfev": 400, "xtol": 1e-4, "ftol": 1e-7, "disp": False},
    )
    if optimized.fun <= values[index]:
        theta, residual = np.asarray(optimized.x, dtype=float), float(optimized.fun)
    else:
        theta, residual = start, float(values[index])
    executed = evaluate(projector, snapshot, name, theta, positions, goals, u_safe, max_speed, scale)
    raw = correction(name, theta, positions, goals, u_safe, max_speed, scale)
    executed_correction = executed - u_safe
    return {
        "best_theta": theta.tolist(),
        "best_executable_action_residual": residual,
        "best_sobol_residual": float(values[index]),
        "optimizer_success": bool(optimized.success),
        "optimizer_evaluations": int(optimized.nfev),
        "raw_correction_norm": float(np.linalg.norm(raw)),
        "executed_correction_norm": float(np.linalg.norm(executed_correction)),
        "projection_retention_ratio": float(np.linalg.norm(executed_correction) / max(np.linalg.norm(raw), 1e-12)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anchors = [row for row in json.loads((OUT / "state_anchor_manifest.json").read_text())["anchors"] if row["trajectory_class"] != "successful_eta_corrected"]
    anchors = [row for index, row in enumerate(anchors) if index % args.shards == args.shard]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {json.loads(line)["state_id"] for line in args.output.read_text().splitlines() if line.strip()}
    unit = qmc.Sobol(3, scramble=True, seed=2026092704).random_base2(10)
    candidates = qmc.scale(unit, P0_LOW, P0_HIGH)
    scale = float(json.loads((OUT / "P1_SCALE.json").read_text())["scale"])
    datasets = {name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46) for name, path in DATASETS.items()}
    lookup = {(name, family): dataset.by_family[family][0] for name, dataset in datasets.items() for family in dataset.family_names}
    expert = CentralizedExpert()
    projector = CertifiedHardSafetyFilter()
    with args.output.open("a") as handle:
        for ordinal, anchor in enumerate(anchors, 1):
            if anchor["state_id"] in completed:
                continue
            started = time.perf_counter()
            try:
                dataset = datasets[anchor["set"]]
                config = Config(**dataset.config)
                episode = lookup[(anchor["set"], anchor["family_id"])]
                with np.load(ROOT / anchor["trajectory_path"], allow_pickle=False) as trace:
                    positions_all = np.asarray(trace["positions"], dtype=float)
                    safe_actions = np.asarray(trace["executed_actions"], dtype=float)
                step = int(anchor["source_step"])
                positions = positions_all[step]
                previous_velocity = episode.initial_velocities if step == 0 else safe_actions[step - 1]
                u_safe = safe_actions[step]
                env = DoubleBottleneckEnv(config)
                env.reset(positions, regime=anchor["regime"])
                state = env.augmented_state()
                state["last_applied_velocity"] = previous_velocity
                env.restore_augmented_state(state)
                snapshot = env.snapshot()
                plan = expert.plan(env)
                if not plan.success or len(plan.actions) == 0:
                    raise RuntimeError("expert continuation did not succeed")
                expert_raw = np.asarray(plan.actions[0], dtype=float)
                expert_safe = np.asarray(projector(snapshot, expert_raw).velocity, dtype=float)
                zero_residual = float(np.linalg.norm(u_safe - expert_safe))
                fits = {
                    name: fit(projector, snapshot, name, candidates, positions, env.goals, u_safe, expert_safe, config.max_speed, scale)
                    for name in ("P0-3D", "P1-OrthoFlow3")
                }
                result = {
                    **anchor,
                    "expert_recovery_valid": True,
                    "expert_terminal": plan.terminal_reason,
                    "expert_steps": plan.episode_steps,
                    "expert_mode": plan.hypothesis.label,
                    "theta_zero_residual": zero_residual,
                    "u_safe": u_safe.tolist(),
                    "u_expert_safe": expert_safe.tolist(),
                    "fits": fits,
                }
            except Exception as error:
                result = {
                    **anchor,
                    "expert_recovery_valid": False,
                    "expert_error_type": type(error).__name__,
                    "expert_error": str(error),
                    "fits": {},
                }
            result["wall_seconds"] = time.perf_counter() - started
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"shard": args.shard, "done": ordinal, "state": anchor["state_id"], "valid": result["expert_recovery_valid"], "seconds": result["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

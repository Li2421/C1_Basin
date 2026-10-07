#!/usr/bin/env python3
"""Uniform-state expert-fit and projection-sensitivity diagnostic."""

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
from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
HARD = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}
FIT_SEED = 830017
FD_STEP = 0.015625
N_CANDIDATES = 1024


def stats_vector(matrix: np.ndarray) -> dict:
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(1e-3, 0.01 * float(singular[0])) if len(singular) else 1e-3
    active = singular[singular >= threshold]
    return {
        "singular_values": singular.tolist(),
        "effective_rank": int(len(active)),
        "rank_threshold": threshold,
        "condition_number_effective": float(active[0] / active[-1]) if len(active) else None,
        "frobenius_norm": float(np.linalg.norm(matrix)),
    }


def candidates(name: str, ordinal: int) -> np.ndarray:
    rep = REPRESENTATIONS[name]
    unit = qmc.Sobol(rep.dimension, scramble=True, seed=FIT_SEED + 1009 * ordinal).random_base2(10)
    return qmc.scale(unit, rep.low, rep.high)


def evaluate(projector, snapshot, rep, theta, positions, goals, u_safe, source_step):
    raw = rep.correction(theta, positions, goals, u_safe, 0.5, source_step, 850)
    executed = np.asarray(projector(snapshot, u_safe + raw).velocity, dtype=np.float64) - u_safe
    return raw, executed


def sensitivity(projector, snapshot, rep, theta, positions, goals, u_safe, source_step):
    theta = np.asarray(theta, dtype=np.float64)
    width = rep.high - rep.low
    raw_columns, executed_columns = [], []
    for index in range(rep.dimension):
        delta = FD_STEP * width[index]
        lower = theta.copy()
        upper = theta.copy()
        lower[index] = max(rep.low[index], theta[index] - delta)
        upper[index] = min(rep.high[index], theta[index] + delta)
        denominator = upper[index] - lower[index]
        if denominator <= 1e-14:
            raw_columns.append(np.zeros(8))
            executed_columns.append(np.zeros(8))
            continue
        raw_low, exec_low = evaluate(projector, snapshot, rep, lower, positions, goals, u_safe, source_step)
        raw_high, exec_high = evaluate(projector, snapshot, rep, upper, positions, goals, u_safe, source_step)
        raw_columns.append(((raw_high - raw_low) / denominator).reshape(-1))
        executed_columns.append(((exec_high - exec_low) / denominator).reshape(-1))
    raw_matrix = np.stack(raw_columns, axis=1)
    executed_matrix = np.stack(executed_columns, axis=1)
    return {"raw": stats_vector(raw_matrix), "executable": stats_vector(executed_matrix)}


def fit_representation(projector, snapshot, rep, sample, positions, goals, u_safe, target, source_step):
    def objective(theta):
        _, correction = evaluate(projector, snapshot, rep, theta, positions, goals, u_safe, source_step)
        return float(np.linalg.norm(u_safe + correction - target))

    values = []
    for theta in sample:
        values.append(objective(theta))
    embed = rep.embed_p0()
    embed_value = objective(embed)
    best_index = int(np.argmin(values))
    start = sample[best_index]
    optimized = minimize(
        objective,
        start,
        method="Powell",
        bounds=list(zip(rep.low, rep.high, strict=True)),
        options={"maxiter": 100, "maxfev": 400, "xtol": 1e-4, "ftol": 1e-7, "disp": False},
    )
    best_theta = np.asarray(optimized.x if optimized.fun <= values[best_index] else start, dtype=np.float64)
    best_residual = float(min(optimized.fun, values[best_index]))
    raw, executable = evaluate(projector, snapshot, rep, best_theta, positions, goals, u_safe, source_step)
    raw_flat, executable_flat = raw.reshape(-1), executable.reshape(-1)
    raw_norm = float(np.linalg.norm(raw_flat))
    executable_norm = float(np.linalg.norm(executable_flat))
    cosine = float(np.dot(raw_flat, executable_flat) / max(raw_norm * executable_norm, 1e-12))
    return {
        "best_theta": best_theta.tolist(),
        "best_executable_action_residual": best_residual,
        "best_sobol_residual": float(values[best_index]),
        "p0_embedding_residual": embed_value,
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "optimizer_evaluations": int(optimized.nfev),
        "raw_correction_norm": raw_norm,
        "executable_correction_norm": executable_norm,
        "projection_retention_ratio": executable_norm / max(raw_norm, 1e-12),
        "raw_executable_cosine": cosine,
        "fraction_raw_norm_not_retained": 1.0 - executable_norm / max(raw_norm, 1e-12),
        "sensitivity_at_p0_embedding": sensitivity(projector, snapshot, rep, embed, positions, goals, u_safe, source_step),
        "sensitivity_at_best_fit": sensitivity(projector, snapshot, rep, best_theta, positions, goals, u_safe, source_step),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((STUDY / "state_anchor_manifest.json").read_text())
    anchors = [row for index, row in enumerate(manifest["anchors"]) if index % args.shards == args.shard]
    completed = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        completed = {json.loads(line)["state_id"] for line in args.output.read_text().splitlines() if line.strip()}
    datasets = {name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46) for name, path in DATASETS.items()}
    lookup = {(name, family): dataset.by_family[family][0] for name, dataset in datasets.items() for family in dataset.family_names}
    samples = {name: candidates(name, ordinal) for ordinal, name in enumerate(("P0-3D", "P1-Agent6", "P2-Pair8", "P3-Temporal6"))}
    projector = CertifiedHardSafetyFilter()
    expert = CentralizedExpert()
    with args.output.open("a") as handle:
        for count, anchor in enumerate(anchors, 1):
            if anchor["state_id"] in completed:
                continue
            started = time.perf_counter()
            trace_path = HARD / "trajectories/hard_safety" / anchor["set"] / f"rollout_{anchor['rollout_id']:03d}.npz"
            with np.load(trace_path, allow_pickle=False) as trace:
                positions_all = np.asarray(trace["positions"], dtype=np.float64)
                safe_actions = np.asarray(trace["executed_actions"], dtype=np.float64)
            episode = lookup[(anchor["set"], anchor["family_id"])]
            step = int(anchor["source_step"])
            positions = positions_all[step]
            u_safe = safe_actions[step]
            config = Config(**datasets[anchor["set"]].config)
            env = DoubleBottleneckEnv(config)
            env.reset(positions, regime=episode.regime)
            state = env.augmented_state()
            state["last_applied_velocity"] = episode.initial_velocities if step == 0 else safe_actions[step - 1]
            env.restore_augmented_state(state)
            snapshot = env.snapshot()
            base_residual = None
            try:
                plan = expert.plan(env)
                if not plan.success or len(plan.actions) == 0:
                    raise RuntimeError("expert continuation did not succeed")
                expert_raw = np.asarray(plan.actions[0], dtype=np.float64)
                expert_safe = np.asarray(projector(snapshot, expert_raw).velocity, dtype=np.float64)
                base_residual = float(np.linalg.norm(u_safe - expert_safe))
                fits = {
                    name: fit_representation(projector, snapshot, REPRESENTATIONS[name], samples[name], positions, env.goals, u_safe, expert_safe, step)
                    for name in ("P0-3D", "P1-Agent6", "P2-Pair8", "P3-Temporal6")
                }
                result = {
                    **anchor,
                    "expert_recovery_valid": True,
                    "expert_terminal": plan.terminal_reason,
                    "expert_steps": plan.episode_steps,
                    "expert_mode": plan.hypothesis.label,
                    "theta_zero_residual": base_residual,
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
                    "theta_zero_residual": base_residual,
                    "fits": {},
                }
            result["wall_seconds"] = time.perf_counter() - started
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"shard": args.shard, "done": count, "state": anchor["state_id"], "valid": result["expert_recovery_valid"], "seconds": result["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

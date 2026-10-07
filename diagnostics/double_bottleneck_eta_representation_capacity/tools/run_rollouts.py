#!/usr/bin/env python3
"""Resumable rollout evaluator for frozen diagnostic eta representations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import numpy as np

from double_bottleneck.environment import Config
from double_bottleneck.evaluate_flowbc_4a import _initialize_env, _radial_bound64
from double_bottleneck.expert_dataset import infer_coordination_mode
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
CHECKPOINT = ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}
EXPECTED = {
    CHECKPOINT: "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd",
    ROOT / "double_bottleneck/flowbc_4a_agent.py": "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8",
    ROOT / "double_bottleneck/environment.py": "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc",
    ROOT / "shared_control/hard_projection.py": "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79",
    ROOT / "shared_control/diagnostic_corrector.py": "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def category(termination, wall, agent):
    if termination == "collision":
        return "wall_and_agent_collision" if wall and agent else "wall_collision" if wall else "agent_collision" if agent else "collision"
    return "strict_deadlock" if termination == "deadlock" else termination


def run_one(policy, dataset, episode, job):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = CertifiedHardSafetyFilter()
    rep = REPRESENTATIONS[job["representation"]]
    theta = np.asarray(job["theta"], dtype=np.float64)
    key = jax.random.fold_in(jax.random.PRNGKey(int(job["seed"])), int(job["rollout_id"]))
    positions = [env.positions.copy()]
    raw_norms, executable_norms, second_norms, retention, cosine = [], [], [], [], []
    active, substantial, mostly = [], [], []
    first_status, second_status = Counter(), Counter()
    min_wall = float(env.distances()[0].min())
    min_pair = float(env.distances()[1].min())
    final_info = None
    started = time.perf_counter()
    for step in range(config.max_steps):
        step_key = jax.random.fold_in(key, step)
        raw_flow = np.asarray(policy.sample_actions(env.observation()[None], step_key)[0], dtype=np.float64)
        u_flow = _radial_bound64(raw_flow, config.max_speed)
        first = projector(env.snapshot(), u_flow)
        u_safe = np.asarray(first.velocity, dtype=np.float64)
        g = rep.correction(theta, env.positions, env.goals, u_safe, config.max_speed, step, config.max_steps)
        second = projector(env.snapshot(), u_safe + g)
        u_exec = np.asarray(second.velocity, dtype=np.float64)
        executed = u_exec - u_safe
        raw_norm = float(np.linalg.norm(g))
        executable_norm = float(np.linalg.norm(executed))
        second_norm = float(np.linalg.norm(u_exec - (u_safe + g)))
        raw_norms.append(raw_norm)
        executable_norms.append(executable_norm)
        second_norms.append(second_norm)
        retention.append(executable_norm / max(raw_norm, 1e-12))
        cosine.append(float(np.sum(g * executed) / max(raw_norm * executable_norm, 1e-12)))
        active.append(second_norm > 1e-6)
        substantial.append(raw_norm > 1e-6 and second_norm >= 0.25 * raw_norm)
        mostly.append(raw_norm > 1e-6 and second_norm >= 0.75 * raw_norm)
        first_status[str(first.status)] += 1
        second_status[str(second.status)] += 1
        _, _, done, info = env.step(u_exec)
        positions.append(env.positions.copy())
        min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
        min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        final_info = info
        if done:
            break
    termination = str(final_info["termination"])
    wall, agent = bool(final_info["wall_collision"]), bool(final_info["agent_collision"])
    positions = np.asarray(positions)
    mode = infer_coordination_mode(positions, env.goals, config)
    return {
        **{key: job[key] for key in ("job_id", "stage", "representation", "parameter_id", "sample_type", "theta", "episode_id", "population", "pilot_stratum", "set", "family_id", "regime", "seed", "rollout_id", "baseline_outcome") if key in job},
        "termination": termination,
        "outcome": category(termination, wall, agent),
        "success": termination == "success",
        "wall_collision": wall,
        "agent_collision": agent,
        "episode_steps": len(raw_norms),
        "minimum_wall_clearance": min_wall,
        "minimum_agent_clearance": min_pair,
        "final_goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1).tolist(),
        "coordination_mode": mode,
        "correction": {
            "raw_norm": summary(raw_norms),
            "executable_norm": summary(executable_norms),
            "second_projection_norm": summary(second_norms),
            "retention_ratio": summary(retention),
            "raw_executable_cosine": summary(cosine),
            "second_projection_active_fraction": float(np.mean(active)),
            "substantially_altered_fraction": float(np.mean(substantial)),
            "mostly_rewritten_fraction": float(np.mean(mostly)),
            "first_status_counts": dict(first_status),
            "second_status_counts": dict(second_status),
        },
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    manifest = json.loads(args.manifest.read_text())
    jobs = [job for index, job in enumerate(manifest["jobs"]) if index % args.shards == args.shard]
    completed = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        completed = {json.loads(line)["job_id"] for line in args.output.read_text().splitlines() if line.strip()}
    datasets = {name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46) for name, path in DATASETS.items()}
    fingerprints = {dataset.environment_fingerprint for dataset in datasets.values()}
    policy, _ = load_checkpoint(CHECKPOINT, next(iter(fingerprints)))
    lookup = {(name, family): dataset.by_family[family][0] for name, dataset in datasets.items() for family in dataset.family_names}
    pending = [job for job in jobs if job["job_id"] not in completed]
    started = time.perf_counter()
    with args.output.open("a") as handle:
        for index, job in enumerate(pending, 1):
            result = run_one(policy, datasets[job["set"]], lookup[(job["set"], job["family_id"])], job)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(json.dumps({"shard": args.shard, "done": index, "pending": len(pending), "job": job["job_id"], "outcome": result["outcome"], "elapsed": time.perf_counter() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

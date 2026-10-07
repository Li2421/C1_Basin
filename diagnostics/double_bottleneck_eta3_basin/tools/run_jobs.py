#!/usr/bin/env python3
"""Resumable frozen 3D-eta rollout evaluator for pre-registered job manifests."""

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
os.environ.setdefault("OMP_NUM_THREADS", "2")

import jax
import numpy as np

from double_bottleneck.environment import Config
from double_bottleneck.evaluate_flowbc_4a import _initialize_env, _radial_bound64
from double_bottleneck.expert_dataset import infer_coordination_mode
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from shared_control.diagnostic_corrector import DiagnosticCorrector
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import (
    CertifiedHardSafetyFilter,
)


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"
CHECKPOINT = ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
DATASETS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}
EXPECTED_HASHES = {
    CHECKPOINT: "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd",
    ROOT / "double_bottleneck/flowbc_4a_agent.py": "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8",
    ROOT / "double_bottleneck/environment.py": "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc",
    ROOT / "shared_control/hard_projection.py": "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79",
    ROOT / "shared_control/diagnostic_corrector.py": "TO_BE_FROZEN_BY_SETUP",
}
ACTIVE_TOL = 1e-6
SUBSTANTIAL_RATIO = 0.25
MOSTLY_REWRITTEN_RATIO = 0.75


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "p90": float(np.percentile(values, 90)) if len(values) else 0.0,
        "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
        "p99": float(np.percentile(values, 99)) if len(values) else 0.0,
        "maximum": float(values.max()) if len(values) else 0.0,
    }


def outcome(termination: str, wall: bool, agent: bool) -> str:
    if termination == "collision":
        if wall and agent:
            return "wall_and_agent_collision"
        return "wall_collision" if wall else "agent_collision" if agent else "collision"
    return "strict_deadlock" if termination == "deadlock" else termination


def run_one(policy, dataset, episode, job):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = CertifiedHardSafetyFilter()
    corrector = DiagnosticCorrector(np.asarray(job["eta"], dtype=np.float64))
    episode_key = jax.random.fold_in(jax.random.PRNGKey(int(job["seed"])), int(job["rollout_id"]))
    positions = [env.positions.copy()]
    executed_actions = []
    first_correction = []
    g_norm = []
    goal_basis_norm = []
    safe_basis_norm = []
    relative_basis_norm = []
    second_correction = []
    actual_eta_delta = []
    rewrite_ratio = []
    second_active = []
    substantial = []
    mostly_rewritten = []
    first_status = Counter()
    second_status = Counter()
    speeds = []
    min_wall = float(env.distances()[0].min())
    min_pair = float(env.distances()[1].min())
    final_info = None
    started = time.perf_counter()
    for step in range(config.max_steps):
        step_key = jax.random.fold_in(episode_key, step)
        raw_flow = np.asarray(
            policy.sample_actions(env.observation()[None], step_key)[0], dtype=np.float64
        )
        u_flow = _radial_bound64(raw_flow, config.max_speed)
        first = projector(env.snapshot(), u_flow)
        u_safe = np.asarray(first.velocity, dtype=np.float64)
        g_raw, b_goal, b_rel = corrector(
            env.positions, env.goals, u_safe, config.max_speed
        )
        candidate = u_safe + g_raw
        second = projector(env.snapshot(), candidate)
        u_exec = np.asarray(second.velocity, dtype=np.float64)
        first_delta = float(np.linalg.norm(u_safe - u_flow))
        second_delta = float(np.linalg.norm(u_exec - candidate))
        raw_g = float(np.linalg.norm(g_raw))
        ratio = second_delta / max(raw_g, 1e-12)
        _, _, done, info = env.step(u_exec)
        positions.append(env.positions.copy())
        executed_actions.append(u_exec)
        first_correction.append(first_delta)
        g_norm.append(raw_g)
        goal_basis_norm.append(float(np.linalg.norm(b_goal)))
        safe_basis_norm.append(float(np.linalg.norm(u_safe)))
        relative_basis_norm.append(float(np.linalg.norm(b_rel)))
        second_correction.append(second_delta)
        actual_eta_delta.append(float(np.linalg.norm(u_exec - u_safe)))
        rewrite_ratio.append(ratio)
        second_active.append(second_delta > ACTIVE_TOL)
        substantial.append(raw_g > ACTIVE_TOL and ratio >= SUBSTANTIAL_RATIO)
        mostly_rewritten.append(raw_g > ACTIVE_TOL and ratio >= MOSTLY_REWRITTEN_RATIO)
        first_status[str(first.status)] += 1
        second_status[str(second.status)] += 1
        speeds.append(np.linalg.norm(u_exec, axis=-1))
        min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
        min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        final_info = info
        if done:
            break
    if final_info is None or not env.done:
        raise AssertionError("eta rollout did not reach a runtime terminal event")
    positions = np.asarray(positions, dtype=np.float64)
    executed_actions = np.asarray(executed_actions, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    termination = str(final_info["termination"])
    wall = bool(final_info["wall_collision"])
    agent = bool(final_info["agent_collision"])
    mode = infer_coordination_mode(positions, env.goals, config)
    result = {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "sample_type": job["sample_type"],
        "eta_id": job["eta_id"],
        "eta": [float(value) for value in job["eta"]],
        "population": job["population"],
        "set": job["set"],
        "family_id": job["family_id"],
        "regime": job["regime"],
        "seed": int(job["seed"]),
        "rollout_id": int(job["rollout_id"]),
        "baseline_outcome": job["baseline_outcome"],
        "termination": termination,
        "outcome": outcome(termination, wall, agent),
        "success": termination == "success",
        "wall_collision": wall,
        "agent_collision": agent,
        "runtime_strict_deadlock": termination == "deadlock",
        "timeout": termination == "timeout",
        "episode_steps": len(executed_actions),
        "completion_time_seconds": (
            len(executed_actions) * config.dt if termination == "success" else None
        ),
        "minimum_wall_clearance": min_wall,
        "minimum_agent_clearance": min_pair,
        "final_goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1).tolist(),
        "path_length": float(np.linalg.norm(executed_actions, axis=-1).sum() * config.dt),
        "per_agent_waiting_seconds": (
            np.mean(speeds < 0.025, axis=0) * len(executed_actions) * config.dt
        ).tolist(),
        "coordination_mode": mode,
        "correction": {
            "g_raw_joint_norm": stats(g_norm),
            "g_raw_mean_squared_norm": float(np.mean(np.square(g_norm))),
            "goal_basis_joint_norm": stats(goal_basis_norm),
            "safe_basis_joint_norm": stats(safe_basis_norm),
            "relative_basis_joint_norm": stats(relative_basis_norm),
            "first_projection_correction_norm": stats(first_correction),
            "second_projection_correction_norm": stats(second_correction),
            "executed_eta_delta_norm": stats(actual_eta_delta),
            "second_projection_to_g_ratio": stats(rewrite_ratio),
            "second_projection_active_fraction": float(np.mean(second_active)),
            "substantially_altered_fraction": float(np.mean(substantial)),
            "mostly_rewritten_fraction": float(np.mean(mostly_rewritten)),
            "substantial_definition": "||u_exec-(u_safe+g)|| >= 0.25||g|| and ||g||>1e-6",
            "mostly_rewritten_definition": "||u_exec-(u_safe+g)|| >= 0.75||g|| and ||g||>1e-6",
            "first_projection_status_counts": dict(sorted(first_status.items())),
            "second_projection_status_counts": dict(sorted(second_status.items())),
        },
        "rng_protocol": "fold_in(fold_in(PRNGKey(seed), rollout_id), absolute_step)",
        "eta_fixed_for_episode": True,
        "second_hard_projection": True,
        "g_phi": False,
        "rollout_wall_seconds": time.perf_counter() - started,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    protocol = json.loads((STUDY / "PREREGISTRATION.json").read_text())
    expected_corrector = protocol["frozen_hashes"]["diagnostic_corrector"]
    for path, expected in EXPECTED_HASHES.items():
        expected_value = expected_corrector if expected == "TO_BE_FROZEN_BY_SETUP" else expected
        if sha(path) != expected_value:
            raise RuntimeError(f"frozen input changed: {path}")
    manifest = json.loads(args.manifest.read_text())
    if manifest["preregistration_sha256"] != sha(STUDY / "PREREGISTRATION.json"):
        raise RuntimeError("manifest/preregistration mismatch")
    jobs = [
        job
        for index, job in enumerate(manifest["jobs"])
        if index % args.shards == args.shard
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["job_id"])
    datasets = {
        name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46)
        for name, path in DATASETS.items()
    }
    fingerprints = {dataset.environment_fingerprint for dataset in datasets.values()}
    if len(fingerprints) != 1:
        raise RuntimeError("evaluation datasets disagree on environment")
    policy, _ = load_checkpoint(CHECKPOINT, next(iter(fingerprints)))
    if jax.default_backend() != "cpu":
        raise RuntimeError("eta basin evaluation is frozen to CPU for this run")
    lookup = {
        (set_name, family): dataset.by_family[family][0]
        for set_name, dataset in datasets.items()
        for family in dataset.family_names
    }
    pending = [job for job in jobs if job["job_id"] not in completed]
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as handle:
        for index, job in enumerate(pending, 1):
            episode = lookup[(job["set"], job["family_id"])]
            result = run_one(policy, datasets[job["set"]], episode, job)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {
                            "shard": args.shard,
                            "done": index,
                            "pending": len(pending),
                            "job": job["job_id"],
                            "outcome": result["outcome"],
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

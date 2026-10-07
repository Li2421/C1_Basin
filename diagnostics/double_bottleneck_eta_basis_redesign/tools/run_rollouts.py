#!/usr/bin/env python3
"""Resumable frozen rollout evaluator for isolated P0/OrthoFlow3 diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import pickle
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
from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import correction
from shared_control.hard_projection import CBFSolverError


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
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


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "maximum": float(values.max())}


def outcome_name(termination, wall, agent):
    if termination == "collision":
        return "wall_and_agent_collision" if wall and agent else "wall_collision" if wall else "agent_collision" if agent else "collision"
    return "strict_deadlock" if termination == "deadlock" else termination


def frozen_digest(value) -> str:
    """Identity check for the exact state/input reused by numerical retries."""
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def project_with_retries(projector, snapshot, nominal_velocity, *, step, stage):
    """Run the frozen projection, permitting at most three identical retries.

    Each retry receives the same snapshot object and a byte-identical nominal
    action.  A retry changes neither the optimization problem nor acceptance
    certificate.  Returning ``None`` is an execution failure, not a task
    outcome.
    """
    state_hash = frozen_digest(snapshot)
    nominal = np.asarray(nominal_velocity, dtype=np.float64).copy()
    nominal_hash = hashlib.sha256(nominal.tobytes()).hexdigest()
    failures = []
    for attempt in range(4):  # initial invocation plus at most 3 additional retries
        if frozen_digest(snapshot) != state_hash or hashlib.sha256(nominal.tobytes()).hexdigest() != nominal_hash:
            raise RuntimeError("frozen projection retry input mutated")
        try:
            result = projector(snapshot, nominal)
        except CBFSolverError as error:
            failures.append({
                "attempt": attempt,
                "status": error.status,
                "max_constraint_violation": error.details.get("max_constraint_violation"),
                "stationarity_residual": error.details.get("stationarity_residual"),
                "complementarity_residual": error.details.get("complementarity_residual"),
            })
            continue
        return result, ({
            "step": int(step), "projection_stage": stage,
            "additional_retries": int(attempt), "state_sha256": state_hash,
            "nominal_sha256": nominal_hash, "failed_attempts": failures,
        } if attempt else None)
    return None, {
        "step": int(step), "projection_stage": stage,
        "additional_retries": 3, "state_sha256": state_hash,
        "nominal_sha256": nominal_hash, "failed_attempts": failures,
        "final_status": "NUMERICAL_SOLVER_FAILURE",
    }


def run_one(policy, dataset, episode, job, scale):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = CertifiedHardSafetyFilter()
    theta = np.asarray(job["theta"], dtype=float)
    key = jax.random.fold_in(jax.random.PRNGKey(int(job["seed"])), int(job["rollout_id"]))
    raw_norms, executable_norms, removal_norms, removal_ratios, cosines, more_half = [], [], [], [], [], []
    first_status, second_status = Counter(), Counter()
    min_wall = float(env.distances()[0].min())
    min_pair = float(env.distances()[1].min())
    positions = [env.positions.copy()]
    final_info = None
    retry_events = []
    started = time.perf_counter()
    for step in range(config.max_steps):
        raw_flow = np.asarray(policy.sample_actions(env.observation()[None], jax.random.fold_in(key, step))[0], dtype=float)
        u_flow = _radial_bound64(raw_flow, config.max_speed)
        snapshot = env.snapshot()
        first, retry = project_with_retries(projector, snapshot, u_flow, step=step, stage="first_projection")
        if retry is not None:
            retry_events.append(retry)
        if first is None:
            return numerical_failure_result(job, env, raw_norms, retry_events, min_wall, min_pair, started)
        u_safe = np.asarray(first.velocity, dtype=float)
        g = correction(job["representation"], theta, env.positions, env.goals, u_safe, config.max_speed, scale)
        second, retry = project_with_retries(projector, snapshot, u_safe + g, step=step, stage="second_projection")
        if retry is not None:
            retry_events.append(retry)
        if second is None:
            return numerical_failure_result(job, env, raw_norms, retry_events, min_wall, min_pair, started)
        u_exec = np.asarray(second.velocity, dtype=float)
        executable = u_exec - u_safe
        raw_norm = float(np.linalg.norm(g))
        executable_norm = float(np.linalg.norm(executable))
        removal_norm = float(np.linalg.norm(u_exec - (u_safe + g)))
        raw_norms.append(raw_norm)
        executable_norms.append(executable_norm)
        removal_norms.append(removal_norm)
        removal_ratios.append(removal_norm / max(raw_norm, 1e-12))
        cosines.append(float(np.sum(g * executable) / max(raw_norm * executable_norm, 1e-12)))
        more_half.append(raw_norm > 1e-12 and removal_norm > 0.5 * raw_norm)
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
    wall = bool(final_info["wall_collision"])
    agent_collision = bool(final_info["agent_collision"])
    return {
        **{key: job[key] for key in ("job_id", "stage", "representation", "parameter_id", "eta_index", "sample_type", "theta", "episode_id", "population", "set", "family_id", "regime", "seed", "rollout_id", "baseline_outcome", "center_eta_index", "local_index") if key in job},
        "success": termination == "success",
        "termination": termination,
        "outcome": outcome_name(termination, wall, agent_collision),
        "episode_steps": len(raw_norms),
        "wall_collision": wall,
        "agent_collision": agent_collision,
        "minimum_wall_clearance": min_wall,
        "minimum_agent_clearance": min_pair,
        "final_goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1).tolist(),
        "coordination_mode": infer_coordination_mode(np.asarray(positions), env.goals, config),
        "correction": {
            "raw_norm": summary(raw_norms),
            "executable_norm": summary(executable_norms),
            "projection_removal_norm": summary(removal_norms),
            "removal_ratio": summary(removal_ratios),
            "raw_executable_cosine": summary(cosines),
            "more_than_half_removed_fraction": float(np.mean(more_half)),
            "first_status_counts": dict(first_status),
            "second_status_counts": dict(second_status),
        },
        "wall_seconds": time.perf_counter() - started,
        "scientific_outcome_valid": True,
        "solver_retry_occurred": bool(retry_events),
        "solver_retry_events": retry_events,
    }


def numerical_failure_result(job, env, raw_norms, retry_events, min_wall, min_pair, started):
    """Separate execution record: never coerced into a task outcome."""
    return {
        **{key: job[key] for key in ("job_id", "stage", "representation", "parameter_id", "eta_index", "sample_type", "theta", "episode_id", "population", "set", "family_id", "regime", "seed", "rollout_id", "baseline_outcome", "center_eta_index", "local_index") if key in job},
        "success": None,
        "termination": "NUMERICAL_SOLVER_FAILURE",
        "outcome": "NUMERICAL_SOLVER_FAILURE",
        "scientific_outcome_valid": False,
        "episode_steps_before_solver_failure": len(raw_norms),
        "wall_collision": None,
        "agent_collision": None,
        "minimum_wall_clearance_before_solver_failure": min_wall,
        "minimum_agent_clearance_before_solver_failure": min_pair,
        "final_goal_errors_before_solver_failure": np.linalg.norm(env.goals - env.positions, axis=-1).tolist(),
        "solver_retry_occurred": True,
        "solver_retry_events": retry_events,
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
    scale = float(json.loads((OUT / "P1_SCALE.json").read_text())["scale"])
    manifest = json.loads(args.manifest.read_text())
    jobs = [job for index, job in enumerate(manifest["jobs"]) if index % args.shards == args.shard]
    completed = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        completed = {json.loads(line)["job_id"] for line in args.output.read_text().splitlines() if line.strip()}
    # Global cache preflight: materialize only semantically exact, conflict-free
    # rows. Ambiguous records are ignored and therefore remain pending.
    try:
        from shared_rollout_db.src.integration import reuse_db_jobs
        cached = reuse_db_jobs([job for job in jobs if job['job_id'] not in completed])
        if cached:
            with args.output.open('a') as handle:
                for job_id,row in cached.items(): handle.write(json.dumps(row,sort_keys=True)+'\n')
            completed.update(cached)
            print(json.dumps({'global_cache_exact_reuse':len(cached)}),flush=True)
    except Exception as error:
        print(json.dumps({'global_cache_preflight':'unavailable','reason':str(error)}),flush=True)
    datasets = {name: FlowBC4ADataset(path, "val", seed=45 if name.startswith("existing") else 46) for name, path in DATASETS.items()}
    policy, _ = load_checkpoint(CHECKPOINT, next(iter({dataset.environment_fingerprint for dataset in datasets.values()})))
    lookup = {(name, family): dataset.by_family[family][0] for name, dataset in datasets.items() for family in dataset.family_names}
    pending = [job for job in jobs if job["job_id"] not in completed]
    started = time.perf_counter()
    new_global_records=[]
    with args.output.open("a") as handle:
        for count, job in enumerate(pending, 1):
            try:
                result = run_one(policy, datasets[job["set"]], lookup[(job["set"], job["family_id"])], job, scale)
            except CBFSolverError as error:
                # An execution error is NOT a scientific terminal outcome.
                # Preserve it separately; continue independent, registered jobs.
                with (OUT / "logs" / (args.output.stem + ".errors.jsonl")).open("a") as errors:
                    errors.write(json.dumps({"job": job, "status": error.status, "details": error.details}, sort_keys=True) + "\n")
                print(json.dumps({"job_id": job["job_id"], "execution_error": error.status}), flush=True)
                continue
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            new_global_records.append(result)
            if count == 1 or count % 10 == 0 or count == len(pending):
                print(json.dumps({"shard": args.shard, "done": count, "pending": len(pending), "outcome": result["outcome"], "elapsed": time.perf_counter() - started}), flush=True)
    if new_global_records:
        try:
            from shared_rollout_db.src.integration import journal_db
            journal_db(new_global_records,OUT)
        except Exception as error:
            print(json.dumps({'global_cache_journal':'failed','reason':str(error)}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

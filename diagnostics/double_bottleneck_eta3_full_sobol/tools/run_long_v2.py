#!/usr/bin/env python3
"""Resumable detailed P0 rollout evaluator for the unattended protocol."""

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


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "maximum": float(values.max())}


def category(termination, wall, agent):
    if termination == "collision":
        return "wall_collision" if wall and not agent else "agent_collision" if agent and not wall else "wall_and_agent_collision"
    return "strict_deadlock" if termination == "deadlock" else termination


def run_one(policy, dataset, episode, job):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    projector = CertifiedHardSafetyFilter()
    theta = np.asarray(job["theta"], dtype=np.float64)
    key = jax.random.fold_in(jax.random.PRNGKey(int(job["seed"])), int(job["rollout_id"]))
    initial_positions = env.positions.copy()
    initial_goal_errors = np.linalg.norm(env.goals - env.positions, axis=-1)
    directions = -np.sign(initial_positions[:, 0]).astype(int)
    first_planes = np.where(directions > 0, -0.79, 0.79)
    second_planes = np.where(directions > 0, 2.01, -2.01)
    first_cross = np.full(4, -1, dtype=int); second_cross = np.full(4, -1, dtype=int); goal_entry = np.full(4, -1, dtype=int)
    positions = [env.positions.copy()]
    raw_norms, exec_norms, removed_norms, removal_ratios, cosines, more_half = [], [], [], [], [], []
    waiting_steps = np.zeros(4, dtype=int)
    min_wall = float(env.distances()[0].min()); min_pair = float(env.distances()[1].min())
    first_status, second_status = Counter(), Counter()
    final_info = None; started = time.perf_counter()
    for step in range(config.max_steps):
        before = env.positions.copy()
        raw_flow = np.asarray(policy.sample_actions(env.observation()[None], jax.random.fold_in(key, step))[0], dtype=np.float64)
        u_flow = _radial_bound64(raw_flow, config.max_speed)
        first = projector(env.snapshot(), u_flow); u_safe = np.asarray(first.velocity, dtype=np.float64)
        g = REPRESENTATIONS["P0-3D"].correction(theta, env.positions, env.goals, u_safe, config.max_speed, step, config.max_steps)
        second = projector(env.snapshot(), u_safe + g); u_exec = np.asarray(second.velocity, dtype=np.float64)
        executed = u_exec - u_safe
        raw_n = float(np.linalg.norm(g)); exec_n = float(np.linalg.norm(executed)); removed_n = float(np.linalg.norm(u_exec-(u_safe+g)))
        raw_norms.append(raw_n); exec_norms.append(exec_n); removed_norms.append(removed_n)
        removal_ratios.append(removed_n / max(raw_n, 1e-12)); more_half.append(raw_n > 1e-12 and removed_n > 0.5*raw_n)
        cosines.append(float(np.sum(g*executed)/max(raw_n*exec_n,1e-12)))
        first_status[str(first.status)] += 1; second_status[str(second.status)] += 1
        _, _, done, info = env.step(u_exec); positions.append(env.positions.copy())
        speed = np.linalg.norm(env.positions-before, axis=-1)/config.dt; waiting_steps += speed < 0.025
        crossed_first = directions*(env.positions[:,0]-first_planes) >= 0
        crossed_second = directions*(env.positions[:,0]-second_planes) >= 0
        entered = np.linalg.norm(env.goals-env.positions, axis=-1) <= config.goal_tolerance
        for agent in range(4):
            if first_cross[agent] < 0 and crossed_first[agent]: first_cross[agent] = step+1
            if second_cross[agent] < 0 and crossed_second[agent]: second_cross[agent] = step+1
            if goal_entry[agent] < 0 and entered[agent]: goal_entry[agent] = step+1
        min_wall = min(min_wall, float(info["min_swept_wall_distance"])); min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        final_info = info
        if done: break
    termination = str(final_info["termination"]); wall = bool(final_info["wall_collision"]); agent_collision = bool(final_info["agent_collision"])
    final_goal_errors = np.linalg.norm(env.goals-env.positions, axis=-1)
    seconds = lambda values: [float(value*config.dt) if value >= 0 else None for value in values]
    return {
        **{k: job[k] for k in ("job_id","stage","representation","parameter_id","eta_index","sample_type","theta","episode_id","population","set","family_id","regime","seed","rollout_id","baseline_outcome","center_eta_index") if k in job},
        "success": termination == "success", "termination": termination, "outcome": category(termination,wall,agent_collision),
        "episode_steps": len(raw_norms), "episode_seconds": len(raw_norms)*config.dt, "wall_collision": wall, "agent_collision": agent_collision,
        "minimum_wall_clearance": min_wall, "minimum_agent_clearance": min_pair,
        "timing": {
            "first_bottleneck_crossing_seconds_per_agent": seconds(first_cross),
            "second_bottleneck_crossing_seconds_per_agent": seconds(second_cross),
            "time_first_bottleneck_clears": float(first_cross.max()*config.dt) if np.all(first_cross>=0) else None,
            "time_second_bottleneck_clears": float(second_cross.max()*config.dt) if np.all(second_cross>=0) else None,
            "goal_entry_seconds_per_agent": seconds(goal_entry),
            "final_goal_entry_time": float(goal_entry.max()*config.dt) if np.all(goal_entry>=0) else None,
            "completion_time": len(raw_norms)*config.dt if termination == "success" else None,
            "waiting_seconds_per_agent": (waiting_steps*config.dt).tolist(),
            "total_waiting_agent_seconds": float(waiting_steps.sum()*config.dt),
            "initial_total_goal_error": float(initial_goal_errors.sum()),
            "final_total_goal_error": float(final_goal_errors.sum()),
            "total_progress": float(initial_goal_errors.sum()-final_goal_errors.sum()),
        },
        "final_goal_errors": final_goal_errors.tolist(),
        "coordination_mode": infer_coordination_mode(np.asarray(positions), env.goals, config),
        "correction": {
            "raw_norm": summary(raw_norms), "executable_norm": summary(exec_norms), "projection_removal_norm": summary(removed_norms),
            "removal_ratio": summary(removal_ratios), "raw_executable_cosine": summary(cosines),
            "more_than_half_removed_fraction": float(np.mean(more_half)), "first_status_counts": dict(first_status), "second_status_counts": dict(second_status),
        },
        "wall_seconds": time.perf_counter()-started,
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--shard",type=int,required=True); parser.add_argument("--shards",type=int,required=True); args=parser.parse_args()
    for path, expected in EXPECTED.items():
        if sha(path)!=expected: raise RuntimeError(f"frozen input changed: {path}")
    manifest=json.loads(args.manifest.read_text()); jobs=[job for i,job in enumerate(manifest["jobs"]) if i%args.shards==args.shard]
    args.output.parent.mkdir(parents=True,exist_ok=True); completed=set()
    if args.output.exists(): completed={json.loads(line)["job_id"] for line in args.output.read_text().splitlines() if line.strip()}
    datasets={name:FlowBC4ADataset(path,"val",seed=45 if name.startswith("existing") else 46) for name,path in DATASETS.items()}
    policy,_=load_checkpoint(CHECKPOINT,next(iter({d.environment_fingerprint for d in datasets.values()})))
    lookup={(name,family):d.by_family[family][0] for name,d in datasets.items() for family in d.family_names}
    pending=[job for job in jobs if job["job_id"] not in completed]; started=time.perf_counter()
    with args.output.open("a") as handle:
        for index,job in enumerate(pending,1):
            result=run_one(policy,datasets[job["set"]],lookup[(job["set"],job["family_id"])],job); handle.write(json.dumps(result,sort_keys=True)+"\n"); handle.flush()
            if index==1 or index%10==0 or index==len(pending): print(json.dumps({"shard":args.shard,"done":index,"pending":len(pending),"outcome":result["outcome"],"elapsed":time.perf_counter()-started}),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())

"""Audit frozen Flow action samples at a saved non-opposing terminal stall."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.hard_projection import HardProjectionConfig

from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .observation import policy_observation_competence
from .scenario import Config


def run(trace: Path, dataset: Path, checkpoint: Path, output: Path, *, anchor: int,
        count: int = 64, evaluation_seed: int = 17, episode_index: int = 0):
    trace, dataset, checkpoint, output = map(Path, (trace, dataset, checkpoint, output))
    manifest = json.loads((dataset/"manifest.json").read_text())
    with np.load(trace, allow_pickle=False) as data:
        positions, goals = np.asarray(data["positions"]), np.asarray(data["goals"])
        executed = np.asarray(data["u_safe"])
        meta = json.loads(str(data["metadata_json"].item()))
    if anchor <= 0 or anchor >= len(executed):
        raise ValueError("anchor must be a saved nonterminal action step")
    config = Config(**meta["config"])
    env = BottleneckEnv(replace(config, seed=int(meta["seed"]), split="dev"))
    env.reset(positions[anchor], goals)
    env.velocities = executed[anchor-1]
    mode = meta["control_mode"]
    if not mode.startswith("temporal_release_"):
        raise ValueError("this terminal diagnostic requires a temporal-release trace")
    first = np.arange(0 if mode.endswith("even_first") else 1, config.num_agents, 2)
    second = np.setdiff1d(np.arange(config.num_agents), first)
    delta = goals-positions[anchor]
    distance = np.linalg.norm(delta, axis=1)
    pending = first[distance[first] > config.goal_tolerance]
    if not len(pending):
        raise ValueError("no unresolved first-group agents at the chosen step")
    unit = delta/np.maximum(distance[:, None], 1e-12)
    agent, _ = load_checkpoint(checkpoint,
        expected_environment_fingerprint=manifest["environment_fingerprint"])
    observation = policy_observation_competence(env)
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(evaluation_seed),
                                               episode_index), anchor)
    commands = np.asarray(sample_bounded_actions(agent,
        np.repeat(observation[None], count, axis=0), key), dtype=np.float64)
    radial = np.sum(commands[:, pending]*unit[pending], axis=2)
    teacher = teacher_reference(env)
    teacher[second] = 0
    expert_safe = np.asarray(CertifiedHardSafetyFilter(HardProjectionConfig())(
        env.snapshot(), teacher).velocity)
    report = dict(schema="gap1_n50_terminal_flow_latent_audit_v1",
        trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256((dataset/"manifest.json").read_bytes()).hexdigest(),
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        anchor=anchor,count=count,evaluation_seed=evaluation_seed,
        episode_index=episode_index,pending_agents=pending.tolist(),
        pending_goal_distances=distance[pending].tolist(),
        flow_goalward_mean_per_agent=np.mean(radial,axis=0).tolist(),
        flow_goalward_quantiles_per_agent=np.quantile(radial,[.1,.5,.9],axis=0).tolist(),
        flow_goalward_mean_all=float(np.mean(radial)),
        fraction_all_pending_goalward=float(np.mean(np.all(radial>0,axis=1))),
        expert_projected_goalward_per_agent=np.sum(expert_safe[pending]*unit[pending],axis=1).tolist())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--anchor",type=int,required=True)
    parser.add_argument("--count",type=int,default=64)
    args=parser.parse_args()
    print(json.dumps(run(args.trace,args.dataset,args.checkpoint,args.output,
                         anchor=args.anchor,count=args.count),indent=2))


if __name__ == "__main__":
    main()

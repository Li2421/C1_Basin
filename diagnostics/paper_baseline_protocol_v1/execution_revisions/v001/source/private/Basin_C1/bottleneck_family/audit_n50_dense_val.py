"""Held-out dense post-gate one-step check for N=50 Flow snapshots.

Only non-opposing expert recoveries from validation states are read. This is
for competence checkpoint screening; full-horizon rollouts remain mandatory.
"""
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
from .environment import BottleneckEnv
from .observation import policy_observation_competence
from .scenario import Config


def run(dataset: Path, recoveries: list[Path], checkpoints: list[Path], output: Path,
        *, seed=847, samples_per_recovery=8):
    dataset, output = Path(dataset), Path(output)
    manifest=json.loads((dataset/"manifest.json").read_text())
    config=Config(**manifest["scenario_config"])
    cases=[]
    for path in map(Path,recoveries):
        with np.load(path,allow_pickle=False) as data:
            states=np.asarray(data["states"])
            actions=np.asarray(data["actions"])
            goals=np.asarray(data["goals"])
        # Compare the first 100 continuation steps, when the dense unfinished
        # population still exists. Uniformly spanning the whole successful
        # continuation mostly samples the easy settled tail.
        indices=np.unique(np.linspace(0,min(len(actions)-1,100),
                       min(samples_per_recovery,len(actions)),dtype=int))
        for t in indices:
            env=BottleneckEnv(replace(config,split="dev",seed=seed))
            env.reset(states[t],goals)
            env.velocities=actions[t-1] if t else np.zeros_like(actions[t])
            d=np.linalg.norm(goals-states[t],axis=1)
            mask=(states[t,:,0]>.55)&(d>config.goal_tolerance)
            if np.sum(mask)<5:
                continue
            cases.append((path,t,env,actions[t],mask))
    if not cases:
        raise ValueError("no dense postgate validation states")
    observations=np.asarray([policy_observation_competence(case[2]) for case in cases])
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    result={}
    for checkpoint in map(Path,checkpoints):
        agent,_=load_checkpoint(checkpoint,
            expected_environment_fingerprint=manifest["environment_fingerprint"])
        predicted=np.asarray(sample_bounded_actions(agent,observations,
                                                    jax.random.PRNGKey(seed)),dtype=np.float64)
        rows=[]
        for index,(path,t,env,expert,mask) in enumerate(cases):
            flow=predicted[index]
            safe=np.asarray(projector(env.snapshot(),flow).velocity)
            delta=env.goals-env.positions
            distance=np.linalg.norm(delta,axis=1)
            unit=delta/np.maximum(distance[:,None],1e-12)
            radial=lambda command:np.sum(command[mask]*unit[mask],axis=1)
            rows.append(dict(recovery=str(path),step=int(t),pending=int(mask.sum()),
                expert_goalward=float(np.mean(radial(expert))),
                flow_goalward=float(np.mean(radial(flow))),
                safe_goalward=float(np.mean(radial(safe))),
                flow_speed=float(np.mean(np.linalg.norm(flow[mask],axis=1))),
                active_agent_correction=float(np.mean(np.linalg.norm(safe[mask]-flow[mask],axis=1))),
                one_step_distance_change=float(np.mean(np.linalg.norm(
                    delta[mask]-config.dt*safe[mask],axis=1)-distance[mask]))))
        result[f"{checkpoint.parent.name}/{checkpoint.stem}"]=dict(checkpoint=str(checkpoint),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            count=len(rows),mean_expert_goalward=float(np.mean([r["expert_goalward"] for r in rows])),
            mean_flow_goalward=float(np.mean([r["flow_goalward"] for r in rows])),
            mean_safe_goalward=float(np.mean([r["safe_goalward"] for r in rows])),
            mean_active_agent_correction=float(np.mean([r["active_agent_correction"] for r in rows])),
            rows=rows)
    report=dict(schema="gap1_n50_dense_dev_one_step_v1",seed=seed,
                samples_per_recovery=samples_per_recovery,results=result,
                caveat="One-step held-out validation only; not an episode competence gate")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--recovery",type=Path,action="append",required=True)
    parser.add_argument("--checkpoint",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=847)
    args=parser.parse_args()
    report=run(args.dataset,args.recovery,args.checkpoint,args.output,seed=args.seed)
    print(json.dumps({k:{key:v[key] for key in ("count","mean_expert_goalward",
        "mean_flow_goalward","mean_safe_goalward","mean_active_agent_correction")}
        for k,v in report["results"].items()},indent=2),flush=True)


if __name__=="__main__":
    main()

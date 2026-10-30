"""Held-out one-step N=50 terminal-recovery action and safety probe.

States are geometric perturbations of DEV, one-way expert terminal states.
They are never written to a training dataset or used to select eta.
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
from .collect_competence import teacher_reference
from .collect_n50_terminal_recovery import _candidate
from .environment import BottleneckEnv
from .observation import policy_observation_competence
from .scenario import Config


def run(dataset: Path, checkpoints: list[Path], output: Path, *, cases=8, seed=637920):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset/"manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    rows = [r for r in manifest["files"] if r["split"] == "dev"
            and r["source"] == "nominal" and "_full_LR_perm0" in r["rollout_id"]]
    rows = rows[:cases]
    if not rows:
        raise ValueError("no held-out full L->R expert trajectories")
    states = []
    for row in rows:
        with np.load(dataset/row["file"], allow_pickle=False) as data:
            history = np.asarray(data["states"],dtype=np.float64)
            initial = json.loads(str(data["initial_state_json"].item()))
        goals = np.asarray(initial["goals"],dtype=np.float64)
        for pending in (5,20):
            perturb_seed = int(np.random.SeedSequence([seed,int(initial["episode_seed"]),pending])
                               .generate_state(1)[0])
            selected, reason = _candidate(replace(config,seed=int(initial["episode_seed"]),split="dev"),
                                          history,goals,seed=perturb_seed,pending_count=pending)
            if selected is None:
                raise ValueError(f"held-out geometric state failed: {reason}")
            anchor, p, moved, attempt = selected
            for direction in ("LR","RL"):
                pp,gg=p.copy(),goals.copy()
                if direction=="RL":
                    pp[:,0]*=-1;gg[:,0]*=-1
                env=BottleneckEnv(replace(config,seed=int(initial["episode_seed"]),split="dev"))
                env.reset(pp,gg)
                d=np.linalg.norm(env.goals-env.positions,axis=1)
                pending_agents=np.flatnonzero(d>config.goal_tolerance)
                reference=teacher_reference(env)
                states.append(dict(parent=row["rollout_id"],anchor=anchor,pending_count=pending,
                                   direction=direction,perturbed_agents=moved,
                                   perturbation_attempt=attempt,env=env,
                                   pending_agents=pending_agents,reference=reference,
                                   observation=policy_observation_competence(env)))
    obs=np.asarray([x["observation"] for x in states],dtype=np.float32)
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    experts=[np.asarray(projector(x["env"].snapshot(),x["reference"]).velocity)
             for x in states]
    output_rows=[]
    for checkpoint in checkpoints:
        checkpoint=Path(checkpoint)
        agent,_=load_checkpoint(checkpoint,
            expected_environment_fingerprint=manifest["environment_fingerprint"])
        predictions=np.asarray(sample_bounded_actions(agent,obs,jax.random.PRNGKey(seed)),dtype=np.float64)
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for i,x in enumerate(states):
            pending=x["pending_agents"]
            env=x["env"]
            delta=env.goals[pending]-env.positions[pending]
            dist=np.linalg.norm(delta,axis=1)
            direction=delta/dist[:,None]
            flow=predictions[i]
            projection=projector(env.snapshot(),flow)
            safe=np.asarray(projection.velocity)
            expert=experts[i]
            def radial(actions):return np.sum(actions[pending]*direction,axis=1)
            output_rows.append(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,
                source_rollout=x["parent"],anchor=x["anchor"],direction=x["direction"],
                perturbed_count=x["pending_count"],pending_count=len(pending),
                goal_distance_quantiles=np.quantile(dist,[0,.5,1]).tolist(),
                expert_goalward_mean=float(np.mean(radial(expert))),
                flow_goalward_mean=float(np.mean(radial(flow))),
                safe_goalward_mean=float(np.mean(radial(safe))),
                flow_goalward_positive_fraction=float(np.mean(radial(flow)>0)),
                safe_goalward_positive_fraction=float(np.mean(radial(safe)>0)),
                projected_joint_correction=float(np.linalg.norm(safe-flow)),
                expert_joint_correction=float(np.linalg.norm(expert-x["reference"])),
                active_pair_count=int(np.sum(np.asarray(projection.diagnostics["active_linear_constraints"])
                    <int(projection.diagnostics["num_pair_constraints"]))),
                active_wall_count=int(np.sum(np.asarray(projection.diagnostics["active_linear_constraints"])
                    >=int(projection.diagnostics["num_pair_constraints"])))))
    grouped={}
    for checkpoint in checkpoints:
        for direction in ("LR","RL"):
            for pending in (5,20):
                subset=[r for r in output_rows if r["checkpoint"]==str(checkpoint)
                        and r["direction"]==direction and r["perturbed_count"]==pending]
                key=f"{Path(checkpoint).parent.name}/{Path(checkpoint).name}_{direction}_p{pending}"
                grouped[key]=dict(states=len(subset),
                    mean_pending=float(np.mean([r["pending_count"] for r in subset])),
                    expert_goalward=float(np.mean([r["expert_goalward_mean"] for r in subset])),
                    flow_goalward=float(np.mean([r["flow_goalward_mean"] for r in subset])),
                    safe_goalward=float(np.mean([r["safe_goalward_mean"] for r in subset])),
                    correction=float(np.mean([r["projected_joint_correction"] for r in subset])))
    report=dict(schema="gap1_n50_heldout_terminal_multi_action_audit_v1",
        dataset_manifest_sha256=hashlib.sha256((dataset/"manifest.json").read_bytes()).hexdigest(),
        seed=seed,dev_physical_seeds=cases,rows=output_rows,grouped=grouped,
        caveat="One-step DEV diagnostic; complete-horizon success is assessed separately")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",type=int,default=8)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.checkpoint,args.output,cases=args.cases)["grouped"],
                     indent=2),flush=True)


if __name__=="__main__":
    main()

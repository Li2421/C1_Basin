"""Replay frozen nominal actions on saved development states, without control changes."""
import argparse
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.frozen_scene_baseline import setup
from single_integrator.c1.train_deadlock_primary import noise,digest
from single_integrator.c1.differentiable_rollout import bounded_nominal
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.environment import GiveWayEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    _,field,plant,_,checkpoint=setup(args.frozen)
    protocol=json.loads((args.evaluation/'protocol.json').read_text())
    if protocol['baseline_sha256']!=digest(checkpoint):raise ValueError('Baseline changed')
    env=GiveWayEnv(plant)
    rows=[]
    for path in sorted(args.evaluation.glob('*.npz')):
        result=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            before,after,applied,safe=(z[k] for k in ('positions_before','positions_after','applied','safe'))
        n=len(before); count=min(n,round(2/plant.dt))
        draws=noise(result['noise_seed'],result['rid'])
        observations=[]
        for t in range(n-count,n):
            env.reset(before[t])
            env.velocities=np.zeros((2,2)) if t==0 else applied[t-1].reshape(2,2)
            observations.append(env.observation())
        nominal=np.asarray(bounded_nominal(field.baseline_sample(jnp.asarray(observations),draws[n-count:n]),plant.max_speed))
        actual=applied[-count:]
        row=dict(case=path.stem,distribution=result['distribution'],outcome=result['six_class_outcome'],
            trace_sha256=digest(path),initial=before[0].tolist(),final=after[-1].tolist(),
            final_goal_errors=np.linalg.norm(after[-1]-env.goals,axis=-1).tolist(),
            minimum_pair_separation=float(np.linalg.norm(after[:,0]-after[:,1],axis=-1).min()),
            last2s_nominal_mean_speed=np.linalg.norm(nominal.reshape(count,2,2),axis=-1).mean(axis=0).tolist(),
            last2s_actual_mean_speed=np.linalg.norm(actual.reshape(count,2,2),axis=-1).mean(axis=0).tolist(),
            last2s_mean_projection_change=float(np.linalg.norm(nominal-safe[-count:],axis=1).mean()),
            last2s_progress=(np.linalg.norm(after[-count]-env.goals,axis=-1)-np.linalg.norm(after[-1]-env.goals,axis=-1)).tolist(),
            min_wall_clearance=result['safety']['min_wall_surface_clearance'])
        rows.append(row)
    summary={}
    for distribution in ('matched','wide'):
        group=[r for r in rows if r['distribution']==distribution]
        summary[distribution]=dict(n=len(group),minimum_pair_separation=min(r['minimum_pair_separation'] for r in group),
            mean_nominal_speed=float(np.mean([r['last2s_nominal_mean_speed'] for r in group])),
            mean_actual_speed=float(np.mean([r['last2s_actual_mean_speed'] for r in group])),
            mean_projection_change=float(np.mean([r['last2s_mean_projection_change'] for r in group])))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(args.out,dict(scope='Already-seen development traces; descriptive replay, not causal intervention or held-out efficacy',
        baseline_sha256=digest(checkpoint),script_sha256=digest(Path(__file__)),summary=summary,rows=rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

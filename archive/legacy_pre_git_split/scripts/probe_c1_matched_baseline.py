"""Development-only matched versus wide-start closed-loop baseline check."""
import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.giveway_initial_state import sample_initial_positions
from single_integrator.c1.frozen_scene_baseline import setup
from single_integrator.c1.train_deadlock_primary import noise,digest
from single_integrator.c1.evaluate_deadlock_primary import execute
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace,OUTCOMES
from single_integrator.environment import GiveWayEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--scenes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    params,field,plant,cbf,checkpoint=setup(args.frozen)
    data=json.loads(args.scenes.read_text())
    scene=next(s for s in data['scenes'] if s['environment']==plant.to_dict())
    rng=np.random.default_rng(2026091676)
    pools=dict(matched=[dict(rid=80000+i,initial=sample_initial_positions(rng).tolist()) for i in range(4)],
               wide=scene['pools']['development'][:4])
    args.out.mkdir(parents=True)
    sources=[Path(__file__),ROOT/'single_integrator/c1/frozen_scene_baseline.py',
        ROOT/'single_integrator/c1/evaluate_deadlock_primary.py']
    atomic_save(args.out/'protocol.json',dict(scope='development only, no held-out test',
        baseline_sha256=digest(checkpoint),frozen_manifest_sha256=digest(args.frozen),
        scenes_sha256=digest(args.scenes),noise_seeds=data['development_noise_seeds'],pools=pools,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},backend=jax.default_backend()))
    rows=[]
    goals=GiveWayEnv(plant).goals
    for distribution,pool in pools.items():
        for item in pool:
            for draw in data['development_noise_seeds']:
                row,trace=execute(params,field,item['initial'],noise(draw,item['rid']),plant,cbf)
                label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(trace['applied'].reshape(-1,2,2),axis=-1).max(axis=1),
                    goal_errors=np.linalg.norm(trace['positions_after']-goals,axis=-1)),
                    'safe_deadlock' if row['any_deadlock'] else ('success' if row['success'] else 'other_timeout'),plant.dt)
                row.update(distribution=distribution,rid=item['rid'],noise_seed=draw,six_class_outcome=label)
                stem=f'{distribution}_{item["rid"]}_{draw}'
                atomic_save(args.out/(stem+'.json'),row)
                np.savez_compressed(args.out/(stem+'.npz'),**trace)
                rows.append(row)
                print(dict(distribution=distribution,rid=item['rid'],noise_seed=draw,outcome=label),flush=True)
    summary={name:dict(n=sum(r['distribution']==name for r in rows),
        counts={label:sum(r['distribution']==name and r['six_class_outcome']==label for r in rows) for label in OUTCOMES}) for name in pools}
    atomic_save(args.out/'summary.json',summary)
    atomic_save(args.out/'complete.json',dict(episodes=len(rows)))
    print(summary,flush=True)


if __name__=='__main__':main()

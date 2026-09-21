"""Inspect saved Safety traces for weak activity at the second projection.

No risk evaluation, controller execution, or solver modification.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFConfig, barrier_constraints


def main():
    folder=ROOT/'results/c1_difficulty_calibration_v1'
    out=ROOT/'results/c1_double_projection_boundary_v1.json'
    if out.exists(): raise FileExistsError(out)
    protocol=json.loads((folder/'protocol.json').read_text())
    env=GiveWayEnv(Config(**protocol['environment']))
    cbf=CBFConfig(**protocol['cbf'])
    rows=[]; hashes={}
    for distribution in ('original','closer_interaction'):
        for rid in range(25):
            path=folder/f'{distribution}_{rid}.npz'
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as tr:
                n=min(100,len(tr['applied']))
                counts={str(tol):0 for tol in (1e-8,1e-10,1e-12)}
                ranks=[]; max_delta=0.
                for t in range(n):
                    env.positions=tr['positions_before'][t].copy()
                    A,b,_=barrier_constraints(env.snapshot(),cbf)
                    p=tr['applied'][t];y=tr['candidate'][t]
                    max_delta=max(max_delta,float(np.max(np.abs(p-y))))
                    slack=np.r_[A@p-b,(.25-np.sum(p.reshape(2,2)**2,axis=1))/2]
                    for tol in (1e-8,1e-10,1e-12):
                        counts[str(tol)]+=int(np.any(slack<tol))
                    balls=np.zeros((2,4))
                    for i in range(2):balls[i,2*i:2*i+2]=-p[2*i:2*i+2]
                    active=np.vstack((A,balls))[slack<1e-8]
                    ranks.append(int(np.linalg.matrix_rank(active,tol=1e-10)) if len(active) else 0)
                rows.append(dict(distribution=distribution,rid=rid,steps=n,
                    boundary_steps=counts,max_second_projection_displacement=max_delta,
                    mean_active_normal_rank=float(np.mean(ranks)),
                    strict_deadlock=bool(tr['deadlock'].any()),success=bool(tr['success'].any())))
    aggregate={}
    for name in ('original','closer_interaction'):
        subset=[r for r in rows if r['distribution']==name]
        n=sum(r['steps'] for r in subset)
        aggregate[name]=dict(episodes=len(subset),early_steps=n,
            boundary_fraction={key:sum(r['boundary_steps'][key] for r in subset)/n for key in subset[0]['boundary_steps']},
            max_second_projection_displacement=max(r['max_second_projection_displacement'] for r in subset),
            mean_active_normal_rank=sum(r['mean_active_normal_rank']*r['steps'] for r in subset)/n)
    for path in ('scripts/audit_c1_double_projection_boundary.py','single_integrator/cbf.py',
                 'single_integrator/environment.py','single_integrator/c1/risk/joint_frozen.py'):
        hashes[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    result=dict(scope='cached first100-step boundary audit; counts are numerical, not exact multiplier certificates',
        aggregate=aggregate,rows=rows,source_sha256=hashes,new_robot_replays=0,
        theoretical_condition='If y=p exactly at a feasible boundary, zero multipliers satisfy projection KKT; classical derivative need not exist.',
        efficacy_established=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__': main()

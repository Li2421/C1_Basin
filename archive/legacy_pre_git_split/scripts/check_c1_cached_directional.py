"""Static projection differences at saved boundary states; no rollout or risk."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from single_integrator.c1.projection_directional import feasible_projection_direction
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity
from single_integrator.environment import Config,GiveWayEnv


def main():
    out=ROOT/'results/c1_cached_directional_v1.json'
    if out.exists():raise FileExistsError(out)
    folder=ROOT/'results/c1_difficulty_calibration_v1'
    protocol=json.loads((folder/'protocol.json').read_text())
    env=GiveWayEnv(Config(**protocol['environment']));cbf=CBFConfig(**protocol['cbf'])
    rng=np.random.default_rng(2026091873);rows=[];hashes={}
    for name in ('original','closer_interaction'):
        for rid in range(25):
            path=folder/f'{name}_{rid}.npz'
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as tr:
                for t in range(min(100,len(tr['applied']))):
                    env.positions=tr['positions_before'][t].copy()
                    A,b,_=barrier_constraints(env.snapshot(),cbf);y=tr['applied'][t]
                    slack=np.r_[A@y-b,(.25-np.sum(y.reshape(2,2)**2,axis=1))/2]
                    if slack.min()>1e-10:continue
                    # First numerical boundary state per trajectory, independent of risk/outcome.
                    base=project_velocity(y,A,b,.5,cbf)[0].reshape(-1)
                    for sign in (-1,1):
                        if sign==-1:d=rng.normal(size=4);d/=np.linalg.norm(d)
                        direction=sign*d
                        v,cert=feasible_projection_direction(y,A,b,direction)
                        measurements=[]
                        for h in (1e-3,1e-4,1e-5):
                            actual=project_velocity(y+h*direction,A,b,.5,cbf)[0].reshape(-1)
                            fd=(actual-base)/h
                            measurements.append(dict(h=h,fd=fd.tolist(),max_error=float(np.max(np.abs(fd-v)))))
                        rows.append(dict(distribution=name,rid=rid,t=t,sign=sign,
                            direction=direction.tolist(),derivative=v.tolist(),certificate=cert,measurements=measurements))
                    break
    for p in ('scripts/check_c1_cached_directional.py','single_integrator/c1/projection_directional.py',
              'single_integrator/cbf.py','single_integrator/environment.py'):
        hashes[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    worst=max(r['measurements'][-1]['max_error'] for r in rows)
    result=dict(scope='fixed-state static projection, no multi-step or risk-gradient validation',
        rng_seed=2026091873,directions=len(rows),boundary_states=len(rows)//2,
        finite_difference_tolerance=.005,smallest_step_max_error=worst,
        passed=worst<.005,rows=rows,source_sha256=hashes,new_robot_replays=0,
        original_solver_configuration=cbf.to_dict())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}))
    if not result['passed']:raise RuntimeError('static directional check failed; saved evidence retained')


if __name__=='__main__':main()

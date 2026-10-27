"""Fixed random snapshots, no controller execution and no rollout labels."""
from experiment import *
from unit_gates import project,minimum
from states import config,walls
from dynamics_spatial_mask_20261022 import curved_wall_minimum
from single_integrator.environment import segment_distance
import numpy as np

def main(scene,chunk):
    gamma=read(HERE/'UNIT_HEAD_ON.json')[scene].get('gamma')
    if scene!='lit_circle_five' and gamma is None:raise ValueError('Head-on gate has no admissible gamma')
    states=read(HERE/f'FLOW_STATES_{scene}.json');out=[];counts=dict(snapshots=0,original_feasible=0,restored=0,solver_error=0,feasible_collisions=0,permutation_failures=0)
    for k in range(chunk*1000,(chunk+1)*1000):
        rng=np.random.default_rng(np.random.SeedSequence([2026102301,('lit_narrow_passage','lit_crowded_random','lit_circle_five').index(scene),k]))
        s=states[int(rng.integers(len(states)))];c=config(scene,s['N']);p=np.asarray(s['positions'],float)
        theta=rng.uniform(-np.pi,np.pi,s['N']);speed=np.zeros(s['N']) if k<5000 else c.max_speed*rng.random(s['N'])
        v=speed[:,None]*np.column_stack([np.cos(theta),np.sin(theta)])
        theta2=rng.uniform(-np.pi,np.pi,s['N']);desired=c.max_speed*rng.random(s['N'])[:,None]*np.column_stack([np.cos(theta2),np.sin(theta2)])
        counts['snapshots']+=1
        try:a,diags=project(p,v,desired,c,gamma)
        except Exception as exc:
            counts['solver_error']+=1
            if len(out)<30:out.append(dict(index=k,error=str(exc),p=p,v=v,desired=desired))
            continue
        feasible=all(d['maximum_violation']<=1e-7 for d in diags)
        counts['original_feasible']+=feasible;counts['restored']+=not feasible
        pair=minimum(p,v,a,c)
        if len(walls(scene)):
            wall=min(curved_wall_minimum(p[i],v[i],a[i],c.dt,walls(scene),c.agent_radius+c.wall_collision_margin) for i in range(len(p)))
        else:wall=np.inf
        if feasible and min(pair,wall)<=0:
            counts['feasible_collisions']+=1
            if len(out)<30:out.append(dict(index=k,reason='feasible_collision',state_index=s['index'],p=p,v=v,desired=desired,a=a,pair=pair,wall=wall,diags=diags))
        # Every snapshot is checked under the same robot-index reversal.
        perm=np.arange(len(p))[::-1]
        try:
            ap,dp=project(p[perm],v[perm],desired[perm],c,gamma)
            if not np.allclose(a[perm],ap,atol=2e-6,rtol=0):
                counts['permutation_failures']+=1
                if len(out)<30:out.append(dict(index=k,reason='permutation',maximum_difference=float(abs(a[perm]-ap).max())))
        except Exception as exc:
            counts['permutation_failures']+=1
            if len(out)<30:out.append(dict(index=k,reason='permutation_solver',error=str(exc)))
        if k%100==0:print(scene,chunk,k,counts,flush=True)
    write(HERE/'unit_snapshots'/f'{scene}_{chunk}.json',dict(scene=scene,chunk=chunk,gamma=gamma,counts=counts,counterexamples=out,
          passed=not any(counts[k] for k in ('solver_error','feasible_collisions','permutation_failures'))))
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

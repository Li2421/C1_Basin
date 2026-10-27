"""No controller rollouts: frozen snapshot and head-on safety gates."""
from experiment import *
from stage2_safety import acceleration_problem,solve_soc,track_acceleration
from states import config,walls
from dynamics_spatial_mask_20261022 import quadratic_pair_clearance
from single_integrator.environment import point_segment_distance
import numpy as np

GRID=[.05,.1,.15,.2,.25,.3]

def project(p,v,desired,c,gamma=None,circle_discrete=False):
    bound=None if c.scene=='lit_circle_five' else 1.5
    outputs=[];diags=[]
    for i in range(len(p)):
        A,b,labels=acceleration_problem(p,v,i,c,walls(c.scene),gamma,circle_discrete)
        a,d=solve_soc(track_acceleration(desired[i],v[i],c.dt,bound),A,b,bound)
        outputs.append(a);diags.append(d)
    return np.asarray(outputs),diags

def minimum(p,v,a,c):
    radius=2*c.agent_radius+c.agent_collision_margin
    if c.scene=='lit_circle_five':
        return min(quadratic_pair_clearance(p[i]-p[j],v[i]-v[j],a[i]-a[j],c.dt,radius) for i in range(len(p)) for j in range(i+1,len(p)))
    nxt=p+c.dt*v
    return min(float(point_segment_distance(np.zeros(2),p[i]-p[j],nxt[i]-nxt[j]))-radius for i in range(len(p)) for j in range(i+1,len(p)))

def head_on(scene,gamma,circle_discrete=False):
    c=config(scene,2);worst=np.inf;bad=[]
    for gap in np.linspace(2,5,61):
        d=2*c.agent_radius+gap
        p=np.asarray([[-d/2,0],[d/2,0]]);v=np.array([[c.max_speed,0],[-c.max_speed,0.]])
        for t in range(300):
            try:a,diags=project(p,v,np.array([[c.max_speed,0],[-c.max_speed,0.]]),c,gamma,circle_discrete)
            except Exception as exc:bad.append(dict(gap=float(gap),step=t,error=str(exc)));break
            dist=minimum(p,v,a,c);worst=min(worst,dist)
            if any(x['safety_maximum_violation']>1e-7 for x in diags) or dist<=0:
                bad.append(dict(gap=float(gap),step=t,clearance=dist,diags=diags,p=p,v=v));break
            p=p+c.dt*v+(.5*c.dt*c.dt*a if scene=='lit_circle_five' else 0);v=v+c.dt*a
            if np.max(np.linalg.norm(v,axis=1))<1e-6:break
    return dict(gamma=gamma,passed=not bad,minimum_physical_clearance=worst,failures=bad,grid_count=61)

def main():
    results={}
    # This file and prereg are sealed before evaluation, not tuned to outcomes.
    if not (HERE/'UNIT_DESIGN.receipt.json').exists():seal('UNIT_DESIGN.receipt.json',[HERE/'unit_gates.py',HERE/'stage2_safety.py',HERE/'PREREGISTRATION.md'])
    for scene in ('lit_narrow_passage','lit_crowded_random'):
        trials=[]
        for gamma in GRID:
            r=head_on(scene,gamma);trials.append(r)
            print(scene,gamma,r['passed'],len(r['failures']),flush=True)
            write(HERE/'UNIT_HEAD_ON_PROGRESS.json',dict(results=results,scene=scene,trials=trials))
        passed=[r['gamma'] for r in trials if r['passed']]
        results[scene]=dict(head_on=trials,gamma=max(passed) if passed else None)
    results['lit_circle_five']=dict(head_on=head_on('lit_circle_five',None))
    write(HERE/'UNIT_HEAD_ON.json',results)
if __name__=='__main__':main()

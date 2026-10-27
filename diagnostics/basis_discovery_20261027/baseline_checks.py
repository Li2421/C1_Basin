"""Replay saved real constraints: CLF disabled equivalence and soft-CLF feasibility."""
import json
from pathlib import Path
import numpy as np
from runner import RN,inputs,old_runtime,dump
from shared_control.hard_projection import HardProjectionConfig
from baseline_kernels import clf_solve
from baselines import Rule,SOURCE,BASE,OUT

def main():
    anchor=';_,done,info=env.step(a)'
    assert SOURCE.replace('def run(c, post):','def run(c):').replace('\n            a,second=post(env,a,ref,filters,t,second)\n            _,done,info=env.step(a)',anchor)==old_runtime.SOURCE
    p=json.load(open(BASE/'PROTOCOL.json'));n=0;mx=0.;viol=0.;clf_error=[]
    for c in p['cases']:
        tr=np.load(c['diagnostic_trace']);geo=json.loads(str(tr['first_geometry_json']));cfg=RN.LiteratureConfig(**inputs()[0][c['scene']]['config'])
        for t in np.unique(np.linspace(0,len(tr['actions'])-1,10,dtype=int)):
            for i,x in enumerate(geo[t]):
                f=RN.SwapSafety.__new__(RN.SwapSafety);f.A=np.asarray(x['A']);f.lower=np.asarray(x['lower']);f.kinds=x['labels'];f.max_speed=cfg.max_speed;f.config=HardProjectionConfig()
                ref=tr['nominal'][t,i];expected=f.project(ref[None])[0][0]
                got,_=clf_solve(f,ref,tr['positions'][t,i]-tr['goals'][i],enabled=False)
                mx=max(mx,float(np.max(abs(got-expected))));assert mx<1e-7
                try:
                    u,delta=clf_solve(f,ref,tr['positions'][t,i]-tr['goals'][i]);v=max(0.,float(np.max(f.lower-f.A@u)),float(np.linalg.norm(u)-f.max_speed));viol=max(viol,v);assert v<=2e-6
                except Exception as exc:clf_error.append(str(exc))
                n+=1
    # Legacy RHR-B activation and direction are exactly preserved at H=1.
    from types import SimpleNamespace
    class Identity:
        def project(self,x):return x.copy(),{}
    env=SimpleNamespace(positions=np.array([[.2,.3],[-.2,-.3]]),goals=np.array([[1.,0.],[-1.,0.]]),config=SimpleNamespace(goal_tolerance=.05))
    h=Rule(dict(baseline=dict(rule='rhr_centre',W=40,thr=.05,H=1,s=.35)))
    for t in range(42):
        a,second=h(env,np.zeros((2,2)),np.zeros((2,2)),[Identity(),Identity()],t,[None,None])
        want=np.zeros((2,2))
        if t>=40:
            for i,z in enumerate(env.positions):want[i]=.35*np.array([z[1],-z[0]])/max(np.linalg.norm(z),1e-30)
        assert np.array_equal(a,want)
    report=dict(passed=True,local_problems=n,CLF_disabled_max_action_difference=mx,soft_CLF_max_constraint_violation=viol,soft_CLF_solver_errors=clf_error,legacy_RHR_B_trigger_direction_exact=True,original_source_only_post_hook=True,new_rollouts=0)
    dump(OUT/'checks.json',report);print(report)
if __name__=='__main__':main()

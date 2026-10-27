"""Regression on real production trajectories, plus B5 semantic invariants."""
import json
from types import SimpleNamespace
import numpy as np
from runtime import OUT, ROOT, RN, dump, run_hook, SOURCE, ADAPTED
from shared_control.correction_b5 import *


def main():
    tested=0;projected=0;max_projection_error=0.
    for row in map(json.loads,(OUT/'candidates.jsonl').open()):
        if row['spec']['name']!='smoke_b3':continue
        tr=np.load(row['trace']);g=tr['goals'];geometries=json.loads(str(tr['first_geometry_json']))
        cases=json.load(open(OUT/'smoke_planned_rollouts.json'))['exact_configs']
        c=next(c for c in cases if c['rollout_uid']==row['rollout_uid'])
        cfg=RN.LiteratureConfig(**c['scenario_payload']['config'])
        for t in range(len(tr['actions'])):
            p=tr['positions'][t];v=tr['safe'][t]
            fields=np.stack([RN.distributed_fields(p,g,v,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg)) for i in range(len(p))])
            expected=v.copy()
            for i in range(len(p)):expected[i]+=sum(float(x)*f for x,f in zip(c['eta_a'],fields[i]))
            fs=[SimpleNamespace(A=np.asarray(f['A']),lower=np.asarray(f['lower']),kinds=f['labels'])for f in geometries[t]]
            for recipe in ('goal','cluster','wall'):
                tang=tangent(recipe,p,fields[:,0],cfg.max_speed,RN.contact_radius(cfg),fs)
                role=FrozenYield(YieldConfig(0,1,'left',1.),p,g,2*cfg.agent_radius)
                for mode in ('add','blend'):
                    actual=target(v,fields,tang,role,p,cfg.max_speed,c['eta_a']+[0.,0.],mode)
                    assert np.array_equal(actual,expected),(recipe,mode,t)
                    tested+=1
            if t%20==0:
                for i,f in enumerate(fs):
                    from shared_control.hard_projection import project_velocity,HardProjectionConfig
                    a,_,_=project_velocity(expected[i:i+1],f.A,f.lower,cfg.max_speed,HardProjectionConfig())
                    err=float(np.max(np.abs(a[0]-tr['actions'][t,i])))
                    max_projection_error=max(max_projection_error,err);assert err<1e-10
                    projected+=1
            # HOLD as a velocity target: ADD leaves its G/P/R/T term; BLEND at 1 cancels it.
            role=FrozenYield(YieldConfig(0,1),p,g,2*cfg.agent_radius)
            eta=[.875,0.,.375,.5,1.]
            tfield=rotate(fields[:,0])
            add=target(v,fields,tfield,role,p,cfg.max_speed,eta,'add')
            blend=target(v,fields,tfield,role,p,cfg.max_speed,eta,'blend')
            assert np.array_equal(blend[0],np.zeros(2))
            np.testing.assert_array_equal(add[1:],blend[1:])
    assert len(yield_menu(0,1))==13
    q=np.array([[0.,0.],[.1,0.],[.2,0.]])
    rot=tangent('cluster',q,q,.5,.15,[])
    np.testing.assert_allclose(rot.sum(0),0,atol=1e-15)
    dump(OUT/'regression.json',dict(passed=True,bitwise_zero_extension_checks=tested,
         original_solver_action_checks=projected,max_projection_error=max_projection_error,
         scope='Every step of two actual smoke trajectories, three T recipes, ADD and BLEND; original GPR operation order.',
         reference_zip_checks='NOT_RUN_PACKAGE_MISSING'))
    print('passed',tested,projected,max_projection_error)


if __name__=='__main__':main()

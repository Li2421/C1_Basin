"""Replay real saved states, plus mechanism invariants; no new nominal rollout."""
import json
from types import SimpleNamespace
import numpy as np
from runner import OUT,RN,ADAPTED,old_runtime,dump
from discovery_design import freeze
from shared_control.basis_dictionary import evaluate,add_target,RoleState,NEW
from shared_control.correction_b5 import tangent
from shared_control.basis_families import OrthoFlow3BasisFamily,ORTHOFLOW3_SCALE
from shared_control.hard_projection import HardProjectionConfig


def main():
    assert ADAPTED.replace('def run(c, hook):','def run(c):').replace('            target=hook(env,v,ref,filters,t,eta,target)\n','')==old_runtime.SOURCE
    checks=0;projections=0;maxerror=0.;times=[]
    for c in freeze()['cases']:
        tr=np.load(c['diagnostic_trace']);geo=json.loads(str(tr['first_geometry_json']));cfg=RN.LiteratureConfig(**__import__('runner').inputs()[0][c['scene']]['config'])
        for t in sorted(set([0,len(tr['actions'])//2,len(tr['actions'])-1])):
            p=tr['positions'][t];g=tr['goals'];a=tr['safe'][t];raw=tr['nominal'][t]
            fs=[]
            for x in geo[t]:
                f=RN.SwapSafety.__new__(RN.SwapSafety);f.A=np.asarray(x['A']);f.lower=np.asarray(x['lower']);f.kinds=x['labels'];f.max_speed=cfg.max_speed;f.config=HardProjectionConfig();fs.append(f)
            b=np.stack([RN.distributed_fields(p,g,a,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg))for i in range(len(p))])
            fields,m=evaluate(p,g,a,raw,b,fs,__import__('runner').inputs()[0][c['scene']]['walls'],cfg,RoleState())
            assert set(fields)==set(('G','P','R')+NEW)
            for n in NEW:assert np.isfinite(fields[n]).all()and np.linalg.norm(fields[n],axis=1).max()<=cfg.max_speed+1e-12
            eta=c['baseline_eta'];original=a.copy()
            for i in range(len(p)):original[i]+=sum(float(x)*v for x,v in zip(eta,b[i]))
            assert np.array_equal(add_target(a,b,fields,eta,{}),original);checks+=1
            assert np.array_equal(fields['T1'],tangent('goal',p,b[:,0],cfg.max_speed,RN.contact_radius(cfg),fs))
            assert np.array_equal(fields['T2'],tangent('cluster',p,b[:,0],cfg.max_speed,RN.contact_radius(cfg),fs))
            for i in range(len(p)):
                x=fs[i].project(original[i:i+1])[0]
                y=fs[i].project(add_target(a,b,fields,eta,{})[i:i+1])[0]
                maxerror=max(maxerror,float(np.max(abs(x-y))));assert np.array_equal(x,y);projections+=1
            times.append(m['seconds'])
    # Strict near-active constraint: T5 respects its chosen tangent nullspace.
    cfg=SimpleNamespace(max_speed=.5,agent_radius=.05,agent_collision_margin=0.,goal_tolerance=.08,wall_radius=0.,wall_collision_margin=.005)
    p=np.array([[0.,0.],[.1,0.]]);g=np.array([[1.,0.],[-1.,0.]]);a=np.zeros_like(p);raw=np.array([[0.,.4],[0.,-.4]])
    fs=[SimpleNamespace(A=np.array([[-1.,0.]]),lower=np.array([0.]),kinds=[['pair',0,1]]),
        SimpleNamespace(A=np.array([[1.,0.]]),lower=np.array([0.]),kinds=[['pair',1,0]])]
    b=np.stack([RN.distributed_fields(p,g,a,i,.5,.08,.125125)for i in range(2)])
    fields,m=evaluate(p,g,a,raw,b,fs,[],cfg,RoleState())
    assert np.linalg.norm(fields['P'])==0 and np.linalg.norm(fields['P2'])>.1
    assert np.array_equal(fields['T3'][0],-fields['T3'][1]);assert m['role']['pair']==(0,1)
    assert np.max(abs(fields['T5'][:,0]))<1e-12
    assert np.linalg.norm(fields['C1'])==0
    # Cancellation remains soft: etaY4=1 cancels safe only, not original GPR.
    assert np.linalg.norm(add_target(a,b,fields,[1.,0.,0.],{'Y4':1.}))>0
    left,_=evaluate(p,g,a,raw,b,fs,[],cfg,RoleState(),names=['Y3'],clearance_side=1)
    right,_=evaluate(p,g,a,raw,b,fs,[],cfg,RoleState(),names=['Y3'],clearance_side=-1)
    assert np.array_equal(left['Y3'],-right['Y3']) and np.linalg.norm(left['Y3'])>0
    walls=np.array([[[-1.,-.2],[1.,-.2]],[[-1.,.2],[1.,.2]]])
    corridor=[SimpleNamespace(A=np.array([[0.,1.],[0.,-1.]]),lower=np.array([-.1449,-.1449]),kinds=[['wall',i,0],['wall',i,1]])for i in range(2)]
    cc,_=evaluate(p,g,a,raw,b,corridor,walls,cfg,RoleState(),names=['C1'])
    assert np.allclose(cc['C1'][:,1],0)and np.allclose(np.linalg.norm(cc['C1'],axis=1),.5)
    bzero=b.copy();bzero[1,0]=0.;gzero=g.copy();gzero[1]=p[1]
    yy,meta=evaluate(p,gzero,a,raw,bzero,fs,[],cfg,RoleState(),names=['Y1'])
    assert np.array_equal(yy['Y1'],np.zeros_like(p))and meta['degenerate']['Y1_zero_yielder_goal']
    result=dict(passed=True,real_state_zero_extension_checks=checks,original_projection_equal_checks=projections,max_projection_error=maxerror,
        T1_T2_prior_alias_exact=True,P_zero_P2_nonzero=True,T5_nullspace_and_pair_antisymmetry=True,C1_credible_corridor=True,Y3_both_sides=True,Y1_zero_goal_no_fallback=True,
        mean_all15_evaluation_seconds=float(np.mean(times)),no_rollouts_generated=True)
    dump(OUT/'checks.json',result);print(result)

if __name__=='__main__':main()

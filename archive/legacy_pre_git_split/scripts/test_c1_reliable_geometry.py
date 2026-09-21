"""Regression tests for the numerical-only geometry update, before unseen evaluation."""
import json
import numpy as np
from audit_c1_joint_witness_risk import ROOT,LocalRisk,Projection,SOLVER_AUDIT,save
from audit_c1_completed_waiting_risk import MAXR
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig,barrier_constraints

p=ROOT/'results/c1_geometry_solver_safety_audit'
cases=json.loads((p/'original_failure_details.json').read_text());refs=json.loads((p/'equivalent_solver_cases.json').read_text())
diff=[]
for x,y in zip(cases,refs):
    actual=LocalRisk(np.array(x['A']),np.array(x['b']),np.full(2,.5)).score(x['v'])
    diff.append(abs(actual['risk']-y['recovered']['risk']))
assert max(diff)<1e-10
fallback_errors=[]
for speeds in [None,np.full(2,.5)]:
    model=Projection(np.eye(4),np.zeros(4),speeds)
    for y in [np.array([-1.,2.,-3.,4.]),np.full(4,.1),np.full(4,-1.)]:
        expected=np.maximum(y,0)
        if speeds is not None:
            expected=expected.reshape(2,2);expected*=np.minimum(1,.5/np.maximum(np.linalg.norm(expected,axis=1),1e-300))[:,None];expected=expected.reshape(4)
        actual=model._fallback(y,'forced regression test')
        fallback_errors.append(float(np.linalg.norm(actual-expected)));assert fallback_errors[-1]<1e-7
env=GiveWayEnv(Config(corridor_half_length=1.3));cfg=CBFConfig();errors=[]
for f in (p/'recovered').glob('*.npz'):
    z=np.load(ROOT/'results/c1_candidate_coverage_expansion/traces'/f.name);expected=np.load(f)['g']
    for j,t in enumerate(range(100,300)):
        A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cfg)
        actual=LocalRisk(A,b,np.full(2,.5)).score(z['candidate'][t])['risk']/MAXR
        errors.append(abs(actual-expected[j]))
assert len(errors)==2000 and max(errors)<1e-9
result=dict(previous_failed_frames=len(cases),max_previous_failed_risk_difference=max(diff),
    forced_fallback_cases=len(fallback_errors),max_analytic_fallback_error=max(fallback_errors),
    complete_branch_frames=len(errors),max_audited_g_difference=max(errors),solver_counters=SOLVER_AUDIT)
save(p/'numerical_update_validation.json',result);print(json.dumps(result,indent=2))

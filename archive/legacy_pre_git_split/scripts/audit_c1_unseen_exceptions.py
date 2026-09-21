"""Observe and reproduce exceptions without changing frozen outcomes or solvers."""
import json
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from evaluate_c1_frozen_unseen import OUT,ROOT,manifest,PLAN_SEED
from audit_c1_joint_witness_risk import Projection,LocalRisk
from audit_c1_candidate_coverage import START,SHORT,delta_at
from audit_c1_completed_waiting_risk import MAXR
from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity,CBFSolverError
from single_integrator.evaluate import load_policy

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());assert manifest()==protocol['hashes']
    rows=[json.loads(f.read_text()) for f in OUT.glob('traces/*/*.json')]
    cfg=CBFConfig();calls=[];context={};original=Projection._fallback
    def observe(self,y,reason):
        p=original(self,y,reason)
        calls.append(dict(**context,reason=reason,base_qp=self.speeds is None,
            max_violation=self.max_violation,max_stationarity=self.max_stationarity))
        return p
    Projection._fallback=observe
    difference=0.
    for r in rows:
        if not r['numerical_counts']['fallback_attempts']:continue
        z=np.load(OUT/'traces'/str(r['rid'])/(r['cid']+'.npz'))
        env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False))
        for t in range(START,min(len(z['success']),SHORT)):
            context.update(rid=r['rid'],cid=r['cid'],step=t)
            A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cfg)
            g=LocalRisk(A,b,np.full(2,.5)).score(z['candidate'][t])['risk']/MAXR
            difference=max(difference,abs(g-z['g'][t-START]))
    Projection._fallback=original
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');errors=[]
    for r in rows:
        if not r['controller_error']:continue
        z=np.load(OUT/'traces'/str(r['rid'])/(r['cid']+'.npz'))
        env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False));env.reset(z['positions_before'][0])
        for u in z['applied']:env.step(u.reshape(2,2))
        np.testing.assert_array_equal(env.positions,z['positions_after'][-1])
        t=len(z['success']);key=jax.random.fold_in(jax.random.PRNGKey(PLAN_SEED),r['rid'])
        raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
        A,b,_=barrier_constraints(env.snapshot(),cfg);stage='baseline_projection'
        try:
            safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
            c=next(c for c in protocol['library'] if c['cid']==r['cid']);stage='intervention_projection'
            project_velocity(safe+delta_at(c,t),A,b,.5,cfg)
        except CBFSolverError as error:
            assert str(error)==r['controller_error']
            errors.append(dict(rid=r['rid'],cid=r['cid'],step=t,time=t*.05,stage=stage,error=str(error),
                zero_control_constraint_margin=float(np.min(-b)),zero_is_strictly_feasible=bool(np.all(b<0))))
        else:raise AssertionError('Controller exception not reproduced')
    assert len(calls)==5 and len(errors)==3 and difference<1e-12
    result=dict(geometry_fallbacks=calls,max_replayed_g_difference=difference,controller_exceptions=errors,
        frozen_results_unchanged=True)
    (OUT/'exception_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

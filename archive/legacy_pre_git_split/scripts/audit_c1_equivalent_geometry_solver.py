"""Equivalent-QP numerical audit only; original risk/controller files stay frozen."""
import json
import time
import numpy as np
import clarabel
from scipy import sparse
from scipy.optimize import minimize,nnls
import audit_c1_joint_witness_risk as risk
from audit_c1_joint_witness_risk import ROOT,save
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig,barrier_constraints

OUT=ROOT/'results/c1_geometry_solver_safety_audit'
BASE=ROOT/'results/c1_candidate_coverage_expansion'
Original=risk.Projection

def qp(A,b,y,update=False,options=None):
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=risk.PROTOCOL['solver_tol'];settings.max_iter=200
    for k,v in (options or {}).items():setattr(settings,k,v)
    solver=clarabel.DefaultSolver(sparse.eye(len(y),format='csc'),np.zeros_like(y) if update else -y,
        sparse.csc_matrix(-A),-b,[clarabel.NonnegativeConeT(len(b))],settings)
    if update:solver.update(q=-y)
    start=time.perf_counter();sol=solver.solve();elapsed=time.perf_counter()-start
    p=np.asarray(sol.x);lam=np.asarray(sol.z);slack=A@p-b;stationarity=p-y-A.T@lam
    return p,dict(status=str(sol.status),iterations=sol.iterations,gap=solver.get_info().gap_abs,
        violation=float(max(0,-slack.min())),stationarity=float(np.max(np.abs(stationarity))),
        complementarity=float(np.max(np.abs(lam*slack))),objective=float(.5*np.sum((p-y)**2)),
        seconds=elapsed,solution=p.tolist())

class StableBase(Original):
    def __call__(self,y):
        if self.speeds is not None:return super().__call__(y)
        p,info=qp(self.A,self.b,np.asarray(y),True,{'equilibrate_enable':False});self.calls+=1
        if info['status'] not in ('Solved','AlmostSolved'):raise RuntimeError(str(info))
        if info['violation']>2e-8 or info['stationarity']>2e-7:raise RuntimeError('uncertified equivalent QP')
        return p

def stable_score(A,b,v):
    risk.Projection=StableBase
    try:return risk.LocalRisk(A,b,np.full(2,.5)).score(v)
    finally:risk.Projection=Original

def reference(A,b,y):
    r=minimize(lambda p:.5*np.sum((p-y)**2),np.zeros_like(y),jac=lambda p:p-y,method='SLSQP',
        constraints={'type':'ineq','fun':lambda p:A@p-b,'jac':lambda p:A},options={'ftol':1e-12,'maxiter':200})
    p=r.x;slack=A@p-b;active=slack<=1e-7;lam,_=nnls(A[active].T,p-y,maxiter=10000)
    st=p-y-A[active].T@lam;comp=lam*slack[active]
    singular=np.linalg.svd(A[active],compute_uv=False)
    return p,dict(success=bool(r.success),violation=float(max(0,-slack.min())),
        stationarity=float(np.max(np.abs(st))),complementarity=float(np.max(np.abs(comp),initial=0)),
        objective=float(r.fun),active_rows=np.flatnonzero(active).tolist(),active_count=int(active.sum()),
        active_rank=int(np.linalg.matrix_rank(A[active])),active_singular_values=singular.tolist())

def main():
    cases=json.loads((OUT/'original_failure_details.json').read_text());results=[]
    variants={'zero_then_update':(True,{}),'cold_exact_target':(False,{}),
        'update_no_equilibration':(True,{'equilibrate_enable':False}),
        'update_no_presolve':(True,{'presolve_enable':False}),
        'update_2000_iterations':(True,{'max_iter':2000})}
    for r in cases:
        A=np.asarray(r['A']);b=np.asarray(r['b'])/.5;v=np.asarray(r['v']);q=v/np.linalg.norm(v)
        runs={name:qp(A,b,q,update,opt)[1] for name,(update,opt) in variants.items()}
        p,ref=reference(A,b,q);stable=np.asarray(runs['update_no_equilibration']['solution'])
        score=stable_score(A,np.asarray(r['b']),v)
        refB=float(np.sum((p-q)**2));assert abs(score['B']-refB)<1e-8
        norms=np.linalg.norm(A,axis=1);normalized=A/norms[:,None]
        cosine=normalized@normalized.T;np.fill_diagonal(cosine,0)
        results.append(dict(cid=r['cid'],step=r['step'],runs=runs,reference=ref,recovered=score,
            stable_reference_distance=float(np.linalg.norm(stable-p)),B_difference=float(score['B']-refB),
            base_origin_ball=float(np.min(-b/norms)),physical_origin_ball=float(min(.5,np.min(-np.asarray(r['b'])/norms))),
            rank=int(np.linalg.matrix_rank(A)),row_norm_range=[float(norms.min()),float(norms.max())],
            closest_opposite_cosine=float(cosine.min()),duplicate_normal_pairs=int(np.sum(np.triu(cosine>1-1e-12,1))),
            independent_g=(refB+risk.PROTOCOL['eta']*score['E'])/(1+.25*.05*np.logaddexp(0,24))))
    save(OUT/'equivalent_solver_cases.json',results)
    # Same A,b, multiple q values: geometry held exactly constant.
    controls=[]
    unique={np.asarray(r['A']).tobytes()+np.asarray(r['b']).tobytes():r for r in cases}
    for r in unique.values():
        A=np.asarray(r['A']);b=np.asarray(r['b'])/.5
        for i in range(4):
            for sign in [-1,1]:
                y=sign*np.eye(4)[i];_,info=qp(A,b,y,True)
                controls.append(dict(source_cid=r['cid'],step=r['step'],axis=i,sign=sign,**info))
    save(OUT/'same_geometry_control_qps.json',controls)
    summary=dict(failed_frame_instances=len(results),unique_failed_inputs=len({(np.asarray(r['A']).tobytes(),np.asarray(r['b']).tobytes(),tuple(r['v'])) for r in cases}),
        variants={name:dict(solved=sum(r['runs'][name]['status'] in ['Solved','AlmostSolved'] for r in results),
            max_iterations=max(r['runs'][name]['iterations'] for r in results),max_gap=max(r['runs'][name]['gap'] for r in results)) for name in variants},
        max_stable_reference_distance=max(r['stable_reference_distance'] for r in results),max_B_difference=max(abs(r['B_difference']) for r in results),
        reference_max_stationarity=max(r['reference']['stationarity'] for r in results),reference_max_complementarity=max(r['reference']['complementarity'] for r in results),
        base_origin_ball_range=[min(r['base_origin_ball'] for r in results),max(r['base_origin_ball'] for r in results)],
        physical_origin_ball_range=[min(r['physical_origin_ball'] for r in results),max(r['physical_origin_ball'] for r in results)],
        same_geometry_control_count=len(controls),same_geometry_control_failures=sum(r['status'] not in ['Solved','AlmostSolved'] for r in controls))
    save(OUT/'equivalent_solver_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

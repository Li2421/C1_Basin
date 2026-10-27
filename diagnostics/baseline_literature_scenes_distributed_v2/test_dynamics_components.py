"""Numerical and physical checks before any optional dynamics rollout."""
from experiment import *
import numpy as np
from dynamics_safety import *
from dynamics_runtime import quadratic_pair_clearance,quadratic_wall_clearance,curved_wall_minimum
from states import config,walls

def main():
    prereg();rng=np.random.default_rng(2026102008);tests={}
    x,diag=solve_soc([2.,3.],[[1.,0.]],[1.],1.5)
    assert np.allclose(x,[1.,np.sqrt(1.25)],atol=2e-5) and diag['maximum_violation']<1e-7
    x,diag=solve_soc([.5,0.],[[1.,0.],[-1.,0.]],[1.,1.],1.5)
    assert diag['restored'] and abs(diag['minimax_relaxation']-1.)<1e-6 and np.linalg.norm(x)<2e-5
    tests['feasible_projection_and_lexicographic_minimax']=True
    c=config('lit_crowded_random',3);p=np.asarray([[0.,0.],[.5,.3],[1.,-.5]]);v=np.asarray([[.2,.1],[-.1,.3],[0.,0.]])
    A,l,k=acceleration_problem(p,v,0,c,walls(c.scene));B,m,h=acceleration_problem(p,v,1,c,walls(c.scene))
    assert np.array_equal(A[0],-B[0]) and l[0]==m[0]
    tests['mutual_hocbf_pair_algebra']=True
    # A curve can touch the obstacle while both endpoints remain separated.
    pair=quadratic_pair_clearance(np.asarray([1.,0.]),np.asarray([-4.,0.]),np.asarray([8.,0.]),1.,.3)
    wall=quadratic_wall_clearance(np.asarray([0.,1.]),np.asarray([0.,-4.]),np.asarray([0.,8.]),1.,np.asarray([[-2.,0.],[2.,0.]]),.15)
    assert abs(pair+.3)<1e-8 and abs(wall+.15)<1e-8
    tests['continuous_curve_collision_between_safe_endpoints']=True
    segments=walls('lit_circle_five');errors=[]
    for _ in range(12):
        p=rng.uniform(-6,6,2);v=rng.normal(size=2);a=rng.normal(size=2)*4;dt=.05
        full=min(quadratic_wall_clearance(p,v,a,dt,s,2.005) for s in segments)
        selected=curved_wall_minimum(p,v,a,dt,segments,2.005)
        errors.append(abs(full-selected))
    assert max(errors)<1e-10
    tests['finite_wall_exact_minimum_pruning_max_error']=max(errors)
    c=config('lit_doorway');state=read(HERE/'FLOW_STATES_lit_doorway.json')[0]
    p=np.asarray(state['positions']);heading=np.asarray([.17,-.23]);u=np.asarray([.12,.8])
    values,jac,labels=discrete_problem(p,heading,0,c,walls(c.scene));eps=1e-6
    numerical=np.column_stack([(values(u+eps*e)-values(u-eps*e))/(2*eps) for e in np.eye(2)])
    error=float(np.max(abs(numerical-jac(u))));assert error<2e-7
    tests['discrete_unicycle_analytic_jacobian_max_error']=error
    control,diag=project_unicycle(np.asarray([.2,.05]),p,heading,0,c,walls(c.scene))
    assert abs(control[0])<=.26+1e-9 and abs(control[1])<=3.8+1e-9 and diag['maximum_violation']<=1e-7
    tests['unicycle_bounds_and_original_discrete_cbf_certificate']=True
    write(HERE/'DYNAMICS_COMPONENT_TESTS.json',dict(utc=utc(),passed=True,tests=tests,new_rollouts=0))
    seal('DYNAMICS_IMPLEMENTATION.receipt.json',[HERE/n for n in ('DYNAMICS_IMPLEMENTATION.md','dynamics_safety.py','dynamics_runtime.py','test_dynamics_components.py','DYNAMICS_COMPONENT_TESTS.json')],before_optional_dynamics_rollouts=True,first_stage_must_finish_before_submission=True)
    print('All optional dynamics component tests passed',flush=True)

if __name__=='__main__':main()

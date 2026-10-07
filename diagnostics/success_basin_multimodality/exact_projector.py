"""Same frozen projection problem with a tighter numerical retry only.

The fallback changes no objective, constraint, feasible set, or accepted
tolerance.  It is invoked only when the production Clarabel call fails its
external certificate despite returning a candidate.
"""
import numpy as np
import clarabel
from scipy import sparse
from scipy.optimize import minimize, nnls
from single_integrator.cbf import project_velocity, CBFSolverError


def _certificate(candidate,target,A,b,max_speed,solution,config):
    flat=np.asarray(candidate,dtype=np.float64).reshape(4)
    linear=A@flat-b;speeds=np.linalg.norm(flat.reshape(2,2),axis=-1)
    matrix,dual,slack=solution['_matrix'],solution['dual'],solution['slack']
    stationarity=float(np.linalg.norm(flat-target+matrix.T@dual,ord=np.inf))
    complementarity=float(abs(np.dot(slack,dual)))
    ok=(np.isfinite(flat).all() and linear.min()>=-config.feasibility_tol and
        speeds.max()-max_speed<=config.speed_tol and
        max(stationarity,complementarity)<=config.optimality_tol)
    return ok,{'min_linear_residual':float(linear.min()),
      'max_speed_excess':float(speeds.max()-max_speed),'stationarity_residual':stationarity,
      'complementarity_residual':complementarity,'primal_residual':solution['primal'],
      'dual_residual':solution['dual_residual'],'conic_iterations':solution['iterations']}


def tighter_identical_socp(target,A,b,max_speed,config):
    target=np.asarray(target,dtype=np.float64).reshape(4);A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    blocks=[sparse.csc_matrix(-A)];rhs=[-b];cones=[clarabel.NonnegativeConeT(len(b))]
    for agent in range(2):
        block=np.zeros((3,4));block[1:,2*agent:2*agent+2]=-np.eye(2)
        blocks.append(sparse.csc_matrix(block));rhs.append(np.asarray([max_speed,0.,0.]));cones.append(clarabel.SecondOrderConeT(3))
    matrix=sparse.vstack(blocks,format='csc');right=np.concatenate(rhs)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=max(400,config.max_iterations)
    settings.tol_feas=1e-12;settings.tol_gap_abs=1e-13;settings.tol_gap_rel=1e-13
    raw=clarabel.DefaultSolver(sparse.csc_matrix(np.eye(4)),-target,matrix,right,cones,settings).solve()
    sol={'_matrix':matrix,'dual':np.asarray(raw.z),'slack':np.asarray(raw.s),'primal':float(raw.r_prim),
         'dual_residual':float(raw.r_dual),'iterations':int(raw.iterations)}
    candidate=np.asarray(raw.x);ok,diag=_certificate(candidate,target,A,b,max_speed,sol,config)
    diag.update(solver_status=str(raw.status),backend='clarabel_identical_socp_tighter_retry')
    if not ok:raise CBFSolverError('identical_socp_retry_failed',diag)
    return candidate.reshape(2,2), 'solved_identical_socp_tighter_retry',diag


def identical_nonlinear_polish(target,A,b,max_speed,config):
    """Independent solve of the same two Euclidean balls, with KKT audit."""
    target=np.asarray(target,dtype=np.float64).reshape(4);A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    # Clarabel's nearly feasible answer is only an initializer, never accepted directly.
    blocks=[sparse.csc_matrix(-A)];rhs=[-b];cones=[clarabel.NonnegativeConeT(len(b))]
    for agent in range(2):
        block=np.zeros((3,4));block[1:,2*agent:2*agent+2]=-np.eye(2)
        blocks.append(sparse.csc_matrix(block));rhs.append(np.asarray([max_speed,0.,0.]));cones.append(clarabel.SecondOrderConeT(3))
    matrix=sparse.vstack(blocks,format='csc');right=np.concatenate(rhs);settings=clarabel.DefaultSettings();settings.verbose=False
    near=clarabel.DefaultSolver(sparse.csc_matrix(np.eye(4)),-target,matrix,right,cones,settings).solve();start=np.asarray(near.x)
    def speed_residual(u):
        q=u.reshape(2,2);return max_speed*max_speed-np.einsum('ij,ij->i',q,q)
    def speed_jac(u):
        value=np.zeros((2,4));value[0,:2]=-2*u[:2];value[1,2:]=-2*u[2:];return value
    result=minimize(lambda u:.5*np.dot(u-target,u-target),start,jac=lambda u:u-target,method='SLSQP',
      constraints=[{'type':'ineq','fun':lambda u:A@u-b,'jac':lambda u:A},
                   {'type':'ineq','fun':speed_residual,'jac':speed_jac}],
      options={'ftol':1e-14,'maxiter':1000,'disp':False})
    u=np.asarray(result.x);linear=A@u-b;speedr=speed_residual(u);rows=[];residual=[]
    for row,value in zip(A,linear):
        if value<=1e-7:rows.append(row);residual.append(value)
    sj=speed_jac(u)
    for row,value in zip(sj,speedr):
        if value<=1e-7:rows.append(row);residual.append(value)
    multipliers,_=nnls(np.asarray(rows).T,u-target,maxiter=5000)
    stationarity=float(np.linalg.norm(u-target-np.asarray(rows).T@multipliers,ord=np.inf))
    complementarity=float(np.max(abs(multipliers*np.asarray(residual))))
    diag={'backend':'scipy_slsqp_identical_nonlinear_constraints','solver_success':bool(result.success),
      'message':str(result.message),'iterations':int(result.nit),'min_linear_residual':float(linear.min()),
      'max_speed_excess':float(np.linalg.norm(u.reshape(2,2),axis=-1).max()-max_speed),
      'stationarity_residual':stationarity,'complementarity_residual':complementarity}
    ok=(np.isfinite(u).all() and linear.min()>=-config.feasibility_tol and
        diag['max_speed_excess']<=config.speed_tol and max(stationarity,complementarity)<=config.optimality_tol)
    if not ok:raise CBFSolverError('identical_nonlinear_polish_failed',diag)
    return u.reshape(2,2),'solved_identical_nonlinear_polish',diag


def project_velocity_with_retry(nominal,A,b,max_speed,config,**kwargs):
    try:
        value,status=project_velocity(nominal,A,b,max_speed,config,**kwargs)
        return value,status,False,None
    except CBFSolverError as first:
        try:
            value,status,diag=tighter_identical_socp(nominal,A,b,max_speed,config)
        except CBFSolverError as second:
            value,status,diag=identical_nonlinear_polish(nominal,A,b,max_speed,config)
            diag['tighter_clarabel_error']=str(second)
        diag['primary_error']=str(first)
        return value,status,True,diag

"""Directional derivative at a feasible input of the physical projection.

Diagnostic primitive, not a linear JVP/VJP or a replacement controller.
For y in C, D Pi_C(y)[d] is projection onto the tangent cone, provided
the linearized active constraints describe that cone. Moving A,b are
supported under directional regularity of the constraint system.
Strictly exterior projection inputs require a different critical-cone QP.
"""
import numpy as np
from scipy.optimize import minimize, nnls, lsq_linear


def projection_direction(y, A, b, projected, direction, *, max_speed=.5,
                         dA=None, db=None, active_tol=1e-8,
                         multiplier_tol=1e-9, certificate_tol=2e-6):
    """Critical-cone directional QP, with checked base KKT and active LICQ.

    Positive-multiplier constraints become equalities; weak constraints stay
    inequalities. Ball multipliers contribute curvature. Numerical tolerances
    approximate activity; failure is explicit, never a fallback derivative.
    This is not a linear map of direction at weak activity.
    """
    y,A,b,p,d=map(lambda x:np.asarray(x,dtype=float),(y,A,b,projected,direction))
    if y.ndim!=1 or not len(y) or len(y)%2 or p.shape!=y.shape or d.shape!=y.shape or A.ndim!=2 or A.shape[1]!=len(y) or b.shape!=(len(A),):
        raise ValueError('invalid projection dimensions')
    if not all(np.isfinite(x).all() for x in (y,A,b,p,d)):
        raise ValueError('finite data required')
    if not all(np.isfinite(x) and x>0 for x in (max_speed,active_tol,multiplier_tol,certificate_tol)):
        raise ValueError('positive finite speed and tolerances required')
    dA=np.zeros_like(A) if dA is None else np.asarray(dA,dtype=float)
    db=np.zeros_like(b) if db is None else np.asarray(db,dtype=float)
    if dA.shape!=A.shape or db.shape!=b.shape or not np.isfinite(dA).all() or not np.isfinite(db).all():
        raise ValueError('invalid constraint directions')
    # Shared wall endpoints may produce the same constraint twice. Merge only
    # matching base AND directional data; distinct moving bounds must remain.
    # This changes the diagnostic representation, never the physical QP.
    keep=[]
    for i in range(len(A)):
        jet=np.r_[A[i],b[i],dA[i],db[i]]
        if not any(np.allclose(jet,np.r_[A[k],b[k],dA[k],db[k]],rtol=0.,atol=1e-12) for k in keep):
            keep.append(i)
    duplicate_rows=len(A)-len(keep)
    A,b,dA,db=A[keep],b[keep],dA[keep],db[keep]
    balls=np.zeros((len(p)//2,len(p)))
    for i in range(len(balls)):balls[i,2*i:2*i+2]=-p[2*i:2*i+2]
    slack=np.r_[A@p-b,(max_speed**2-np.sum(p.reshape(-1,2)**2,axis=1))/2]
    if slack.min() < -active_tol:raise ValueError('base projection infeasible')
    active=slack<=active_tol;J=np.vstack((A,balls))[active]
    if len(J) and np.linalg.matrix_rank(J,tol=1e-10)<len(J):
        raise ValueError('active LICQ not verified; redundant constraint case unsupported')
    lam=nnls(J.T,p-y,maxiter=10000)[0] if len(J) else np.empty(0)
    base_error=float(np.max(np.abs(p-y-J.T@lam)))
    base_complementarity=float(np.max(np.abs(lam*slack[active]))) if len(J) else 0.
    if max(base_error,base_complementarity)>certificate_tol:
        raise ValueError('base projection KKT certificate failed')
    full=np.zeros(len(slack));full[active]=lam
    H=np.eye(len(p))
    for i,weight in enumerate(full[len(b):]):H[2*i:2*i+2,2*i:2*i+2]+=weight*np.eye(2)
    linear=d+dA.T@full[:len(b)]
    rhs=np.r_[db-dA@p,np.zeros(len(balls))][active]
    strong=lam>multiplier_tol;weak=~strong
    constraints=[]
    for mask,kind in ((strong,'eq'),(weak,'ineq')):
        if mask.any():
            mat=J[mask];bound=rhs[mask]
            constraints.append(dict(type=kind,fun=lambda v,mat=mat,bound=bound:mat@v-bound,
                                    jac=lambda v,mat=mat:mat))
    if not len(J):v=np.linalg.solve(H,linear)
    else:
        result=minimize(lambda v:.5*v@H@v-linear@v,np.linalg.solve(H,linear),
            jac=lambda v:H@v-linear,method='SLSQP',constraints=constraints,
            options=dict(ftol=1e-13,maxiter=200))
        v=result.x
    residual=J@v-rhs
    tight=strong|(residual<=1e-8)
    multipliers=np.zeros(len(J))
    if tight.any():
        lower=np.where(strong[tight],-np.inf,0.)
        multipliers[tight]=lsq_linear(J[tight].T,H@v-linear,
            bounds=(lower,np.full(tight.sum(),np.inf)),tol=1e-13,max_iter=1000).x
    stationarity=float(np.max(np.abs(H@v-linear-J.T@multipliers)))
    feasibility=max(float(np.max(np.abs(residual[strong]))) if strong.any() else 0.,
                    float(max(0.,-residual[weak].min())) if weak.any() else 0.)
    complementarity=float(np.max(np.abs(multipliers[weak]*residual[weak]))) if weak.any() else 0.
    info=dict(active_constraints=len(J),strong_constraints=int(strong.sum()),
              merged_identical_linear_rows=duplicate_rows,
              base_stationarity=base_error,base_complementarity=base_complementarity,
              stationarity=stationarity,feasibility=feasibility,complementarity=complementarity)
    if not np.isfinite(v).all() or max(stationarity,feasibility,complementarity)>certificate_tol:
        raise RuntimeError(f'critical-cone QP certificate failed: {info}')
    return v,info


def feasible_projection_direction(y, A, b, direction, *, max_speed=.5,
                                  dA=None, db=None, active_tol=1e-10,
                                  certificate_tol=1e-8):
    y,A,b,d=map(lambda x:np.asarray(x,dtype=np.float64),(y,A,b,direction))
    if y.ndim!=1 or len(y)%2 or d.shape!=y.shape or A.ndim!=2 or A.shape[1]!=len(y) or b.shape!=(len(A),):
        raise ValueError('invalid projection dimensions')
    if not all(np.isfinite(v).all() for v in (y,A,b,d)):
        raise ValueError('finite data required')
    if not np.isfinite(max_speed) or max_speed<=0 or not 0<active_tol<=certificate_tol:
        raise ValueError('positive speed and ordered tolerances required')
    dA=np.zeros_like(A) if dA is None else np.asarray(dA,dtype=float)
    db=np.zeros_like(b) if db is None else np.asarray(db,dtype=float)
    if dA.shape!=A.shape or db.shape!=b.shape or not np.isfinite(dA).all() or not np.isfinite(db).all():
        raise ValueError('invalid constraint directions')
    linear=A@y-b
    ball=(max_speed**2-np.sum(y.reshape(-1,2)**2,axis=1))/2
    if np.min(np.r_[linear,ball]) < -active_tol:
        raise ValueError('requires feasible projection input; exterior case unsupported')
    ball_rows=np.zeros((len(ball),len(y)))
    for i in range(len(ball)):ball_rows[i,2*i:2*i+2]=-y[2*i:2*i+2]
    active=np.r_[linear,ball]<=active_tol
    J=np.vstack((A,ball_rows))[active]
    rhs=np.r_[db-dA@y,np.zeros(len(ball))][active]
    if not len(J):
        return d.copy(),dict(active_constraints=0,stationarity=0.,feasibility=0.,complementarity=0.)
    result=minimize(lambda v:.5*np.sum((v-d)**2),d,
        jac=lambda v:v-d,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda v:J@v-rhs,jac=lambda v:J)],
        options=dict(ftol=1e-13,maxiter=200))
    v=result.x;slack=J@v-rhs
    tight=slack<=certificate_tol
    lam=np.zeros(len(J))
    if tight.any():lam[tight]=nnls(J[tight].T,v-d,maxiter=10000)[0]
    info=dict(active_constraints=int(active.sum()),
        stationarity=float(np.max(np.abs(v-d-J.T@lam))),
        feasibility=float(max(0.,-slack.min())),
        complementarity=float(np.max(np.abs(lam*slack))))
    if not np.isfinite(v).all() or max(info[k] for k in ('stationarity','feasibility','complementarity'))>certificate_tol:
        raise RuntimeError(f'directional QP certificate failed: {info}; {result.message}')
    return v,info

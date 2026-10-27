"""Read-only reuse of audited baseline_comparison_v1 pure kernels; no legacy initializer."""

import numpy as np
import clarabel
from scipy import sparse

def groups(q, arrived, stuck):
    todo=set(np.flatnonzero(~arrived).tolist()); result=[]
    while todo:
        stack=[min(todo)]; todo.remove(stack[0]); comp=[]
        while stack:
            i=stack.pop(); comp.append(i)
            nb=[j for j in sorted(todo) if np.linalg.norm(q[i]-q[j])<=.5]
            todo.difference_update(nb); stack.extend(nb)
        if stuck[comp].any():result.append(sorted(comp))
    return result

def clf_solve(f, ref, d, lam=1., penalty=100., enabled=True):
    """Exact SOC speed bound; soft CLF. Returns raw solution or raises."""
    ref=np.asarray(ref).reshape(2); m=len(f.lower)
    if not enabled:
        # Same original objective scaling, constraints, tolerances, and feasible fast path.
        if np.min(f.A@ref-f.lower)>=0 and np.linalg.norm(ref)<=f.max_speed:return ref.copy(),0.
        mat=sparse.vstack([sparse.csc_matrix(-f.A),sparse.csc_matrix(np.vstack([np.zeros(2),-np.eye(2)]))],format='csc')
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=f.config.max_iterations
        settings.tol_feas=min(f.config.feasibility_tol,f.config.speed_tol)
        settings.tol_gap_abs=settings.tol_gap_rel=f.config.solver_ftol
        sol=clarabel.DefaultSolver(sparse.csc_matrix(np.eye(2)),-ref,mat,np.r_[-f.lower,f.max_speed,0.,0.],
            [clarabel.NonnegativeConeT(m),clarabel.SecondOrderConeT(3)],settings).solve()
        if str(sol.status) not in ('Solved','AlmostSolved'):raise RuntimeError(str(sol.status))
        return np.asarray(sol.x),0.
    A=np.vstack([np.c_[-f.A,np.zeros(m)],np.r_[2*d,-1.], [0.,0.,-1.]])
    b=np.r_[-f.lower,-lam*(d@d),0.]
    ball=np.zeros((3,3)); ball[1:,:2]=-np.eye(2)
    mat=sparse.vstack([sparse.csc_matrix(A),sparse.csc_matrix(ball)],format='csc')
    settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_iter=200
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
    sol=clarabel.DefaultSolver(sparse.diags([2.,2.,2*penalty],format='csc'),np.r_[-2*ref,0.],mat,np.r_[b,f.max_speed,0.,0.],
        [clarabel.NonnegativeConeT(m+2),clarabel.SecondOrderConeT(3)],settings).solve()
    if str(sol.status) not in ('Solved','AlmostSolved'):raise RuntimeError('CLF solve '+str(sol.status))
    x=np.asarray(sol.x)
    if not np.isfinite(x).all() or np.max(A@x-b,initial=0)>2e-6 or np.linalg.norm(x[:2])>f.max_speed+1e-8:
        raise RuntimeError('CLF certificate failed')
    return x[:2],float(x[2])

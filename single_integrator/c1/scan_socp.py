"""JIT-compatible adapter for the pinned CVXPYLayers 1.2 / DIFFCP backend.

The upstream DIFFCP JAX wrapper keeps adjoints in Python closures and cannot
run inside scan. This adapter keeps canonicalization/recovery unchanged and
moves only NumPy solving and implicit adjoints into pure callbacks. Backward
re-solves the same cone problem, avoiding a persistent per-step solver cache.
No finite differences, substitute safety filter, or surrogate derivative.
"""
import diffcp
import jax
import numpy as np
from cvxpylayers.jax.cvxpylayer import CvxpyLayer, _recover_results
from cvxpylayers.interfaces.diffcp_if import _build_diffcp_matrices, _compute_gradients
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict


class CallbackCvxpyLayer(CvxpyLayer):
    def _solve_with_custom_vjp(self,P_eval,q_eval,A_eval,batch,solver_args):
        if P_eval is not None:
            raise ValueError('callback adapter requires the existing linear cone canonicalization')
        ctx=self.ctx.solver_ctx
        B=1 if q_eval.ndim==1 else q_eval.shape[1]
        shapes=(jax.ShapeDtypeStruct((B,ctx.A_shape[1]-1),q_eval.dtype),
                jax.ShapeDtypeStruct((B,ctx.A_shape[0]),q_eval.dtype))
        cones=[dims_to_solver_dict(ctx.dims)]*B
        options=dict(ctx.options,**solver_args)

        def data(q,A):
            q,A=np.asarray(q),np.asarray(A)
            return _build_diffcp_matrices(A[:,None] if A.ndim==1 else A,
                    q[:,None] if q.ndim==1 else q,ctx.A_structure,ctx.A_shape,ctx.b_idx,B)

        def forward_host(q,A):
            As,bs,cs,_=data(q,A)
            xs,ys,_=diffcp.solve_only_batch(As,bs,cs,cones,n_jobs_forward=1,**options)
            return np.asarray(xs,dtype=q.dtype),np.asarray(ys,dtype=q.dtype)

        def backward_host(q,A,gx,gy):
            As,bs,cs,b_idxs=data(q,A)
            _,_,_,_,adj=diffcp.solve_and_derivative_batch(As,bs,cs,cones,
                         n_jobs_forward=1,n_jobs_backward=1,**options)
            dq,dA=_compute_gradients(adj,gx,gy,bs,b_idxs,B)
            dq,dA=np.asarray(dq,dtype=q.dtype).T,np.asarray(dA,dtype=A.dtype).T
            if q.ndim==1:dq=dq[:,0]
            if A.ndim==1:dA=dA[:,0]
            return dq,dA

        @jax.custom_vjp
        def solve(q,A):
            return jax.pure_callback(forward_host,shapes,q,A)

        def forward(q,A):
            return solve(q,A),(q,A)

        def backward(saved,g):
            q,A=saved
            spec=(jax.ShapeDtypeStruct(q.shape,q.dtype),jax.ShapeDtypeStruct(A.shape,A.dtype))
            return jax.pure_callback(backward_host,spec,q,A,*g)

        solve.defvjp(forward,backward)
        primal,dual=solve(q_eval,A_eval)
        return _recover_results(primal,dual,self.ctx,batch)

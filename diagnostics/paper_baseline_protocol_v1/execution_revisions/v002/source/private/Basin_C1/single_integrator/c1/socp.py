"""Exact mathematical SOCP projection, with diffcp implicit cone derivatives.

The runtime CBF implementation remains authoritative for geometry. Numerical
solutions are checked at its feasibility/speed tolerances; never clipped.
Candidate control and state-dependent geometry both retain derivatives.
Requires the optional Python 3.11 C1 environment; no imports in baseline paths.
"""
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cvxpylayers.jax import CvxpyLayer

from single_integrator.cbf import CBFConfig, CBFSolverError


class ExactProjection:
    def __init__(self, rows=17, scan_compatible=False):
        self.rows,self.scan_compatible=rows,scan_compatible
        u = cp.Variable(4)
        target = cp.Parameter(4)
        A = cp.Parameter((rows, 4))
        b = cp.Parameter(rows)
        speed = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(.5 * cp.sum_squares(u-target)),
                             [A @ u >= b, cp.norm(u[:2], 2) <= speed,
                              cp.norm(u[2:], 2) <= speed])
        assert problem.is_dpp()
        layer_class=CvxpyLayer
        if scan_compatible:
            from single_integrator.c1.scan_socp import CallbackCvxpyLayer
            layer_class=CallbackCvxpyLayer
        self.layer = layer_class(problem, parameters=[target, A, b, speed], variables=[u])

    def for_scan(self):
        if self.scan_compatible:return self
        if not hasattr(self,'_scanned'):
            self._scanned=ExactProjection(self.rows,scan_compatible=True)
        return self._scanned

    def __call__(self, target, A, b, speed):
        if not jax.config.x64_enabled:
            raise ValueError('C1 SOCP requires JAX_ENABLE_X64=1 for safety tolerances')
        target = jnp.asarray(target, dtype=jnp.float64)
        A, b, speed = (jnp.asarray(x, dtype=jnp.float64)
                       for x in (A, b, speed))
        u = self.layer(target, A, b, speed,
                       solver_args={'eps': 1e-11, 'max_iters': 100000})[0]
        jax.debug.callback(self.validate, u, A, b, speed)
        return u

    @staticmethod
    def validate(u, A, b, speed):
        u, A, b = map(np.asarray, (u, A, b))
        cbf = float(np.min(np.einsum('...ij,...j->...i', A, u)-b))
        excess = float(np.max(np.linalg.norm(u.reshape(-1, 2, 2), axis=-1)-speed))
        cfg = CBFConfig()
        if not np.isfinite(u).all() or cbf < -cfg.feasibility_tol or excess > cfg.speed_tol:
            raise CBFSolverError('socp_invalid', dict(min_cbf=cbf, max_speed_excess=excess))
        return dict(min_cbf=cbf, max_speed_excess=excess)

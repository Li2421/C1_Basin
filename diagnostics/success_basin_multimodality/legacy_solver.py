"""Exact frozen pre-Clarabel projection backend, retained only for failure replay.

The implementation below is the projection routine from git object
cc8579e:single_integrator/cbf.py (SHA256 f127d09c...).  It is never used to
generate SBMA outcomes; it only reproduces the earlier numerical exception.
"""
import numpy as np
from scipy.optimize import minimize, nnls
from single_integrator.cbf import CBFSolverError


def project_velocity_legacy(nominal, A, b, max_speed, config):
    target = np.asarray(nominal, dtype=np.float64).reshape(4)
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite QP input')
    if A.shape != (len(b), 4) or not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError('Invalid QP dimensions or speed limit')
    if (np.min(A @ target-b) >= 0 and
            np.linalg.norm(target.reshape(2, 2), axis=-1).max() <= max_speed):
        return target.reshape(2, 2).copy(), 'nominal_feasible'
    matrix = np.concatenate([A, np.eye(4), -np.eye(4)])
    lower = np.concatenate([b, np.full(8, -max_speed)])
    start = np.zeros(4)
    for cut_round in range(config.max_speed_cuts+1):
        result = minimize(
            lambda u: .5*np.dot(u-target, u-target), start,
            jac=lambda u: u-target, method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda u: matrix@u-lower,
                         'jac': lambda u: matrix},
            options={'ftol': config.solver_ftol, 'maxiter': config.max_iterations,
                     'disp': False})
        candidate = np.asarray(result.x)
        residual = matrix@candidate-lower
        details = dict(message=str(result.message), solver_success=bool(result.success),
                       min_constraint_residual=float(residual.min()), cut_round=cut_round)
        if not np.isfinite(candidate).all() or residual.min() < -config.feasibility_tol:
            raise CBFSolverError('qp_failed', details)
        active = residual <= 1e-7
        gradient = candidate-target
        if np.any(active):
            try:
                multipliers, _ = nnls(matrix[active].T, gradient, maxiter=1000)
            except RuntimeError as exc:
                raise CBFSolverError('kkt_failed', details) from exc
            stationarity = np.linalg.norm(gradient-matrix[active].T@multipliers, ord=np.inf)
            complementarity = float(np.max(np.abs(multipliers*residual[active])))
        else:
            stationarity, complementarity = np.linalg.norm(gradient, ord=np.inf), 0.
        if max(stationarity, complementarity) > config.optimality_tol:
            details.update(stationarity=float(stationarity), complementarity=complementarity)
            raise CBFSolverError('kkt_failed', details)
        speed = np.linalg.norm(candidate.reshape(2, 2), axis=-1)
        if speed.max() <= max_speed+config.speed_tol:
            return candidate.reshape(2, 2).copy(), ('solved' if result.success else 'solved_kkt_certified')
        for agent in np.flatnonzero(speed > max_speed+config.speed_tol):
            row = np.zeros(4)
            row[2*agent:2*agent+2] = -candidate.reshape(2, 2)[agent]/speed[agent]
            matrix = np.vstack([matrix, row])
            lower = np.r_[lower, -max_speed]
        start = candidate
    raise CBFSolverError('speed_cut_limit', {'max_speed_cuts': config.max_speed_cuts})

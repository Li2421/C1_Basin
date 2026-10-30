"""Hard, post-hoc CBF projection for the existing direct-SI plant.

Finite segments are the actual environment walls, not their infinite supporting
lines. All segment barriers are enforced: min_j h_ij >= 0 is their intersection,
including both/all active branches at ties and corners. No goal, mode, stage,
priority, slack, or recovery controller enters the optimization.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize, nnls

from single_integrator.filters import FilterResult


@dataclass(frozen=True)
class CBFConfig:
    gamma: float = 1.0
    gamma_wall: float = 1.0
    # Fixed numerical separation from the evaluator's inclusive collision tests.
    # This is 0.1 mm, in addition to the UNCHANGED reporting collision margins.
    separation_buffer: float = 1e-4
    solver_ftol: float = 1e-12
    feasibility_tol: float = 1e-9
    optimality_tol: float = 2e-6
    speed_tol: float = 1e-10
    intervention_tol: float = 1e-6
    max_iterations: int = 200
    max_speed_cuts: int = 32

    def __post_init__(self):
        for name in ['gamma', 'gamma_wall', 'separation_buffer', 'solver_ftol',
                     'feasibility_tol', 'optimality_tol', 'speed_tol', 'intervention_tol']:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ['max_iterations', 'max_speed_cuts']:
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')

    def to_dict(self):
        return asdict(self)


class CBFSolverError(RuntimeError):
    """Experiment error; never an episode outcome or permission to use a fallback."""
    def __init__(self, status, details):
        self.status, self.details = status, details
        super().__init__(f'{status}: {details}')


def barrier_geometry(snapshot, config):
    p = np.asarray(snapshot['positions'], dtype=np.float64)
    walls = np.asarray(snapshot['walls'], dtype=np.float64)
    plant = snapshot['config']
    if p.shape != (2, 2) or walls.ndim != 3 or walls.shape[1:] != (2, 2):
        raise ValueError('Expected two agents and finite wall segments')
    if not np.isfinite(p).all() or not np.isfinite(walls).all():
        raise ValueError('Nonfinite barrier geometry')
    a, b = walls[:, 0], walls[:, 1]
    segment = b-a
    length2 = np.sum(segment*segment, axis=-1)
    if np.any(length2 <= 0):
        raise ValueError('Degenerate wall segment')
    t = np.clip(np.sum((p[:, None]-a)*segment, axis=-1)/length2, 0., 1.)
    nearest = a+t[..., None]*segment
    delta = p[:, None]-nearest
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance <= 1e-14):
        raise CBFSolverError('undefined_wall_gradient', {'positions': p.tolist()})
    wall_h = (distance-plant['agent_radius']-plant['wall_radius']
              -plant['wall_collision_margin']-config.separation_buffer)
    wall_gradient = delta/distance[..., None]
    relative = p[0]-p[1]
    d_safe = 2*plant['agent_radius']+plant['agent_collision_margin']+config.separation_buffer
    pairwise_h = float(relative@relative-d_safe*d_safe)
    pair_gradient = np.concatenate([2*relative, -2*relative])
    return dict(pairwise_h=pairwise_h, wall_h=wall_h,
                min_wall_h=float(wall_h.min()), wall_gradient=wall_gradient,
                pair_gradient=pair_gradient, d_safe=d_safe)


def barrier_constraints(snapshot, config):
    """Return A,b for A @ flatten(u) >= b, and pre-action barrier values."""
    geometry = barrier_geometry(snapshot, config)
    rows = [geometry['pair_gradient']]
    lower = [-config.gamma*geometry['pairwise_h']]
    for agent in range(2):
        for gradient, h in zip(geometry['wall_gradient'][agent], geometry['wall_h'][agent]):
            row = np.zeros(4)
            row[2*agent:2*agent+2] = gradient
            rows.append(row)
            lower.append(-config.gamma_wall*h)
    return np.asarray(rows), np.asarray(lower), geometry


def project_velocity(nominal, A, b, max_speed, config):
    """Euclidean projection using convex QPs with actuator-ball cutting planes.

Each QP has the requested quadratic objective and linear hard CBF inequalities.
The existing per-agent Euclidean speed balls are represented by adaptive OUTER
supporting planes. There is no inscribed polygon, reduced speed cap, or clipping
after solving. A QP optimum that is also in the balls solves the full constrained
projection (to the declared tolerances). The objective's unconstrained target is
the same u_nom in every solve.
"""
    target = np.asarray(nominal, dtype=np.float64).reshape(4)
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite QP input')
    if A.shape != (len(b), 4) or not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError('Invalid QP dimensions or speed limit')
    if (np.min(A@target-b) >= 0 and
            np.linalg.norm(target.reshape(2, 2), axis=-1).max() <= max_speed):
        return target.reshape(2, 2).copy(), 'nominal_feasible'

    # The initial square is an outer bound of the actual speed balls.
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
        # Solver status is not an optimality certificate. SLSQP can report
        # incompatible inequalities for a boundary optimum at roundoff scale.
        # Accept only after the unchanged feasibility, nonnegative-dual KKT,
        # complementarity and actual speed-ball certificates below all pass.
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
        # Tangents are supporting halfspaces of the original circles, not a
        # heuristic velocity modification. Never apply a post-QP projection.
        for agent in np.flatnonzero(speed > max_speed+config.speed_tol):
            row = np.zeros(4)
            row[2*agent:2*agent+2] = -candidate.reshape(2, 2)[agent]/speed[agent]
            matrix = np.vstack([matrix, row])
            lower = np.r_[lower, -max_speed]
        start = candidate
    raise CBFSolverError('speed_cut_limit', {'max_speed_cuts': config.max_speed_cuts})


class CBFSafetyFilter:
    def __init__(self, config=None):
        self.config = config or CBFConfig()

    def __call__(self, snapshot, nominal_velocity):
        dt = snapshot['config']['dt']
        if max(self.config.gamma, self.config.gamma_wall)*dt > 1:
            raise ValueError('Require gamma*dt <= 1 for the held-velocity interval certificate')
        A, b, _ = barrier_constraints(snapshot, self.config)
        velocity, status = project_velocity(nominal_velocity, A, b,
                                            snapshot['config']['max_speed'], self.config)
        return FilterResult(velocity=velocity, status=status)


def cbf_factory(plant_config):
    """Existing evaluator filter seam, with fixed default CBF parameters."""
    return CBFSafetyFilter()

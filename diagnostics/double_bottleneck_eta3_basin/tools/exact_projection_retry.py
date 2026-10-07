"""Numerical retry for the identical frozen N-agent projection problem.

This changes no objective, constraint, feasible set, or accepted certificate.
It is invoked only after the canonical projector rejects a numerical candidate.
"""

from __future__ import annotations

import clarabel
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, nnls

from shared_control.hard_projection import (
    CBFSolverError,
    HardProjectionConfig,
    HardSafetyFilter,
    barrier_constraints,
)
from single_integrator.filters import FilterResult


def _certificate(candidate, target, matrix, lower, max_speed, config, stationarity, complementarity):
    flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    linear = matrix @ flat - lower
    speeds = np.linalg.norm(flat.reshape((-1, 2)), axis=-1)
    return bool(
        np.isfinite(flat).all()
        and linear.min() >= -config.feasibility_tol
        and speeds.max() - max_speed <= config.speed_tol
        and max(stationarity, complementarity) <= config.optimality_tol
    ), {
        "objective": float(0.5 * np.dot(flat - target, flat - target)),
        "min_linear_residual": float(linear.min()),
        "linear_residuals": linear.tolist(),
        "per_agent_speed": speeds.tolist(),
        "max_speed_excess": float(speeds.max() - max_speed),
        "stationarity_residual": float(stationarity),
        "complementarity_residual": float(complementarity),
        "max_constraint_violation": float(max(0.0, (-linear).max(), (speeds - max_speed).max())),
        "active_linear_constraints": np.flatnonzero(linear <= 1e-7).tolist(),
        "active_speed_constraints": np.flatnonzero(np.abs(speeds - max_speed) <= 1e-7).tolist(),
        "projection_input": target.tolist(),
        "solver_output_candidate": flat.tolist(),
    }


def tighter_identical_socp(nominal, matrix, lower, max_speed, config):
    nominal = np.asarray(nominal, dtype=np.float64)
    target = nominal.reshape(-1)
    num_agents = len(nominal)
    blocks = [sparse.csc_matrix(-matrix)]
    rhs = [-lower]
    cones = [clarabel.NonnegativeConeT(len(lower))]
    for agent in range(num_agents):
        block = np.zeros((3, 2 * num_agents), dtype=np.float64)
        block[1:, 2 * agent : 2 * agent + 2] = -np.eye(2)
        blocks.append(sparse.csc_matrix(block))
        rhs.append(np.asarray((max_speed, 0.0, 0.0)))
        cones.append(clarabel.SecondOrderConeT(3))
    cone_matrix = sparse.vstack(blocks, format="csc")
    cone_rhs = np.concatenate(rhs)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = max(400, config.max_iterations)
    settings.tol_feas = 1e-12
    settings.tol_gap_abs = 1e-13
    settings.tol_gap_rel = 1e-13
    solution = clarabel.DefaultSolver(
        sparse.csc_matrix(np.eye(2 * num_agents)),
        -target,
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    ).solve()
    candidate = np.asarray(solution.x, dtype=np.float64)
    dual = np.asarray(solution.z, dtype=np.float64)
    slack = np.asarray(solution.s, dtype=np.float64)
    stationarity = float(
        np.linalg.norm(candidate - target + cone_matrix.T @ dual, ord=np.inf)
    )
    complementarity = float(abs(np.dot(slack, dual)))
    accepted, diagnostics = _certificate(
        candidate,
        target,
        matrix,
        lower,
        max_speed,
        config,
        stationarity,
        complementarity,
    )
    diagnostics.update(
        backend="clarabel_identical_socp_tighter_retry",
        solver_status=str(solution.status),
        accepted=accepted,
        conic_iterations=int(solution.iterations),
        primal_residual=float(solution.r_prim),
        dual_residual=float(solution.r_dual),
    )
    if not accepted:
        raise CBFSolverError("identical_socp_retry_failed", diagnostics)
    return candidate.reshape((num_agents, 2)), "solved_identical_socp_tighter_retry", diagnostics


def identical_nonlinear_polish(nominal, matrix, lower, max_speed, config, initial):
    nominal = np.asarray(nominal, dtype=np.float64)
    target = nominal.reshape(-1)
    num_agents = len(nominal)
    initial = np.asarray(initial, dtype=np.float64).reshape(-1)
    if initial.shape != target.shape or not np.isfinite(initial).all():
        initial = target.copy()

    def speed_residual(value):
        rows = value.reshape((num_agents, 2))
        return max_speed * max_speed - np.einsum("ij,ij->i", rows, rows)

    def speed_jacobian(value):
        result = np.zeros((num_agents, 2 * num_agents), dtype=np.float64)
        for agent in range(num_agents):
            result[agent, 2 * agent : 2 * agent + 2] = -2 * value[
                2 * agent : 2 * agent + 2
            ]
        return result

    solution = minimize(
        lambda value: 0.5 * np.dot(value - target, value - target),
        initial,
        jac=lambda value: value - target,
        method="SLSQP",
        constraints=(
            {"type": "ineq", "fun": lambda value: matrix @ value - lower, "jac": lambda value: matrix},
            {"type": "ineq", "fun": speed_residual, "jac": speed_jacobian},
        ),
        options={"ftol": 1e-14, "maxiter": 1000, "disp": False},
    )
    candidate = np.asarray(solution.x, dtype=np.float64)
    linear = matrix @ candidate - lower
    speed_values = speed_residual(candidate)
    active_rows = []
    active_residuals = []
    for row, residual in zip(matrix, linear, strict=True):
        if residual <= 1e-7:
            active_rows.append(row)
            active_residuals.append(residual)
    for row, residual in zip(speed_jacobian(candidate), speed_values, strict=True):
        if residual <= 1e-7:
            active_rows.append(row)
            active_residuals.append(residual)
    if active_rows:
        multipliers, _ = nnls(
            np.asarray(active_rows).T,
            candidate - target,
            maxiter=10000,
        )
        stationarity = float(
            np.linalg.norm(
                candidate - target - np.asarray(active_rows).T @ multipliers,
                ord=np.inf,
            )
        )
        complementarity = float(
            np.max(np.abs(multipliers * np.asarray(active_residuals)))
        )
    else:
        stationarity = float(np.linalg.norm(candidate - target, ord=np.inf))
        complementarity = 0.0
    accepted, diagnostics = _certificate(
        candidate,
        target,
        matrix,
        lower,
        max_speed,
        config,
        stationarity,
        complementarity,
    )
    diagnostics.update(
        backend="scipy_slsqp_identical_nonlinear_constraints",
        solver_status="Solved" if solution.success else str(solution.message),
        accepted=accepted,
        nonlinear_iterations=int(solution.nit),
        solver_success=bool(solution.success),
    )
    if not accepted:
        raise CBFSolverError("identical_nonlinear_polish_failed", diagnostics)
    return candidate.reshape((num_agents, 2)), "solved_identical_nonlinear_polish", diagnostics


class CertifiedHardSafetyFilter:
    """Canonical projector with same-problem numerical retry on rejection only."""

    def __init__(self, config: HardProjectionConfig | None = None):
        self.config = config or HardProjectionConfig()
        self.primary = HardSafetyFilter(self.config)

    def __call__(self, snapshot, nominal_velocity):
        try:
            return self.primary(snapshot, nominal_velocity)
        except CBFSolverError as primary_error:
            matrix, lower, geometry = barrier_constraints(snapshot, self.config)
            try:
                velocity, status, diagnostics = tighter_identical_socp(
                    nominal_velocity,
                    matrix,
                    lower,
                    snapshot["config"]["max_speed"],
                    self.config,
                )
                retry_chain = "canonical_failed -> tighter_identical_socp"
            except CBFSolverError as tighter_error:
                velocity, status, diagnostics = identical_nonlinear_polish(
                    nominal_velocity,
                    matrix,
                    lower,
                    snapshot["config"]["max_speed"],
                    self.config,
                    primary_error.details.get("solver_output_candidate", nominal_velocity),
                )
                retry_chain = "canonical_failed -> tighter_failed -> identical_nonlinear_polish"
                diagnostics["tighter_error_status"] = tighter_error.status
            diagnostics.update(
                numerical_retry=True,
                retry_chain=retry_chain,
                primary_error_status=primary_error.status,
                num_agents=int(geometry["num_agents"]),
                num_pair_constraints=len(geometry["pair_indices"]),
                num_wall_constraints=int(np.prod(geometry["wall_h"].shape)),
                pair_indices=[list(pair) for pair in geometry["pair_indices"]],
                pairwise_h=geometry["pairwise_h"].tolist(),
                min_wall_h=geometry["min_wall_h"],
            )
            return FilterResult(velocity=velocity, status=status, diagnostics=diagnostics)


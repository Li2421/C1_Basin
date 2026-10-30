"""Exact N-agent Euclidean CBF projection with unchanged constraint semantics.

This is the dimension-generic counterpart of the frozen two-agent Clarabel
operator.  It changes only the number of joint action blocks, unordered pair
rows, wall rows, and speed cones; it adds no slack, priority, mode, or fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import clarabel
import numpy as np
from scipy import sparse

from single_integrator.filters import FilterResult


@dataclass(frozen=True)
class HardProjectionConfig:
    gamma: float = 1.0
    gamma_wall: float = 1.0
    separation_buffer: float = 1e-4
    solver_ftol: float = 1e-12
    feasibility_tol: float = 1e-9
    optimality_tol: float = 2e-6
    speed_tol: float = 1e-10
    intervention_tol: float = 1e-6
    max_iterations: int = 200

    def __post_init__(self) -> None:
        for name in (
            "gamma",
            "gamma_wall",
            "separation_buffer",
            "solver_ftol",
            "feasibility_tol",
            "optimality_tol",
            "speed_tol",
            "intervention_tol",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")

    def to_dict(self) -> dict:
        return asdict(self)


class CBFSolverError(RuntimeError):
    """Numerical experiment error, never an episode outcome or fallback."""

    def __init__(self, status: str, details: dict):
        self.status, self.details = status, details
        super().__init__(f"{status}: {details}")


def _pair_indices(snapshot: dict, num_agents: int) -> tuple[tuple[int, int], ...]:
    supplied = snapshot.get("pair_indices")
    if supplied is None:
        return tuple(combinations(range(num_agents), 2))
    pairs = tuple(tuple(map(int, pair)) for pair in np.asarray(supplied))
    expected = set(combinations(range(num_agents), 2))
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("pair_indices must contain every unordered pair exactly once")
    return pairs


def barrier_geometry(snapshot: dict, config: HardProjectionConfig) -> dict:
    positions = np.asarray(snapshot["positions"], dtype=np.float64)
    walls = np.asarray(snapshot["walls"], dtype=np.float64)
    plant = snapshot["config"]
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2:
        raise ValueError("expected positions [N,2], N >= 2")
    if walls.ndim != 3 or walls.shape[1:] != (2, 2):
        raise ValueError("expected finite wall segments [M,2,2]")
    if not np.isfinite(positions).all() or not np.isfinite(walls).all():
        raise ValueError("nonfinite barrier geometry")
    num_agents = len(positions)
    pairs = _pair_indices(snapshot, num_agents)

    starts, ends = walls[:, 0], walls[:, 1]
    segment = ends - starts
    length2 = np.sum(segment * segment, axis=-1)
    if np.any(length2 <= 0):
        raise ValueError("degenerate wall segment")
    t = np.clip(
        np.sum((positions[:, None] - starts) * segment, axis=-1) / length2,
        0.0,
        1.0,
    )
    nearest = starts + t[..., None] * segment
    delta = positions[:, None] - nearest
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance <= 1e-14):
        raise CBFSolverError("undefined_wall_gradient", {"positions": positions.tolist()})
    wall_h = (
        distance
        - plant["agent_radius"]
        - plant["wall_radius"]
        - plant["wall_collision_margin"]
        - config.separation_buffer
    )
    wall_gradient = delta / distance[..., None]

    d_safe = (
        2 * plant["agent_radius"]
        + plant["agent_collision_margin"]
        + config.separation_buffer
    )
    pairwise_h = []
    pair_gradients = []
    for i, j in pairs:
        relative = positions[i] - positions[j]
        pairwise_h.append(float(relative @ relative - d_safe * d_safe))
        row = np.zeros(2 * num_agents, dtype=np.float64)
        row[2 * i : 2 * i + 2] = 2 * relative
        row[2 * j : 2 * j + 2] = -2 * relative
        pair_gradients.append(row)
    pairwise_h = np.asarray(pairwise_h, dtype=np.float64)
    pair_gradients = np.asarray(pair_gradients, dtype=np.float64)
    return {
        "num_agents": num_agents,
        "pair_indices": pairs,
        "pairwise_h": pairwise_h,
        "pair_gradients": pair_gradients,
        "wall_h": wall_h,
        "wall_gradient": wall_gradient,
        "min_wall_h": float(wall_h.min()),
        "d_safe": float(d_safe),
    }


def barrier_constraints(
    snapshot: dict, config: HardProjectionConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``A, lower`` for ``A @ flatten(u) >= lower``."""
    geometry = barrier_geometry(snapshot, config)
    num_agents = geometry["num_agents"]
    rows = list(geometry["pair_gradients"])
    lower = list(-config.gamma * geometry["pairwise_h"])
    for agent in range(num_agents):
        for gradient, h in zip(geometry["wall_gradient"][agent], geometry["wall_h"][agent]):
            row = np.zeros(2 * num_agents, dtype=np.float64)
            row[2 * agent : 2 * agent + 2] = gradient
            rows.append(row)
            lower.append(-config.gamma_wall * h)
    return np.asarray(rows), np.asarray(lower), geometry


def _diagnostics(
    candidate: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    max_speed: float,
) -> dict:
    flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    num_agents = flat.size // 2
    linear = matrix @ flat - lower
    speeds = np.linalg.norm(flat.reshape(num_agents, 2), axis=-1)
    return {
        "solver_output_candidate": flat.tolist(),
        "objective": float(0.5 * np.dot(flat - target, flat - target)),
        "min_linear_residual": float(linear.min()),
        "linear_residuals": linear.tolist(),
        "per_agent_speed": speeds.tolist(),
        "max_speed_excess": float(speeds.max() - max_speed),
        "max_constraint_violation": float(max(0.0, (-linear).max(), (speeds - max_speed).max())),
        "active_linear_constraints": np.flatnonzero(linear <= 1e-7).tolist(),
        "active_speed_constraints": np.flatnonzero(np.abs(speeds - max_speed) <= 1e-7).tolist(),
    }


def project_velocity(
    nominal: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    max_speed: float,
    config: HardProjectionConfig,
) -> tuple[np.ndarray, str, dict]:
    """Project onto all linear CBF rows and one 2-D speed SOC per agent."""
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 2:
        raise ValueError("nominal must have shape [N,2]")
    num_agents = len(nominal)
    target = nominal.reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (target, matrix, lower)):
        raise ValueError("nonfinite projection input")
    if matrix.shape != (len(lower), 2 * num_agents) or not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("invalid projection dimensions or speed limit")

    if matrix.size and np.min(matrix @ target - lower) >= 0 and np.max(
        np.linalg.norm(nominal, axis=-1)
    ) <= max_speed:
        diagnostics = _diagnostics(target, target, matrix, lower, max_speed)
        diagnostics.update(
            backend="clarabel_exact_socp",
            solver_status="nominal_feasible",
            accepted=True,
            conic_iterations=0,
            projection_input=target.tolist(),
        )
        return nominal.copy(), "nominal_feasible", diagnostics

    blocks = [sparse.csc_matrix(-matrix)]
    rhs = [-lower]
    cones = [clarabel.NonnegativeConeT(len(lower))]
    for agent in range(num_agents):
        block = np.zeros((3, 2 * num_agents), dtype=np.float64)
        block[1:, 2 * agent : 2 * agent + 2] = -np.eye(2)
        blocks.append(sparse.csc_matrix(block))
        rhs.append(np.asarray([max_speed, 0.0, 0.0]))
        cones.append(clarabel.SecondOrderConeT(3))
    cone_matrix = sparse.vstack(blocks, format="csc")
    cone_rhs = np.concatenate(rhs)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = config.max_iterations
    settings.tol_feas = min(config.feasibility_tol, config.speed_tol)
    settings.tol_gap_abs = config.solver_ftol
    settings.tol_gap_rel = config.solver_ftol
    solution = clarabel.DefaultSolver(
        sparse.csc_matrix(np.eye(2 * num_agents)),
        -target,
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    ).solve()
    candidate = np.asarray(solution.x, dtype=np.float64)
    diagnostics = _diagnostics(candidate, target, matrix, lower, max_speed)
    dual = np.asarray(solution.z, dtype=np.float64)
    slack = np.asarray(solution.s, dtype=np.float64)
    stationarity = float(np.linalg.norm(candidate - target + cone_matrix.T @ dual, ord=np.inf))
    complementarity = float(abs(np.dot(slack, dual)))
    status = str(solution.status)
    diagnostics.update(
        backend="clarabel_exact_socp",
        solver_status=status,
        accepted=False,
        conic_iterations=int(solution.iterations),
        primal_residual=float(solution.r_prim),
        dual_residual=float(solution.r_dual),
        stationarity_residual=stationarity,
        complementarity_residual=complementarity,
        projection_input=target.tolist(),
    )
    certified = (
        np.isfinite(candidate).all()
        and diagnostics["min_linear_residual"] >= -config.feasibility_tol
        and diagnostics["max_speed_excess"] <= config.speed_tol
        and max(stationarity, complementarity) <= config.optimality_tol
    )
    if not certified:
        raise CBFSolverError("conic_failed", diagnostics)
    diagnostics["accepted"] = True
    return (
        candidate.reshape(num_agents, 2).copy(),
        "solved" if status == "Solved" else "solved_kkt_certified",
        diagnostics,
    )


class HardSafetyFilter:
    """Post-hoc filter seam matching the legacy ``FilterResult`` contract."""

    def __init__(self, config: HardProjectionConfig | None = None):
        self.config = config or HardProjectionConfig()

    def __call__(self, snapshot: dict, nominal_velocity: np.ndarray) -> FilterResult:
        dt = snapshot["config"]["dt"]
        if max(self.config.gamma, self.config.gamma_wall) * dt > 1:
            raise ValueError("require gamma*dt <= 1 for the held-velocity certificate")
        matrix, lower, geometry = barrier_constraints(snapshot, self.config)
        velocity, status, diagnostics = project_velocity(
            nominal_velocity,
            matrix,
            lower,
            snapshot["config"]["max_speed"],
            self.config,
        )
        diagnostics.update(
            num_agents=int(geometry["num_agents"]),
            num_pair_constraints=len(geometry["pair_indices"]),
            num_wall_constraints=int(np.prod(geometry["wall_h"].shape)),
            pair_indices=[list(pair) for pair in geometry["pair_indices"]],
            pairwise_h=geometry["pairwise_h"].tolist(),
            min_wall_h=geometry["min_wall_h"],
        )
        return FilterResult(velocity=velocity, status=status, diagnostics=diagnostics)

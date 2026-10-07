"""Dimension-generic form of the existing three-gain diagnostic corrector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bounded_rows(vectors: np.ndarray, max_speed: float) -> np.ndarray:
    """Apply the frozen radial row bound without changing vector direction."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or not np.isfinite(vectors).all():
        raise ValueError("expected finite vectors with shape [N,2]")
    if not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("max_speed must be finite and positive")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, max_speed / np.maximum(norms, 1e-30))


def all_pair_mean_relative_basis(positions: np.ndarray, max_speed: float) -> np.ndarray:
    """Mean of individually bounded away-from-other pair vectors.

    For agent ``i`` this is

      1/(N-1) * sum_{j != i} bound(p_i - p_j).

    The order (bound each pair, then average) is deliberate.  At ``N=2`` it
    exactly reduces to the original single-opponent relation basis.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2:
        raise ValueError("positions must have shape [N,2] with N >= 2")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    values = []
    for i in range(len(positions)):
        differences = positions[i] - np.delete(positions, i, axis=0)
        values.append(bounded_rows(differences, max_speed).mean(axis=0))
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class DiagnosticEta:
    """The unchanged episode-level three-dimensional diagnostic parameter."""

    goal_feedback: float
    safe_feedback: float
    relative_feedback: float

    def __post_init__(self) -> None:
        if not np.isfinite(np.asarray(self.vector, dtype=np.float64)).all():
            raise ValueError("eta must be finite")

    @classmethod
    def from_value(cls, eta: "DiagnosticEta | tuple[float, float, float] | np.ndarray") -> "DiagnosticEta":
        if isinstance(eta, cls):
            return eta
        value = np.asarray(eta, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError("eta must contain exactly three scalars")
        return cls(*map(float, value))

    @property
    def vector(self) -> tuple[float, float, float]:
        return (float(self.goal_feedback), float(self.safe_feedback), float(self.relative_feedback))


class DiagnosticCorrector:
    """Recompute the existing three-basis correction at every physical step."""

    stochastic = False

    def __init__(self, eta: DiagnosticEta | tuple[float, float, float] | np.ndarray):
        self.eta = DiagnosticEta.from_value(eta)
        self.call_count = 0

    def bases(
        self,
        positions: np.ndarray,
        goals: np.ndarray,
        u_safe: np.ndarray,
        max_speed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        u_safe = np.asarray(u_safe, dtype=np.float64)
        if positions.shape != goals.shape or u_safe.shape != positions.shape:
            raise ValueError("positions, goals, and u_safe must share shape [N,2]")
        goal_basis = bounded_rows(goals - positions, max_speed)
        relative_basis = all_pair_mean_relative_basis(positions, max_speed)
        return goal_basis, relative_basis

    def __call__(
        self,
        positions: np.ndarray,
        goals: np.ndarray,
        u_safe: np.ndarray,
        max_speed: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        goal_basis, relative_basis = self.bases(positions, goals, u_safe, max_speed)
        a, b, c = self.eta.vector
        self.call_count += 1
        correction = a * goal_basis + b * np.asarray(u_safe, dtype=np.float64) + c * relative_basis
        return correction, goal_basis, relative_basis

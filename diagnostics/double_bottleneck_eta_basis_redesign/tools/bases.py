"""Isolated P0 and OrthoFlow3 diagnostic basis definitions."""

from __future__ import annotations

import numpy as np

from shared_control.diagnostic_corrector import bounded_rows, all_pair_mean_relative_basis


P0_LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
P0_HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)


def raw_ortho_flow(goal_basis: np.ndarray, u_safe: np.ndarray, max_speed: float) -> np.ndarray:
    """Smooth goal-orthogonal Flow residual without inventing a lateral vector."""
    goal_basis = np.asarray(goal_basis, dtype=np.float64)
    u_safe = np.asarray(u_safe, dtype=np.float64)
    epsilon = 1e-6 * float(max_speed)
    denominator = np.sum(goal_basis * goal_basis, axis=-1, keepdims=True) + epsilon * epsilon
    projection = np.sum(u_safe * goal_basis, axis=-1, keepdims=True) / denominator * goal_basis
    return u_safe - projection


def basis_terms(name: str, positions, goals, u_safe, max_speed: float, ortho_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    u_safe = np.asarray(u_safe, dtype=np.float64)
    goal = bounded_rows(goals - positions, max_speed)
    relation = all_pair_mean_relative_basis(positions, max_speed)
    if name == "P0-3D":
        return goal, u_safe, relation
    if name == "P1-OrthoFlow3":
        return goal, float(ortho_scale) * raw_ortho_flow(goal, u_safe, max_speed), relation
    raise ValueError(name)


def correction(name: str, theta, positions, goals, u_safe, max_speed: float, ortho_scale: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (3,):
        raise ValueError("theta must have shape (3,)")
    terms = basis_terms(name, positions, goals, u_safe, max_speed, ortho_scale)
    return sum(float(value) * term for value, term in zip(theta, terms, strict=True))


__all__ = ("P0_LOW", "P0_HIGH", "basis_terms", "correction", "raw_ortho_flow")

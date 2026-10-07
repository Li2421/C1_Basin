"""Isolated diagnostic eta representations; canonical corrector is untouched."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from shared_control.diagnostic_corrector import bounded_rows, all_pair_mean_relative_basis


PAIR_INDICES = tuple(combinations(range(4), 2))
BASE_LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
BASE_HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
P0_ANCHOR = np.asarray((0.5364547464996576, 0.18910571560263634, 0.024928873172029853))


@dataclass(frozen=True)
class Representation:
    name: str
    dimension: int
    low: np.ndarray
    high: np.ndarray

    def correction(self, theta, positions, goals, u_safe, max_speed, step=0, horizon=850):
        theta = np.asarray(theta, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        u_safe = np.asarray(u_safe, dtype=np.float64)
        if theta.shape != (self.dimension,):
            raise ValueError(f"{self.name} theta must have shape {(self.dimension,)}")
        goal = bounded_rows(goals - positions, max_speed)
        if self.name == "P0-3D":
            relation = all_pair_mean_relative_basis(positions, max_speed)
            return theta[0] * goal + theta[1] * u_safe + theta[2] * relation
        if self.name == "P1-Agent6":
            relation = all_pair_mean_relative_basis(positions, max_speed)
            return theta[:4, None] * goal + theta[4] * u_safe + theta[5] * relation
        if self.name == "P2-Pair8":
            relation = np.zeros_like(positions)
            for pair_index, (i, j) in enumerate(PAIR_INDICES):
                bounded = bounded_rows(np.asarray((positions[i] - positions[j],)), max_speed)[0]
                weight = theta[2 + pair_index] / 3.0
                relation[i] += weight * bounded
                relation[j] -= weight * bounded
            return theta[0] * goal + theta[1] * u_safe + relation
        if self.name == "P3-Temporal6":
            if horizon < 2 or not 0 <= step < horizon:
                raise ValueError("invalid temporal step/horizon")
            fraction = step / (horizon - 1)
            eta = (1.0 - fraction) * theta[:3] + fraction * theta[3:]
            relation = all_pair_mean_relative_basis(positions, max_speed)
            return eta[0] * goal + eta[1] * u_safe + eta[2] * relation
        if self.name == "P4-Agent12":
            relation = all_pair_mean_relative_basis(positions, max_speed)
            values = theta.reshape(3, 4)
            return values[0, :, None] * goal + values[1, :, None] * u_safe + values[2, :, None] * relation
        raise ValueError(self.name)

    def embed_p0(self, eta=P0_ANCHOR):
        a, b, c = map(float, eta)
        if self.name == "P0-3D":
            return np.asarray((a, b, c))
        if self.name == "P1-Agent6":
            return np.asarray((a, a, a, a, b, c))
        if self.name == "P2-Pair8":
            return np.asarray((a, b, c, c, c, c, c, c))
        if self.name == "P3-Temporal6":
            return np.asarray((a, b, c, a, b, c))
        if self.name == "P4-Agent12":
            return np.asarray((a,) * 4 + (b,) * 4 + (c,) * 4)
        raise ValueError(self.name)


REPRESENTATIONS = {
    "P0-3D": Representation("P0-3D", 3, BASE_LOW.copy(), BASE_HIGH.copy()),
    "P1-Agent6": Representation(
        "P1-Agent6", 6,
        np.asarray((0.5, 0.5, 0.5, 0.5, -0.5, 0.0)),
        np.asarray((1.25, 1.25, 1.25, 1.25, 0.5, 0.75)),
    ),
    "P2-Pair8": Representation(
        "P2-Pair8", 8,
        np.asarray((0.5, -0.5) + (0.0,) * 6),
        np.asarray((1.25, 0.5) + (0.75,) * 6),
    ),
    "P3-Temporal6": Representation(
        "P3-Temporal6", 6,
        np.tile(BASE_LOW, 2), np.tile(BASE_HIGH, 2),
    ),
    "P4-Agent12": Representation(
        "P4-Agent12", 12,
        np.asarray((0.5,) * 4 + (-0.5,) * 4 + (0.0,) * 4),
        np.asarray((1.25,) * 4 + (0.5,) * 4 + (0.75,) * 4),
    ),
}


def zero_theta(name: str) -> np.ndarray:
    return np.zeros(REPRESENTATIONS[name].dimension, dtype=np.float64)


__all__ = ("PAIR_INDICES", "P0_ANCHOR", "REPRESENTATIONS", "Representation", "zero_theta")

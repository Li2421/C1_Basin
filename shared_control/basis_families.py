"""Versioned fixed-D basis families for future eta-oracle/data pipelines.

The historical correctors are intentionally not modified.  New callers must
select a family explicitly through :func:`get_basis_family`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from shared_control.diagnostic_corrector import (
    all_pair_mean_relative_basis,
    bounded_rows,
)


ORTHOFLOW3_SCALE: Final[float] = 3.303687238760696
ORTHOFLOW3_SOURCE_SHA256: Final[str] = (
    "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38"
)
ORTHOFLOW3_SCALE_SHA256: Final[str] = (
    "ecdca9e63e5d345ec45ae264bcdbc1006f9281cc5a32c3fdfec901e55609a133"
)


@dataclass(frozen=True)
class BasisMetadata:
    family: str
    version: str
    dimension: int
    names: tuple[str, str, str]
    authoritative_source_sha256: str
    scale: float | None
    normalization: str
    clipping: str


@dataclass(frozen=True)
class BasisFields:
    metadata: BasisMetadata
    values: tuple[np.ndarray, np.ndarray, np.ndarray]

    def correction(self, eta: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        vector = np.asarray(eta, dtype=np.float64)
        if vector.shape != (self.metadata.dimension,) or not np.isfinite(vector).all():
            raise ValueError("eta must be a finite vector of length three")
        return sum(
            float(coefficient) * field
            for coefficient, field in zip(vector, self.values, strict=True)
        )


class BasisFamily:
    """Common three-field interface; implementations are stateless."""

    metadata: BasisMetadata

    def compute(
        self,
        positions: np.ndarray,
        goals: np.ndarray,
        u_safe: np.ndarray,
        max_speed: float,
    ) -> BasisFields:
        positions = np.asarray(positions, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        u_safe = np.asarray(u_safe, dtype=np.float64)
        if positions.shape != goals.shape or positions.shape != u_safe.shape:
            raise ValueError("positions, goals, and u_safe must share shape [N,2]")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("basis inputs must have shape [N,2]")
        if not all(np.isfinite(value).all() for value in (positions, goals, u_safe)):
            raise ValueError("basis inputs must be finite")
        return BasisFields(
            metadata=self.metadata,
            values=self._compute(positions, goals, u_safe, float(max_speed)),
        )

    def _compute(
        self,
        positions: np.ndarray,
        goals: np.ndarray,
        u_safe: np.ndarray,
        max_speed: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class P0BasisFamily(BasisFamily):
    metadata = BasisMetadata(
        family="p0",
        version="p0_legacy_v1",
        dimension=3,
        names=("B_goal", "u_safe", "B_rel"),
        authoritative_source_sha256=(
            "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40"
        ),
        scale=None,
        normalization="bounded_rows for B_goal and each relative pair; no u_safe normalization",
        clipping="row radial bound at max_speed for B_goal/B_rel only",
    )

    def _compute(self, positions, goals, u_safe, max_speed):
        return (
            bounded_rows(goals - positions, max_speed),
            u_safe.copy(),
            all_pair_mean_relative_basis(positions, max_speed),
        )


class OrthoFlow3BasisFamily(BasisFamily):
    metadata = BasisMetadata(
        family="orthoflow3",
        version="p1_orthoflow3_v1",
        dimension=3,
        names=("B_goal", "B_flow_perp_scaled", "B_rel"),
        authoritative_source_sha256=ORTHOFLOW3_SOURCE_SHA256,
        scale=ORTHOFLOW3_SCALE,
        normalization=(
            "fixed global RMS scale 3.303687238760696 from 96 archived baseline anchors; "
            "no per-state normalization"
        ),
        clipping="row radial bound at max_speed for B_goal/B_rel; no additional B_flow_perp clipping",
    )

    @staticmethod
    def raw_ortho_flow(
        goal_basis: np.ndarray, u_safe: np.ndarray, max_speed: float
    ) -> np.ndarray:
        """Exact archived P1-OrthoFlow3 smooth Tikhonov residual."""
        epsilon = 1e-6 * float(max_speed)
        denominator = (
            np.sum(goal_basis * goal_basis, axis=-1, keepdims=True)
            + epsilon * epsilon
        )
        projection = (
            np.sum(u_safe * goal_basis, axis=-1, keepdims=True)
            / denominator
            * goal_basis
        )
        return u_safe - projection

    def _compute(self, positions, goals, u_safe, max_speed):
        goal = bounded_rows(goals - positions, max_speed)
        flow_perp = self.metadata.scale * self.raw_ortho_flow(
            goal, u_safe, max_speed
        )
        relation = all_pair_mean_relative_basis(positions, max_speed)
        return goal, flow_perp, relation


_FAMILIES: Final[dict[str, BasisFamily]] = {
    "p0": P0BasisFamily(),
    "p0-3d": P0BasisFamily(),
    "orthoflow3": OrthoFlow3BasisFamily(),
    "p1-orthoflow3": OrthoFlow3BasisFamily(),
}


def get_basis_family(name: str) -> BasisFamily:
    try:
        return _FAMILIES[str(name).strip().lower()]
    except KeyError as error:
        raise ValueError(f"unknown basis family {name!r}") from error


__all__ = (
    "BasisFamily",
    "BasisFields",
    "BasisMetadata",
    "ORTHOFLOW3_SCALE",
    "OrthoFlow3BasisFamily",
    "P0BasisFamily",
    "get_basis_family",
)

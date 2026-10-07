"""Hard-safety smoke adapter for Ring Exchange; not a safety experiment.

The generic projector consumes line segments whereas this benchmark's obstacle
is a disk.  We therefore expose a conservative *circumscribed* polygonal disk
only at the safety seam.  The physical plant remains exactly circular and no
polygon, CBF value, or coordination mode is fed to MACFlow.
"""
from __future__ import annotations

import math
import numpy as np

from shared_control.hard_projection import (
    CBFSolverError, HardProjectionConfig, HardSafetyFilter, barrier_constraints,
    barrier_geometry, project_velocity,
)


def central_obstacle_polygon(radius: float, sides: int = 48) -> np.ndarray:
    """Circumscribed regular polygon whose inradius is exactly ``radius``."""
    if sides < 8 or radius <= 0:
        raise ValueError("need a positive radius and at least 8 polygon sides")
    rho = float(radius) / math.cos(math.pi / sides)
    theta = np.arange(sides, dtype=float) * 2 * math.pi / sides + math.pi / sides
    vertices = rho * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)


def outer_boundary_polygon(radius: float, sides: int = 48) -> np.ndarray:
    """Inscribed polygon used as a conservative circular containment wall.

    Agents start inside this polygon.  The generic unsigned segment-distance
    barrier therefore has an inward-pointing gradient, while keeping an agent
    radius plus collision margin away from every chord also keeps the physical
    disc inside the true circular boundary.
    """
    if sides < 8 or radius <= 0:
        raise ValueError("need a positive radius and at least 8 polygon sides")
    theta = np.arange(sides, dtype=float) * 2 * math.pi / sides
    vertices = float(radius) * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)


def polygonal_obstacle_snapshot(env, *, sides: int = 48) -> dict:
    """Make a generic CBF snapshot with six pair rows plus disk-safe walls."""
    snapshot = env.snapshot()
    # The circumscribed polygon is conservative: exterior invariance of its
    # boundary entails safety from the true central disk.  Do not approximate
    # with an inscribed polygon, which could permit physical disk collision.
    # The original adapter exposed only the central obstacle and accidentally
    # omitted the plant's outer boundary.  Both geometries are physical hard
    # constraints, so expose both through the same generic wall-CBF seam.
    snapshot["walls"] = np.concatenate((
        central_obstacle_polygon(env.config.obstacle_radius, sides),
        outer_boundary_polygon(env.config.outer_radius, sides),
    ))
    snapshot["wall_names"] = (tuple("central_obstacle" for _ in range(sides)) +
                              tuple("outer_boundary" for _ in range(sides)))
    snapshot["config"] = {**snapshot["config"], "wall_radius": 0.0,
                          "wall_collision_margin": env.config.collision_margin}
    return snapshot


def all_pairwise_and_obstacle_constraints(env, config: HardProjectionConfig | None = None, *, sides: int = 48):
    """Return all six agent-pair constraints and conservative disk constraints."""
    cfg = config or HardProjectionConfig()
    matrix, lower, geometry = barrier_constraints(polygonal_obstacle_snapshot(env, sides=sides), cfg)
    if tuple(geometry["pair_indices"]) != env.pair_indices or len(geometry["pair_indices"]) != 6:
        raise AssertionError("Ring Exchange must expose all six unordered pair constraints")
    return matrix, lower, geometry


def hard_safety_smoke(env=None, config: HardProjectionConfig | None = None) -> dict:
    """One compatibility check; deliberately no policy rollout or comparison."""
    from .environment import RingExchangeEnv
    plant = env or RingExchangeEnv()
    filter_ = HardSafetyFilter(config)
    result = filter_(polygonal_obstacle_snapshot(plant), np.zeros((4, 2)))
    if result.diagnostics["num_pair_constraints"] != 6:
        raise AssertionError("hard safety smoke omitted an agent pair")
    return {"status": result.status, "num_pair_constraints": result.diagnostics["num_pair_constraints"],
            "num_obstacle_constraints": result.diagnostics["num_wall_constraints"],
            "pair_indices": result.diagnostics["pair_indices"], "obstacle": "central_disk_circumscribed_polygon"}


__all__ = ("HardProjectionConfig", "CBFSolverError", "HardSafetyFilter", "barrier_geometry", "barrier_constraints",
           "project_velocity", "central_obstacle_polygon", "outer_boundary_polygon", "polygonal_obstacle_snapshot",
           "all_pairwise_and_obstacle_constraints", "hard_safety_smoke")

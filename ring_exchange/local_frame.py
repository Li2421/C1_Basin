"""Rotation-equivariant physical coordinate transforms for Ring v6.

The world plant remains Cartesian.  Only the data/policy chart changes: each
agent's coordinates use its instantaneous outward radial and CCW tangent axes.
No discrete direction, priority, or coordination variable is constructed.
"""
from __future__ import annotations
import numpy as np


def frames(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return outward radial and CCW tangent unit vectors, each ``[4,2]``."""
    p = np.asarray(positions, dtype=np.float64)
    if p.shape != (4, 2): raise ValueError("positions must be [4,2]")
    radial = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    tangent = np.stack((-radial[:, 1], radial[:, 0]), axis=-1)
    return radial, tangent


def vectors_to_local(vectors: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Express one world vector per agent in its radial/tangential frame."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape != (4, 2): raise ValueError("vectors must be [4,2]")
    radial, tangent = frames(positions)
    return np.stack((np.sum(v * radial, axis=-1), np.sum(v * tangent, axis=-1)), axis=-1)


def local_actions_to_world(local_actions: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Deterministically map the joint local eight-vector to world velocity."""
    local = np.asarray(local_actions, dtype=np.float64)
    if local.shape != (4, 2): raise ValueError("local_actions must be [4,2]")
    radial, tangent = frames(positions)
    return local[:, :1] * radial + local[:, 1:2] * tangent


def world_actions_to_local(world_actions: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return vectors_to_local(world_actions, positions)


def local_observation(positions: np.ndarray, velocities: np.ndarray, goals: np.ndarray, config) -> np.ndarray:
    """Rotation-invariant local observation, shape ``[4,23]``.

    Field layout deliberately preserves the v1 dimensions/order: own
    ``[radius, 0]``, own local velocity, goal displacement local, each other's
    relative position and velocity expressed in *observer i*'s frame, then
    fixed obstacle geometry.  `radius,0` is a physical position encoding, not
    an artificial mode token.
    """
    p, v, g = (np.asarray(value, dtype=np.float64) for value in (positions, velocities, goals))
    if any(value.shape != (4, 2) for value in (p, v, g)): raise ValueError("state arrays must be [4,2]")
    radial, tangent = frames(p)
    rows=[]
    for i in range(4):
        basis=np.stack((radial[i], tangent[i]), axis=0)
        project=lambda x: basis @ x
        relative=[]
        for j in range(4):
            if i != j: relative.extend((project(p[j]-p[i]), project(v[j]-v[i])))
        radius=float(np.linalg.norm(p[i]))
        own_position=np.array((radius, 0.0))
        own_velocity=project(v[i]); goal_delta=project(g[i]-p[i])
        geometry=np.array((-radius, 0.0, config.obstacle_radius, config.outer_radius,
                           radius-config.obstacle_radius-config.agent_radius))
        rows.append(np.concatenate((own_position, own_velocity, goal_delta, *relative, geometry)))
    return np.asarray(rows,dtype=np.float32)


__all__=("frames","vectors_to_local","world_actions_to_local","local_actions_to_world","local_observation")

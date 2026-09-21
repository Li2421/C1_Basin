"""Trajectory identity preservation for later C1 rollout-to-FM extraction."""
from typing import Iterable, Mapping

import numpy as np


def concatenate_trajectory_samples(trajectories: Iterable[Mapping[str, np.ndarray]]):
    """Concatenate per-trajectory step arrays and attach immutable ``trajectory_id``.

    Each mapping must contain identically sized leading time axes.  No risk is
    calculated here: callers attach a supplied trajectory-level scalar later.
    """
    pieces = list(trajectories)
    if not pieces:
        raise ValueError("at least one trajectory is required")
    common = set(pieces[0]).difference({"trajectory_id"})
    if not common or any(set(x).difference({"trajectory_id"}) != common for x in pieces):
        raise ValueError("trajectories must have identical non-identity fields")
    result = {key: [] for key in common}
    identities = []
    for fallback_id, trajectory in enumerate(pieces):
        lengths = {np.asarray(trajectory[key]).shape[0] for key in common}
        if len(lengths) != 1:
            raise ValueError("all fields of a trajectory need the same leading length")
        length = lengths.pop()
        trajectory_id = int(trajectory.get("trajectory_id", fallback_id))
        for key in common:
            result[key].append(np.asarray(trajectory[key]))
        identities.append(np.full(length, trajectory_id, dtype=np.int64))
    return {**{key: np.concatenate(value) for key, value in result.items()},
            "trajectory_id": np.concatenate(identities)}

"""Hard-safety compatibility exports for Four-Way (not an evaluation)."""
from shared_control.hard_projection import (
    HardProjectionConfig, CBFSolverError, barrier_geometry, barrier_constraints,
    project_velocity, HardSafetyFilter,
)


def all_pairwise_constraints(env, config: HardProjectionConfig | None = None):
    """Return the six unordered agent constraints plus boundary rows.

    This is a smoke-test adapter only; it does not modify the environment or
    invoke a scientific hard-safety comparison.
    """
    cfg=config or HardProjectionConfig()
    matrix, lower, geometry=barrier_constraints(env.snapshot(),cfg)
    if tuple(geometry["pair_indices"]) != env.pair_indices or len(geometry["pair_indices"]) != 6:
        raise AssertionError("Four-Way must expose all six pairwise constraints")
    return matrix, lower, geometry

__all__=("HardProjectionConfig","CBFSolverError","HardSafetyFilter","barrier_geometry","barrier_constraints","project_velocity","all_pairwise_constraints")

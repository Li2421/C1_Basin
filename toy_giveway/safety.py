"""Toy Give-Way hard-safety compatibility exports.

These names alias the frozen two-agent implementation.  N-agent safety code
for other scenarios must not be added here.
"""

from single_integrator.cbf import (
    CBFConfig,
    CBFSafetyFilter,
    CBFSolverError,
    barrier_constraints,
    barrier_geometry,
    cbf_factory,
    project_velocity,
)
from single_integrator.filters import FilterResult, IdentityFilter, identity_factory

__all__ = [
    "CBFConfig",
    "CBFSafetyFilter",
    "CBFSolverError",
    "FilterResult",
    "IdentityFilter",
    "barrier_constraints",
    "barrier_geometry",
    "cbf_factory",
    "identity_factory",
    "project_velocity",
]

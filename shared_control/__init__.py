"""Scenario-independent closed-loop control primitives.

The legacy :mod:`single_integrator` package remains the source of truth for
the original two-agent experiment.  This package contains only the minimal
dimension-generic pieces needed to reuse the same mathematical semantics in
new scenarios.
"""

from .diagnostic_corrector import (
    DiagnosticCorrector,
    DiagnosticEta,
    all_pair_mean_relative_basis,
    bounded_rows,
)
from .hard_projection import (
    CBFSolverError,
    HardProjectionConfig,
    HardSafetyFilter,
    barrier_constraints,
    barrier_geometry,
    project_velocity,
)

__all__ = (
    "CBFSolverError",
    "DiagnosticCorrector",
    "DiagnosticEta",
    "HardProjectionConfig",
    "HardSafetyFilter",
    "all_pair_mean_relative_basis",
    "barrier_constraints",
    "barrier_geometry",
    "bounded_rows",
    "project_velocity",
)

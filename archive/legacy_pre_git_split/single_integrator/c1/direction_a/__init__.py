"""Frozen Direction A stochastic residual policy and score estimators.

This package is deliberately separate from the archived certificate risks and
from the primal--dual training entry points.  It does not provide a training
loop.
"""

from .policy import (
    DirectionADistribution,
    DirectionAResidual,
    initialization_logits,
    log_probability,
    policy_parameters,
    sample_residual,
)
from .randomness import IndexedRandomTape

__all__ = [
    "DirectionADistribution",
    "DirectionAResidual",
    "IndexedRandomTape",
    "initialization_logits",
    "log_probability",
    "policy_parameters",
    "sample_residual",
]

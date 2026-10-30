"""Post-hoc filter contract. Identity is a plumbing test, NOT a CBF method."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FilterResult:
    velocity: np.ndarray
    status: str = 'ok'
    diagnostics: dict = field(default_factory=dict)


class IdentityFilter:
    def __call__(self, snapshot, nominal_velocity):
        return FilterResult(nominal_velocity.copy(), status='identity')


def identity_factory(config):
    return IdentityFilter()

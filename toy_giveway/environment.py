"""Toy Give-Way plant compatibility exports.

There is intentionally no copied implementation here.  The exported objects
are the original objects, not subclasses, so class identity, serialization,
configuration fingerprints, and dynamics remain unchanged.
"""

from single_integrator.environment import (
    AgentView,
    Config,
    GiveWayEnv,
    bounded_nominal,
    point_segment_distance,
    segment_distance,
)

__all__ = [
    "AgentView",
    "Config",
    "GiveWayEnv",
    "bounded_nominal",
    "point_segment_distance",
    "segment_distance",
]

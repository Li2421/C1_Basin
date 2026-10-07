"""Stable Toy Give-Way scenario namespace.

The historical implementation remains in :mod:`single_integrator` so old
experiments, source manifests, and imports keep working byte-for-byte.  This
package is the non-destructive scenario entry point for new code.

Importing :mod:`toy_giveway` itself is deliberately lightweight: it does not
load JAX, a Flow-BC checkpoint, or any generated result.
"""

from toy_giveway.environment import (
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

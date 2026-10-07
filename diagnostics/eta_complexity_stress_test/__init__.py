"""Isolated 8-agent stress test for the repository's shared 3-D eta law.

The package intentionally contains no learned policy, checkpoint, or training
entry point.  Its plant and oracle are diagnostic-only and do not alter the
Double-Bottleneck pipeline.
"""

from .environment import CoupledDualIntersectionEnv, StressConfig

__all__ = ("CoupledDualIntersectionEnv", "StressConfig")

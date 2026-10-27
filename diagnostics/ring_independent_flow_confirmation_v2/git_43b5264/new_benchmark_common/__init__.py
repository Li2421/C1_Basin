"""Shared, scenario-neutral Stage-I machinery for the new four-agent benchmarks.

Submodules are deliberately not imported here: geometry/expert pilot code may
use :mod:`new_benchmark_common.protocol` without importing JAX, Flax, or the
official MACFlow checkout.  Import concrete APIs from their defining module,
for example ``from new_benchmark_common.macflow import JointMACFlowAgent``.
"""

__all__ = ()

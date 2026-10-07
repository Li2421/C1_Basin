"""Frozen Toy Give-Way Flow-BC architecture compatibility exports.

This module is opt-in because importing it loads the JAX/Flax stack.  It does
not locate or load a checkpoint by itself.
"""

# The maintained evaluator owns the historical MACFlow dependency-path setup.
# Import through it so this namespace behaves exactly like checkpoint loading in
# the legacy entry point, rather than introducing a second path convention.
from single_integrator.evaluate import GiveWayFlowBCAgent, get_config

__all__ = ["GiveWayFlowBCAgent", "get_config"]

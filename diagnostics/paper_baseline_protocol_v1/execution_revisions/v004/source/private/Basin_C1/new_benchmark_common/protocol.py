"""Explicit, small scenario contracts used by the shared benchmark pipeline.

The protocol deliberately contains no coordination-mode field.  An expert may
search orders or circulation hypotheses internally, but the deployed policy
only receives physical observations supplied by the scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ExpertContinuation:
    """A collision-free centralized expert continuation.

    ``states`` and ``observations`` include the anchor state, while ``actions``
    contains one action per transition.  ``metadata`` may carry *analysis only*
    mode statistics; it is never consumed by :mod:`new_benchmark_common.macflow`.
    """

    success: bool
    terminal_reason: str
    states: Array
    observations: Array
    actions: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ScenarioProtocol(Protocol):
    """Minimum scenario API for nominal/recovery data and closed-loop tests.

    Implementations normally wrap a fresh environment per query.  ``state`` is
    intentionally opaque: a scenario can preserve velocities, timers, or other
    physical plant state without forcing a common representation.
    """

    name: str
    agent_order: tuple[str, ...]
    observation_shape: tuple[int, int]
    action_shape: tuple[int, int]
    environment_fingerprint: str

    def sample_initial_state(self, split: str, rng: np.random.Generator) -> Any:
        """Draw an independent train/dev/test state from the declared split."""

    def expert(self, initial_state: Any, rng: np.random.Generator) -> ExpertContinuation:
        """Return one centralized joint-expert continuation."""

    def perturb_state(self, state: Any, rng: np.random.Generator) -> Any:
        """Apply the scenario's common local recovery perturbation distribution."""

    def valid_state(self, state: Any) -> bool:
        """Return false for physical invalidity (inside a wall/obstacle, etc.)."""


@dataclass(frozen=True)
class RolloutAdapter:
    """Duck-typed closed-loop adapter for generic evaluation.

    The functions are injected rather than prescribed on an environment class,
    so existing scenario environments stay isolated.
    """

    make_env: Callable[[], Any]
    reset: Callable[[Any, Any], None]
    observation: Callable[[Any], Array]
    step: Callable[[Any, Array], tuple[Array, bool, Mapping[str, Any]]]
    snapshot: Callable[[Any], Any]
    restore: Callable[[Any, Any], None]
    state_distance: Callable[[Any, Any], float]

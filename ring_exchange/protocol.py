"""Neutral adapter from Ring Exchange to the shared Stage-I data protocol.

The opaque state carries positions, velocities, and goals because recovery
queries must preserve the particular opposite-side exchange task.  It contains
no CW/CCW/mixed field; those names may appear only in expert metadata for later
analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np

from .environment import AGENT_NAMES, Config, RingExchangeEnv, sample_initial_state
from .expert import CentralizedExpert, circulation_signature, infer_circulation_hypothesis


@dataclass(frozen=True)
class ExpertContinuation:
    """Structural twin of the shared protocol result.

    Keeping this tiny result type local means importing the physical benchmark
    does not import Flax/MACFlow.  The shared dataset code uses this documented
    structural contract (six identically named fields), so no adapter or mode
    field is needed when its optional training dependencies are installed.
    """
    success: bool
    terminal_reason: str
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RingExchangeScenario:
    """``ScenarioProtocol`` implementation for nominal and recovery data."""
    config: Config = Config()
    perturb_position_std: float = 0.075
    perturb_velocity_std: float = 0.055
    name: str = "ring_exchange"
    agent_order: tuple[str, ...] = AGENT_NAMES
    observation_shape: tuple[int, int] = (4, 23)
    action_shape: tuple[int, int] = (4, 2)

    @property
    def environment_fingerprint(self) -> str:
        return self.config.fingerprint

    def sample_initial_state(self, split: str, rng: np.random.Generator) -> dict:
        # The split-specific generator inside sample_initial_state protects
        # test independence even when a caller uses the same upstream seed.
        seed = int(rng.integers(0, np.iinfo(np.int64).max))
        declared_split = "development" if split == "dev" else split
        draw = sample_initial_state(split=declared_split, seed=seed, config=self.config)
        return {"positions": draw.positions.copy(), "velocities": draw.velocities.copy(),
                "goals": draw.goals.copy(), "split": split, "draw_seed": seed}

    def valid_state(self, state) -> bool:
        try:
            state = self._state_dict(state)
            RingExchangeEnv(self.config).reset(state["positions"], velocities=state["velocities"], goals=state["goals"])
            return True
        except (ValueError, TypeError, KeyError):
            return False

    def perturb_state(self, state, rng: np.random.Generator) -> dict:
        source = self._state_dict(state)
        return {"positions": source["positions"] + rng.normal(0, self.perturb_position_std, (4, 2)),
                "velocities": source["velocities"] + rng.normal(0, self.perturb_velocity_std, (4, 2)),
                # Goals deliberately survive recovery re-query intact.
                "goals": source["goals"].copy(), "split": source.get("split", "train"), "recovery": True}

    def expert(self, initial_state, rng: np.random.Generator) -> ExpertContinuation:
        state = self._state_dict(initial_state)
        env = RingExchangeEnv(self.config)
        env.reset(state["positions"], velocities=state["velocities"], goals=state["goals"])
        inferred = infer_circulation_hypothesis(state["positions"], state["velocities"])
        try:
            if bool(initial_state.get("recovery", False)) and inferred is not None:
            # A post-entry recovery state already physically encodes its
            # circulation sign in v.  Continue it instead of allowing a
            # shortest-path re-plan to demand a 180-degree reversal.
                plan = CentralizedExpert().plan_hypothesis(env, inferred)
                selection = "velocity_inferred_continuation"
            else:
            # Nominal/radial states stay genuinely direction-multimodal; the
            # expert searches internally and chooses the shorter realization.
                plan = CentralizedExpert().plan(env)
                selection = "shortest_joint_hypothesis_search"
        except RuntimeError:
            return ExpertContinuation(False, "no_valid_recovery", np.empty((0,4,2)), np.empty((0,4,23),dtype=np.float32), np.empty((0,4,2)), {})
        return ExpertContinuation(plan.success, plan.terminal_reason, plan.positions, plan.observations, plan.actions,
                                  {**plan.metadata, "realized_circulation": circulation_signature(plan.positions),
                                   "expert_mode_selection": selection})

    def recovery_state_at(self, trajectory, time: int) -> dict:
        """Recover a complete valid physical state from an expert trajectory.

        Generic trajectory arrays contain positions only.  Ring goals must be
        copied from the nominal initial state rather than regenerated, and the
        observation's last applied velocity is the preceding expert action.
        This helper is intentionally mode-free and is used before local
        perturb-and-requery acquisition.
        """
        index = int(time)
        positions = np.asarray(trajectory.states, dtype=np.float64)
        actions = np.asarray(trajectory.actions, dtype=np.float64)
        if index < 0 or index >= len(positions):
            raise IndexError("recovery time must index an existing trajectory state")
        initial = self._state_dict(trajectory.initial_state)
        velocity = np.zeros((4, 2), dtype=np.float64) if index == 0 else actions[index - 1].copy()
        return {"positions": positions[index].copy(), "velocities": velocity,
                "goals": initial["goals"].copy(), "split": initial.get("split", trajectory.split),
                "recovery": True, "source_time": index}

    @staticmethod
    def _state_dict(state) -> dict:
        if not isinstance(state, dict):
            raise TypeError("Ring protocol states must preserve positions, velocities, and goals in a dictionary")
        return {"positions": np.asarray(state["positions"], dtype=np.float64),
                "velocities": np.asarray(state.get("velocities", np.zeros((4, 2))), dtype=np.float64),
                "goals": np.asarray(state["goals"], dtype=np.float64), "split": state.get("split", "train")}


Scenario = RingExchangeScenario

__all__ = ("RingExchangeScenario", "Scenario")

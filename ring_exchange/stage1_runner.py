"""Thin Ring Exchange wiring for generic Stage-I collection/evaluation.

This module does not train on import and contains no OrthoFlow3, eta, basin,
G_phi, trajectory chunk, persistent latent, or circulation-conditioned policy.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np

from new_benchmark_common.dataset import collect_nominal_and_uniform
from new_benchmark_common.protocol import RolloutAdapter
from .environment import Config, RingExchangeEnv
from .protocol import RingExchangeScenario


def make_scenario(config: Config | None = None) -> RingExchangeScenario:
    return RingExchangeScenario(config or Config())


def collect_base_u(root: str | Path, *, config: Config | None = None,
                   nominal_counts: dict[str, int] | None = None, uniform_anchors: int = 8, seed: int = 0):
    """Collect broad nominal and uniformly anchored recovery data only.

    Test data contains nominal expert trajectories and is never used as a
    recovery source, enforcing untouched-test integrity at the API boundary.
    """
    scenario = make_scenario(config)
    counts = nominal_counts or {"train": 120, "dev": 30, "test": 60}
    return collect_nominal_and_uniform(scenario, root, scenario_config=asdict(scenario.config),
                                       nominal_counts=counts, uniform_anchors=uniform_anchors, seed=seed)


def rollout_adapter(config: Config | None = None) -> RolloutAdapter:
    cfg = config or Config()
    def reset(env, state):
        env.reset(state["positions"], velocities=state.get("velocities"), goals=state["goals"])
    def step(env, action):
        observation, _, done, info = env.step(action)
        return observation, done, info
    def restore(env, state): env.restore_augmented_state(state)
    def distance(left, right):
        return float(np.sqrt(np.mean((np.asarray(left["positions"]) - np.asarray(right["positions"])) ** 2)))
    return RolloutAdapter(make_env=lambda: RingExchangeEnv(cfg), reset=reset,
                          observation=lambda env: env.observation(), step=step,
                          snapshot=lambda env: env.augmented_state(), restore=restore, state_distance=distance)


def hard_safety_smoke(config: Config | None = None) -> dict:
    """Smoke-test six pairs plus the central obstacle, with no formal eval."""
    from .safety import hard_safety_smoke as smoke
    return smoke(RingExchangeEnv(config or Config()))


__all__ = ("make_scenario", "collect_base_u", "rollout_adapter", "hard_safety_smoke")

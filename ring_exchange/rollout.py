"""Controller-agnostic rollout utilities and analysis-only circulation labels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .environment import RingExchangeEnv
from .expert import circulation_signature


@dataclass(frozen=True)
class RolloutResult:
    positions: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    summary: dict
    terminal_reason: str
    circulation: str


def run_rollout(env: RingExchangeEnv, policy, max_steps: int | None = None) -> RolloutResult:
    """Run ``policy(observation) -> [4,2]`` without hidden action clipping."""
    positions, observations, actions, info = [env.positions.copy()], [env.observation().copy()], [], None
    horizon = env.config.max_steps if max_steps is None else min(int(max_steps), env.config.max_steps)
    for _ in range(horizon):
        action = np.asarray(policy(env.observation()), dtype=np.float64)
        observation, _, done, info = env.step(action)
        actions.append(action.copy()); positions.append(env.positions.copy()); observations.append(observation.copy())
        if done:
            break
    terminal = "truncated" if info is None or not env.done else str(info["termination"])
    trajectory = np.asarray(positions)
    return RolloutResult(trajectory, np.asarray(actions), np.asarray(observations, dtype=np.float32),
                         env.summary(), terminal, circulation_signature(trajectory))


__all__ = ("RolloutResult", "run_rollout")

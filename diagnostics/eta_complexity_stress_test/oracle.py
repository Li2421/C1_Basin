"""Separate centralized scheduler used only to prove benchmark solvability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import WaypointNominalController
from .environment import CoupledDualIntersectionEnv, StressConfig


ORACLE_MODES = {
    "ltr_then_vertical_then_rtl": ((0,), (1,), (4, 6), (5, 7), (2,), (3,)),
    "vertical_then_ltr_then_rtl": ((4, 6), (5, 7), (0,), (1,), (2,), (3,)),
}

# The incoming right-to-left queue begins intentionally close to the right
# doorway.  The oracle first parks it deeper in the staging bay; this explicit
# action prevents an immobile queue from occupying the other wave's exit.  It
# is a centralized right-of-way action and is never exposed to eta rollouts.
PARK_AGENTS = (2, 3, 4, 5, 6, 7)
PARK_TARGETS = np.asarray(((3.85, 0.35), (3.85, -0.35), (-1.68, 2.00), (-1.32, -2.00), (1.68, 2.00), (1.32, -2.00)), dtype=np.float64)


@dataclass(frozen=True)
class OracleTrace:
    mode: str
    termination: str
    positions: np.ndarray
    actions: np.ndarray
    active_agents: np.ndarray
    infos: tuple[dict, ...]

    @property
    def success(self) -> bool:
        return self.termination == "success"


def rollout_oracle(
    initial_positions: np.ndarray,
    mode: str = "ltr_then_vertical_then_rtl",
    config: StressConfig | None = None,
) -> OracleTrace:
    """Move exactly one scheduled agent at a time along its geometric route.

    This deliberately explicit right-of-way schedule is excluded from eta
    comparisons.  It proves that a failed eta rollout is not a geometric
    impossibility of the fixed benchmark instance.
    """
    if mode not in ORACLE_MODES:
        raise ValueError(f"unknown oracle mode {mode!r}")
    env = CoupledDualIntersectionEnv(config)
    env.reset(initial_positions)
    nominal = WaypointNominalController(env.goals, env.config.max_speed)
    schedule = ORACLE_MODES[mode]
    current = -1
    positions, actions, active_agents, infos = [env.positions.copy()], [], [], []
    termination = "running"
    for _ in range(env.config.max_steps):
        while current >= 0 and current < len(schedule) and all(nominal.complete(env.positions, agent) for agent in schedule[current]):
            current += 1
        active = np.zeros(8, dtype=bool)
        if current == -1:
            active[list(PARK_AGENTS)] = True
            displacement = PARK_TARGETS - env.positions[list(PARK_AGENTS)]
            action = np.zeros((8, 2), dtype=np.float64)
            action[list(PARK_AGENTS)] = env.config.max_speed * displacement / np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-12)
            if np.all(np.linalg.norm(displacement, axis=-1) <= nominal.waypoint_tolerance):
                current = 0
                action = nominal.action(env.positions, active=np.zeros(8, dtype=bool))
        elif current < len(schedule):
            active[list(schedule[current])] = True
            action = nominal.action(env.positions, active=active)
        else:
            action = nominal.action(env.positions, active=active)
        _, _, done, info = env.step(action)
        positions.append(env.positions.copy()); actions.append(action); active_agents.append(active.copy()); infos.append(info)
        termination = str(info["termination"])
        if done:
            break
    return OracleTrace(mode, termination, np.asarray(positions), np.asarray(actions), np.asarray(active_agents, dtype=bool), tuple(infos))


def verify_oracle(items, config: StressConfig | None = None, mode: str = "ltr_then_vertical_then_rtl") -> dict:
    items = list(items)
    results = [rollout_oracle(item.positions, mode=mode, config=config) for item in items]
    successes = sum(result.success for result in results)
    failures = [
        {
            "id": getattr(item, "identifier", str(index)),
            "family": getattr(item, "family", "unknown"),
            "termination": result.termination,
            "steps": len(result.actions),
        }
        for index, (item, result) in enumerate(zip(items, results)) if not result.success
    ]
    return {
        "mode": mode, "rollouts": len(results), "successes": successes,
        "success_rate": successes / len(results) if results else float("nan"),
        "terminations": {name: sum(result.termination == name for result in results) for name in ("success", "collision", "deadlock", "timeout")},
        "mean_steps": float(np.mean([len(result.actions) for result in results])) if results else float("nan"),
        "failures": failures,
    }

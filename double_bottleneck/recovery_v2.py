"""Scenario-local Recovery-V2 expert-query and coverage utilities.

This module never changes or conditions the MACFlow policy.  A normal
``CoordinationHypothesis`` chosen by the existing centralized planner seeds a
local feedback replay of that planner's validated reference.  The hypothesis
and reference are expert-side dataset-generation context only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .environment import Config, DoubleBottleneckEnv
from .expert_dataset import infer_coordination_mode


PHASES = (
    "initial_approach",
    "first_bottleneck_approach",
    "waiting_yielding",
    "coordination_mode_transition",
    "bottleneck_traversal",
    "chamber_traversal",
    "second_bottleneck",
    "final_goal_approach",
    "near_goal_termination",
)


def radial_bound(values: np.ndarray, limit: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values * np.minimum(1.0, float(limit) / np.maximum(norm, 1e-30))


def observation_from_state(
    positions: np.ndarray, velocities: np.ndarray, goals: np.ndarray
) -> np.ndarray:
    rows = []
    for agent in range(4):
        relative = []
        for other in range(4):
            if other != agent:
                relative.extend(
                    (
                        positions[other] - positions[agent],
                        velocities[other] - velocities[agent],
                    )
                )
        rows.append(
            np.concatenate(
                (
                    positions[agent],
                    velocities[agent],
                    goals[agent] - positions[agent],
                    *relative,
                )
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _state_speed(episode) -> np.ndarray:
    actions = np.asarray(episode.actions, dtype=np.float64)
    speed = np.linalg.norm(actions, axis=-1)
    return np.concatenate((speed, speed[-1:]), axis=0)


def transition_mask(episode, radius: int = 3) -> np.ndarray:
    actions = np.asarray(episode.actions, dtype=np.float64)
    speed = _state_speed(episode)
    active = speed > 0.025
    event = np.zeros(len(episode.positions), dtype=bool)
    event[1:] |= np.any(active[1:] != active[:-1], axis=1)
    if len(actions) > 1:
        delta = np.linalg.norm(actions[1:] - actions[:-1], axis=-1)
        event[1 : len(actions)] |= np.any(delta > 0.08, axis=1)
    expanded = np.zeros_like(event)
    for index in np.flatnonzero(event):
        expanded[max(0, index - radius) : min(len(event), index + radius + 1)] = True
    return expanded


def phase_labels(episode, config: Config) -> np.ndarray:
    """Exclusive, geometry/event-defined labels used by V1 and V2 audits."""
    positions = np.asarray(episode.positions, dtype=np.float64)
    goals = np.asarray(episode.metadata["environment"]["goals"], dtype=np.float64)
    speed = _state_speed(episode)
    active = speed > 0.025
    goal_error = np.linalg.norm(goals[None] - positions, axis=-1)
    transition = transition_mask(episode)
    outer_gate = config.chamber_half_length + config.bottleneck_length
    inner = config.chamber_half_length
    travel_sign = np.asarray((1.0, 1.0, -1.0, -1.0))
    labels: list[str] = []
    for time, point in enumerate(positions):
        unfinished = goal_error[time] > config.goal_tolerance
        if time >= len(positions) - 21 or np.max(goal_error[time]) <= 0.20:
            labels.append("near_goal_termination")
            continue
        goal_side_outer = travel_sign * point[:, 0] >= outer_gate
        if np.any(unfinished & goal_side_outer & (goal_error[time] <= 0.75)):
            labels.append("final_goal_approach")
            continue
        if transition[time]:
            labels.append("coordination_mode_transition")
            continue
        waiting = (~active[time]) & unfinished & (goal_error[time] > 0.20)
        if np.any(waiting) and np.any(active[time]):
            labels.append("waiting_yielding")
            continue
        left_gate = (point[:, 0] >= -outer_gate) & (point[:, 0] <= -inner)
        right_gate = (point[:, 0] >= inner) & (point[:, 0] <= outer_gate)
        destination_gate = np.where(travel_sign > 0, right_gate, left_gate)
        origin_gate = np.where(travel_sign > 0, left_gate, right_gate)
        if np.any(destination_gate):
            labels.append("second_bottleneck")
            continue
        if np.any(origin_gate):
            labels.append("bottleneck_traversal")
            continue
        if np.any((np.abs(point[:, 0]) < inner) & active[time]):
            labels.append("chamber_traversal")
            continue
        near_origin_gate = (
            (travel_sign * point[:, 0] < -inner)
            & (travel_sign * point[:, 0] >= -outer_gate - 0.60)
            & active[time]
        )
        labels.append(
            "first_bottleneck_approach" if np.any(near_origin_gate) else "initial_approach"
        )
    return np.asarray(labels)


def _zones(positions: np.ndarray, config: Config) -> np.ndarray:
    x = np.asarray(positions, dtype=np.float64)[..., 0]
    outer = config.chamber_half_length + config.bottleneck_length
    inner = config.chamber_half_length
    result = np.full(x.shape, 2, dtype=np.int8)  # chamber
    result[x < -outer] = 0
    result[(x >= -outer) & (x <= -inner)] = 1
    result[(x >= inner) & (x <= outer)] = 3
    result[x > outer] = 4
    return result


def transition_anchors(episode, config: Config) -> list[dict[str, Any]]:
    """Return sparse critical transition/wait anchors with interpretable event tags."""
    actions = np.asarray(episode.actions, dtype=np.float64)
    speed = _state_speed(episode)
    active = speed > 0.025
    positions = np.asarray(episode.positions, dtype=np.float64)
    zones = _zones(positions, config)
    labels = phase_labels(episode, config)
    events: dict[int, set[str]] = {}

    def add(step: int, label: str) -> None:
        if 0 <= step < len(actions):
            events.setdefault(int(step), set()).add(label)

    for step in range(1, len(actions)):
        started = np.flatnonzero((~active[step - 1]) & active[step])
        stopped = np.flatnonzero(active[step - 1] & (~active[step]))
        if len(started):
            add(step, "yield_to_move_or_convoy_release")
        if len(stopped):
            add(step, "move_to_yield_or_convoy_park")
        changed_zone = np.flatnonzero(zones[step] != zones[step - 1])
        if len(changed_zone):
            add(step, "bottleneck_or_chamber_entry_exit")
        if np.any(np.linalg.norm(actions[step] - actions[step - 1], axis=-1) > 0.08):
            add(step, "velocity_or_heading_transition")
        if len(started) and np.any(
            np.abs(positions[step, started, 0]) < config.chamber_half_length
        ):
            add(step, "chamber_reacceleration")
    # Directed centre crossings are explicit coordination-order events.
    travel = np.asarray((1.0, 1.0, -1.0, -1.0))
    signed = travel[None] * positions[:, :, 0]
    for step in range(1, len(actions)):
        if np.any((signed[step - 1] < 0.0) & (signed[step] >= 0.0)):
            add(step, "coordination_order_crossing")

    # Sample boundaries and interior quantiles of each long waiting interval.
    is_wait = labels[: len(actions)] == "waiting_yielding"
    begin = None
    for step in range(len(is_wait) + 1):
        value = bool(is_wait[step]) if step < len(is_wait) else False
        if value and begin is None:
            begin = step
        if not value and begin is not None:
            end = step - 1
            if end - begin + 1 >= 5:
                for fraction, tag in (
                    (0.0, "waiting_entry"),
                    (0.5, "waiting_interior"),
                    (1.0, "waiting_release_edge"),
                ):
                    add(round(begin + fraction * (end - begin)), tag)
            begin = None

    # Use a +/-2-step neighborhood, but cap density deterministically.
    expanded: dict[int, set[str]] = {}
    for step, tags in events.items():
        for offset in (-2, 0, 2):
            target = step + offset
            if 0 <= target < len(actions):
                expanded.setdefault(target, set()).update(tags)
    ordered = sorted(expanded.items())
    if len(ordered) > 36:
        keep = np.linspace(0, len(ordered) - 1, 36, dtype=int)
        ordered = [ordered[index] for index in keep]
    return [
        {"step": step, "events": sorted(tags), "phase": str(labels[step])}
        for step, tags in ordered
    ]


def goal_anchors(episode, config: Config) -> list[dict[str, Any]]:
    positions = np.asarray(episode.positions, dtype=np.float64)
    goals = np.asarray(episode.metadata["environment"]["goals"], dtype=np.float64)
    errors = np.linalg.norm(goals[None] - positions, axis=-1)
    candidates: dict[int, set[str]] = {}

    def add(step: int, tag: str) -> None:
        if 0 <= step < len(episode.actions):
            candidates.setdefault(step, set()).add(tag)

    for agent in range(4):
        for radius in (0.60, 0.35, 0.20, 0.12, 0.085):
            step = int(np.argmin(np.abs(errors[:, agent] - radius)))
            add(step, f"agent_{agent}_goal_radius_{radius:.3f}")
    for remaining in (60, 40, 25, 15, 8, 3):
        add(len(episode.actions) - remaining, f"termination_minus_{remaining}")
    ordered = sorted(candidates.items())
    if len(ordered) > 18:
        keep = np.linspace(0, len(ordered) - 1, 18, dtype=int)
        ordered = [ordered[index] for index in keep]
    labels = phase_labels(episode, config)
    return [
        {"step": step, "events": sorted(tags), "phase": str(labels[step])}
        for step, tags in ordered
    ]


@dataclass(frozen=True)
class RecoveryResult:
    success: bool
    terminal_reason: str
    wall_collision: bool
    agent_collision: bool
    deadlock: bool
    timeout: bool
    actions: np.ndarray
    positions: np.ndarray
    observations: np.ndarray
    recovery_steps: int
    min_wall_clearance: float
    min_pair_clearance: float
    mode_signature: str
    source_mode_signature: str
    mode_signature_match: bool


def query_reference_recovery(
    episode,
    source_step: int,
    positions: np.ndarray,
    last_velocity: np.ndarray,
    tracking_gain: float = 3.0,
) -> RecoveryResult:
    """Validate a local centralized recovery query against the actual plant."""
    config = Config(**episode.metadata["environment"]["config"])
    env = DoubleBottleneckEnv(config)
    env.reset(np.asarray(positions, dtype=np.float64), regime=episode.regime)
    state = env.augmented_state()
    state["last_applied_velocity"] = radial_bound(last_velocity, config.max_speed)
    env.restore_augmented_state(state)
    reference_positions = np.asarray(episode.positions[source_step:], dtype=np.float64)
    reference_actions = np.asarray(episode.actions[source_step:], dtype=np.float64)
    actions = []
    recovered_positions = [env.positions.copy()]
    observations = [env.observation().copy()]
    min_wall = math.inf
    min_pair = math.inf
    terminal = "running"
    final_info = None
    for local_step in range(config.max_steps):
        if local_step < len(reference_actions):
            reference_index = min(local_step, len(reference_positions) - 1)
            action = reference_actions[local_step] + tracking_gain * (
                reference_positions[reference_index] - env.positions
            )
        else:
            action = 2.0 * (env.goals - env.positions)
        action = radial_bound(action, config.max_speed)
        observation, _, done, info = env.step(action)
        actions.append(action.copy())
        recovered_positions.append(env.positions.copy())
        observations.append(observation.copy())
        min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
        min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        terminal = str(info["termination"])
        final_info = info
        if done:
            break
    if final_info is None:
        raise AssertionError("empty recovery rollout")
    summary = env.summary()
    stitched = np.concatenate(
        (
            np.asarray(episode.positions[:source_step], dtype=np.float64),
            np.asarray(recovered_positions, dtype=np.float64),
        ),
        axis=0,
    )
    mode = infer_coordination_mode(stitched, env.goals, config)
    source_mode = str(episode.metadata["coordination_mode"]["signature"])
    signature = str(mode["signature"])
    return RecoveryResult(
        success=bool(summary["collision_free_success"]),
        terminal_reason=terminal,
        wall_collision=bool(summary["wall_collision"]),
        agent_collision=bool(summary["agent_collision"]),
        deadlock=bool(summary["deadlock"]),
        timeout=terminal == "timeout",
        actions=np.asarray(actions, dtype=np.float64),
        positions=np.asarray(recovered_positions, dtype=np.float64),
        observations=np.asarray(observations, dtype=np.float32),
        recovery_steps=len(actions),
        min_wall_clearance=float(min_wall),
        min_pair_clearance=float(min_pair),
        mode_signature=signature,
        source_mode_signature=source_mode,
        mode_signature_match=signature == source_mode,
    )


__all__ = (
    "PHASES",
    "RecoveryResult",
    "goal_anchors",
    "observation_from_state",
    "phase_labels",
    "query_reference_recovery",
    "radial_bound",
    "transition_anchors",
    "transition_mask",
)

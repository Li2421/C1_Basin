"""Centralized deterministic expert for the four-agent Double-Bottleneck.

The expert is a small space-time planner, not a reactive right-of-way rule.
It enumerates coordination hypotheses (which directional wave crosses first and
the order inside each same-direction wave), builds collision-free scheduled
paths, and replays every candidate through :class:`DoubleBottleneckEnv` before
returning it.  A caller can request a particular hypothesis to obtain alternate
valid coordination modes for the same initial condition.

The construction is deliberately specialized to the fixed two-bottleneck
topology.  It uses the outer rooms as waiting areas, sends two agents travelling
in the same direction as a separated convoy, and only releases the opposing
convoy after the first has reached its goals.  No priority is installed in the
environment and no agent has a privileged identity: all eight directional and
within-wave order hypotheses are considered by :meth:`CentralizedExpert.plan`.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .environment import DoubleBottleneckEnv
from .scenario import LEFT_TO_RIGHT, RIGHT_TO_LEFT


@dataclass(frozen=True)
class CoordinationHypothesis:
    """One discrete initialization of the centralized space-time planner.

    ``first_direction`` is ``"left_to_right"`` or ``"right_to_left"``.
    Each order is a permutation of the two agent indices travelling in that
    direction; the first entry is the front member of its convoy.
    """

    first_direction: str
    left_to_right_order: tuple[int, int] = LEFT_TO_RIGHT
    right_to_left_order: tuple[int, int] = RIGHT_TO_LEFT

    def __post_init__(self) -> None:
        if self.first_direction not in ("left_to_right", "right_to_left"):
            raise ValueError(
                "first_direction must be 'left_to_right' or 'right_to_left'"
            )
        if set(self.left_to_right_order) != set(LEFT_TO_RIGHT):
            raise ValueError("left_to_right_order must permute agents (0, 1)")
        if set(self.right_to_left_order) != set(RIGHT_TO_LEFT):
            raise ValueError("right_to_left_order must permute agents (2, 3)")

    @property
    def first_order(self) -> tuple[int, int]:
        return (
            self.left_to_right_order
            if self.first_direction == "left_to_right"
            else self.right_to_left_order
        )

    @property
    def second_order(self) -> tuple[int, int]:
        return (
            self.right_to_left_order
            if self.first_direction == "left_to_right"
            else self.left_to_right_order
        )

    @property
    def label(self) -> str:
        first = "+" if self.first_direction == "left_to_right" else "-"
        return (
            f"first={first};ltr={self.left_to_right_order[0]}-{self.left_to_right_order[1]};"
            f"rtl={self.right_to_left_order[0]}-{self.right_to_left_order[1]}"
        )


@dataclass(frozen=True)
class ExpertPlan:
    """Validated expert rollout, ready for dataset serialization.

    ``positions[k]`` is the state before ``actions[k]`` and therefore
    ``len(positions) == len(actions) + 1``.  The arrays include only the
    executed prefix up to the environment's terminal event.
    """

    hypothesis: CoordinationHypothesis
    positions: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    terminal_reason: str
    success: bool
    collision: bool
    timeout: bool
    deadlock: bool
    episode_steps: int
    solve_time_seconds: float
    path_length: float
    min_inter_agent_surface_distance: float
    min_wall_clearance: float
    candidate_count: int = 1

    @property
    def metadata(self) -> dict:
        return {
            "expert": "centralized_scheduled_convoy_v1",
            "coordination_mode": self.hypothesis.label,
            "first_direction": self.hypothesis.first_direction,
            "left_to_right_order": list(self.hypothesis.left_to_right_order),
            "right_to_left_order": list(self.hypothesis.right_to_left_order),
            "candidate_count": self.candidate_count,
            "solve_time_seconds": self.solve_time_seconds,
            "episode_steps": self.episode_steps,
            "path_length": self.path_length,
            "min_inter_agent_surface_distance": self.min_inter_agent_surface_distance,
            "min_wall_clearance": self.min_wall_clearance,
        }


class _TimedPath:
    """Piecewise-linear point trajectory with explicit waiting intervals."""

    def __init__(self, start: np.ndarray):
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.position = self.start.copy()
        self.time = 0.0
        self._segments: list[tuple[float, float, np.ndarray, np.ndarray]] = []

    def move_at(
        self, start_time: float, points: Sequence[np.ndarray], speed: float
    ) -> list[float]:
        if speed <= 0 or not math.isfinite(speed):
            raise ValueError("speed must be positive and finite")
        if start_time < self.time - 1e-12:
            raise ValueError(
                "cannot schedule a move before the preceding move finishes"
            )
        self.time = max(self.time, float(start_time))
        arrival_times = []
        for point in points:
            target = np.asarray(point, dtype=np.float64)
            distance = float(np.linalg.norm(target - self.position))
            if distance > 1e-12:
                end = self.time + distance / speed
                self._segments.append(
                    (self.time, end, self.position.copy(), target.copy())
                )
                self.time = end
                self.position = target.copy()
            arrival_times.append(self.time)
        return arrival_times

    def at(self, query_time: float) -> np.ndarray:
        value = self.start
        for begin, end, start, target in self._segments:
            if query_time < begin:
                break
            if query_time <= end:
                alpha = min(max((query_time - begin) / (end - begin), 0.0), 1.0)
                return (1.0 - alpha) * start + alpha * target
            value = target
        return np.asarray(value, dtype=np.float64).copy()


def all_coordination_hypotheses() -> tuple[CoordinationHypothesis, ...]:
    """Return all eight non-privileged direction/order planning seeds."""

    result = []
    for first_direction in ("left_to_right", "right_to_left"):
        for ltr in itertools.permutations(LEFT_TO_RIGHT):
            for rtl in itertools.permutations(RIGHT_TO_LEFT):
                result.append(CoordinationHypothesis(first_direction, ltr, rtl))
    return tuple(result)


class CentralizedExpert:
    """Validated model-based joint expert producing actions of shape ``[4, 2]``."""

    def __init__(
        self,
        route_speed: float = 0.45,
        reposition_speed: float = 0.40,
        convoy_spacing: float = 0.62,
        phase_padding: float = 0.15,
    ):
        self.route_speed = float(route_speed)
        self.reposition_speed = float(reposition_speed)
        self.convoy_spacing = float(convoy_spacing)
        self.phase_padding = float(phase_padding)
        if not all(
            math.isfinite(value) and value > 0
            for value in (route_speed, reposition_speed, convoy_spacing, phase_padding)
        ):
            raise ValueError("planner constants must be positive and finite")

    def plan(
        self,
        env: DoubleBottleneckEnv,
        hypotheses: Iterable[CoordinationHypothesis] | None = None,
        require_success: bool = True,
    ) -> ExpertPlan:
        """Plan and validate candidate joint trajectories from ``env``.

        By default all eight hypotheses are evaluated and the shortest
        successful rollout is returned.  Pass an explicit iterable (often one
        element) to deliberately sample alternate coordination modes.
        """

        if env.step_count != 0:
            raise ValueError(
                "expert planning currently requires a freshly reset environment"
            )
        if env.done:
            raise ValueError("cannot plan from a terminated environment")
        seeds = tuple(
            all_coordination_hypotheses() if hypotheses is None else hypotheses
        )
        if not seeds:
            raise ValueError("at least one coordination hypothesis is required")
        started = time.perf_counter()
        plans = [self._plan_one(env, seed) for seed in seeds]
        elapsed = time.perf_counter() - started
        successful = [plan for plan in plans if plan.success]
        if require_success and not successful:
            details = ", ".join(
                f"{p.hypothesis.label}:{p.terminal_reason}" for p in plans
            )
            raise RuntimeError(
                f"centralized expert found no successful hypothesis ({details})"
            )
        pool = successful or plans
        chosen = min(
            pool,
            key=lambda plan: (
                plan.episode_steps,
                plan.path_length,
                -plan.min_inter_agent_surface_distance,
                plan.hypothesis.label,
            ),
        )
        return ExpertPlan(
            **{
                **chosen.__dict__,
                "solve_time_seconds": elapsed,
                "candidate_count": len(seeds),
            }
        )

    def plan_hypothesis(
        self, env: DoubleBottleneckEnv, hypothesis: CoordinationHypothesis
    ) -> ExpertPlan:
        """Convenience interface for dataset mode probing with one seed."""

        return self.plan(env, (hypothesis,))

    def _plan_one(
        self, env: DoubleBottleneckEnv, hypothesis: CoordinationHypothesis
    ) -> ExpertPlan:
        config = env.config
        if self.route_speed > config.max_speed + 1e-12:
            raise ValueError("route_speed exceeds the environment speed bound")
        if self.reposition_speed > config.max_speed + 1e-12:
            raise ValueError("reposition_speed exceeds the environment speed bound")

        paths = [_TimedPath(point) for point in env.positions]
        first_sign = 1 if hypothesis.first_direction == "left_to_right" else -1
        first_order = hypothesis.first_order
        second_order = hypothesis.second_order

        outer_gate = config.chamber_half_length + config.bottleneck_length
        # The front slot is outside the bottleneck, and the rear slot provides
        # more than one diameter of along-route separation.
        front_stage = outer_gate + 0.25
        rear_stage = front_stage + self.convoy_spacing
        park_x = outer_gate + 0.25
        safe_outer_y = (
            config.outer_half_height
            - config.agent_radius
            - config.wall_radius
            - config.wall_collision_margin
            - 0.025
        )
        park_y = min(0.60, safe_outer_y)
        if park_y <= 2 * config.agent_radius:
            raise ValueError(
                "outer staging area is too narrow for the expert waiting layout"
            )

        # Park the second wave away from both the centre-line and the goal
        # approach paths while the first wave is being assembled.
        second_side = -first_sign
        for agent in second_order:
            vertical_sign = 1.0 if env.positions[agent, 1] >= 0 else -1.0
            target = np.array((-second_side * park_x, vertical_sign * park_y))
            # second_side is its travel direction; -second_side is its origin side.
            paths[agent].move_at(0.0, (target,), self.reposition_speed)

        # Assemble the first convoy sequentially.  Sequential assembly makes
        # both within-wave permutations valid even when straight staging paths
        # would cross for the reversed order.
        staging_end = 0.0
        origin_side = -first_sign
        for slot, agent in enumerate(first_order):
            magnitude = front_stage if slot == 0 else rear_stage
            target = np.array((origin_side * magnitude, 0.0))
            arrival = paths[agent].move_at(
                staging_end, (target,), self.reposition_speed
            )[-1]
            staging_end = arrival + self.phase_padding
        first_start = (
            max(staging_end, *(path.time for path in paths)) + self.phase_padding
        )

        first_goal_times = self._schedule_wave(
            paths, env.goals, first_order, first_sign, first_start, outer_gate
        )

        # Only after the first wave is completely parked at its goals do we
        # assemble the second wave.  This conservative release gives clean,
        # reproducible clearance around the outer-room merge.
        second_staging_end = max(first_goal_times) + self.phase_padding
        second_origin_side = first_sign
        for slot, agent in enumerate(second_order):
            magnitude = front_stage if slot == 0 else rear_stage
            target = np.array((second_origin_side * magnitude, 0.0))
            arrival = paths[agent].move_at(
                second_staging_end, (target,), self.reposition_speed
            )[-1]
            second_staging_end = arrival + self.phase_padding
        second_start = second_staging_end + self.phase_padding
        self._schedule_wave(
            paths, env.goals, second_order, -first_sign, second_start, outer_gate
        )

        final_time = max(path.time for path in paths)
        reference_steps = min(config.max_steps, math.ceil(final_time / config.dt) + 2)
        reference_times = np.arange(reference_steps + 1, dtype=np.float64) * config.dt
        reference = np.stack(
            [[path.at(t) for path in paths] for t in reference_times], axis=0
        )
        actions = np.diff(reference, axis=0) / config.dt
        speed = np.linalg.norm(actions, axis=-1)
        if speed.max(initial=0.0) > config.max_speed + 1e-9:
            raise AssertionError("internal schedule violated the speed bound")
        return self._validate(env, hypothesis, actions)

    def _schedule_wave(
        self,
        paths: list[_TimedPath],
        goals: np.ndarray,
        order: tuple[int, int],
        direction: int,
        start_time: float,
        outer_gate: float,
    ) -> tuple[float, float]:
        """Schedule a same-direction convoy and return its two goal times."""

        # Only x monotonicity and y=0 through both gates matter.  Points just
        # inside each outer opening retain clearance from the step corners.
        branch = outer_gate + 0.60
        center_points = (
            np.array((-direction * (outer_gate - 0.10), 0.0)),
            np.array((-direction * 0.90, 0.0)),
            np.array((direction * 0.90, 0.0)),
            np.array((direction * (outer_gate - 0.10), 0.0)),
            np.array((direction * (outer_gate + 0.25), 0.0)),
        )
        arrival = []
        for agent in order:
            goal = goals[agent]
            branch_point = np.array((direction * branch, goal[1]))
            times = paths[agent].move_at(
                start_time, (*center_points, branch_point, goal), self.route_speed
            )
            arrival.append(times[-1])
        return tuple(arrival)

    @staticmethod
    def _validate(
        source_env: DoubleBottleneckEnv,
        hypothesis: CoordinationHypothesis,
        scheduled_actions: np.ndarray,
    ) -> ExpertPlan:
        validation = DoubleBottleneckEnv(source_env.config)
        # Preserve the complete initial Markov state, including the last
        # applied velocity exposed in observations.  The plant is first order,
        # so this velocity has no inertial effect once the first expert action
        # is applied, but silently zeroing it would corrupt dataset states.
        validation.restore_augmented_state(source_env.augmented_state())
        positions = [validation.positions.copy()]
        observations = [validation.observation().copy()]
        actions = []
        terminal = "timeout"
        min_pair = math.inf
        min_wall = math.inf
        info = None
        for action in scheduled_actions:
            observation, _, done, info = validation.step(action)
            actions.append(np.asarray(action, dtype=np.float64).copy())
            positions.append(validation.positions.copy())
            observations.append(observation.copy())
            min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
            min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
            terminal = str(info["termination"])
            if done:
                break
        if info is None:
            raise AssertionError("expert schedule contained no action")
        summary = validation.summary()
        executed = np.asarray(actions, dtype=np.float64)
        path_length = float(
            np.linalg.norm(executed, axis=-1).sum() * validation.config.dt
        )
        return ExpertPlan(
            hypothesis=hypothesis,
            positions=np.asarray(positions, dtype=np.float64),
            actions=executed,
            observations=np.asarray(observations, dtype=np.float32),
            terminal_reason=terminal,
            success=bool(summary["collision_free_success"]),
            collision=bool(summary["wall_collision"] or summary["agent_collision"]),
            timeout=terminal == "timeout",
            deadlock=bool(summary["deadlock"]),
            episode_steps=int(summary["episode_steps"]),
            solve_time_seconds=0.0,
            path_length=path_length,
            min_inter_agent_surface_distance=min_pair,
            min_wall_clearance=min_wall,
        )


def rollout_expert(
    env: DoubleBottleneckEnv,
    hypothesis: CoordinationHypothesis | None = None,
    expert: CentralizedExpert | None = None,
) -> ExpertPlan:
    """One-call interface used by pilot diagnostics and dataset generation."""

    planner = expert or CentralizedExpert()
    return planner.plan(env, None if hypothesis is None else (hypothesis,))


__all__ = (
    "CentralizedExpert",
    "CoordinationHypothesis",
    "ExpertPlan",
    "all_coordination_hypotheses",
    "rollout_expert",
)

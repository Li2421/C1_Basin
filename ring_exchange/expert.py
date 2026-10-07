"""Centralized joint expert for the Ring Exchange benchmark.

The planner explicitly searches clockwise and counter-clockwise circulation
hypotheses, realizes each as joint spatiotemporal paths in the annulus, and
replays every candidate in the physical environment.  Direction is solely an
internal expert search hypothesis: it is never installed in the environment or
returned as a policy feature.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from .environment import RingExchangeEnv, point_segment_distance


@dataclass(frozen=True)
class CirculationHypothesis:
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in ("cw", "ccw"):
            raise ValueError("direction must be 'cw' or 'ccw'")

    @property
    def sign(self) -> int:
        return -1 if self.direction == "cw" else 1

    @property
    def label(self) -> str:
        return self.direction


def all_circulation_hypotheses() -> tuple[CirculationHypothesis, CirculationHypothesis]:
    return (CirculationHypothesis("cw"), CirculationHypothesis("ccw"))


@dataclass(frozen=True)
class ExpertPlan:
    hypothesis: CirculationHypothesis
    positions: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    terminal_reason: str
    success: bool
    collision: bool
    timeout: bool
    episode_steps: int
    solve_time_seconds: float
    path_length: float
    min_inter_agent_surface_distance: float
    min_obstacle_clearance: float
    min_outer_clearance: float
    candidate_count: int = 1

    @property
    def mode_signature(self) -> str:
        return self.hypothesis.direction

    @property
    def metadata(self) -> dict:
        return {"expert": "centralized_annular_spatiotemporal_search_v1",
                "circulation_mode": self.hypothesis.direction,
                "candidate_count": self.candidate_count,
                "solve_time_seconds": self.solve_time_seconds,
                "episode_steps": self.episode_steps,
                "path_length": self.path_length,
                "min_inter_agent_surface_distance": self.min_inter_agent_surface_distance,
                "min_obstacle_clearance": self.min_obstacle_clearance,
                "min_outer_clearance": self.min_outer_clearance}


class _TimedPath:
    def __init__(self, start: np.ndarray):
        self.position = np.asarray(start, dtype=np.float64).copy()
        self.start = self.position.copy()
        self.time = 0.0
        self.segments: list[tuple[float, float, np.ndarray, np.ndarray]] = []

    def move(self, points: list[np.ndarray], speed: float) -> None:
        for point in points:
            point = np.asarray(point, dtype=np.float64)
            distance = float(np.linalg.norm(point - self.position))
            if distance > 1e-12:
                end = self.time + distance / speed
                self.segments.append((self.time, end, self.position.copy(), point.copy()))
                self.time, self.position = end, point.copy()

    def at(self, query: float) -> np.ndarray:
        value = self.start
        for begin, end, start, target in self.segments:
            if query < begin:
                break
            if query <= end:
                alpha = (query - begin) / max(end - begin, 1e-30)
                return (1 - alpha) * start + alpha * target
            value = target
        return value.copy()


def _angle(point: np.ndarray) -> float:
    return float(math.atan2(point[1], point[0]))


def _directed_delta(start: float, goal: float, sign: int) -> float:
    """Shortest positive/negative angular delta in the requested direction."""
    if sign > 0:
        return (goal - start) % (2 * math.pi)
    return -((start - goal) % (2 * math.pi))


def infer_circulation_hypothesis(positions: np.ndarray, velocities: np.ndarray, *,
                                 min_tangential_speed: float = 0.08,
                                 min_coherent_agents: int = 2) -> CirculationHypothesis | None:
    """Infer CW/CCW from the physical tangent velocity, or return ``None``.

    This helper is for centralized *recovery re-query* only.  It is never
    serialized as a policy feature.  It prevents a recovery expert from
    relabelling an already circulating state with an immediate reverse turn.
    """
    p, v = np.asarray(positions, dtype=np.float64), np.asarray(velocities, dtype=np.float64)
    if p.shape != (4, 2) or v.shape != (4, 2):
        raise ValueError("positions and velocities must have shape [4,2]")
    radii = np.maximum(np.linalg.norm(p, axis=-1), 1e-12)
    tangential_speed = (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]) / radii
    active = np.abs(tangential_speed) >= min_tangential_speed
    if int(active.sum()) < min_coherent_agents:
        return None
    signs = np.sign(tangential_speed[active])
    majority = 1.0 if signs.sum() > 0 else -1.0
    if float(np.mean(signs == majority)) < 0.75:
        return None
    return CirculationHypothesis("ccw" if majority > 0 else "cw")


class CentralizedExpert:
    """Joint four-agent circulation planner, with validated CW/CCW alternatives."""
    def __init__(self, route_radius: float = 1.85, route_speed: float = 0.46,
                 arc_increment: float = 0.10, direct_goal_distance: float = 1.10):
        self.route_radius = float(route_radius)
        self.route_speed = float(route_speed)
        self.arc_increment = float(arc_increment)
        self.direct_goal_distance = float(direct_goal_distance)
        if not all(math.isfinite(x) and x > 0 for x in (self.route_radius, self.route_speed, self.arc_increment, self.direct_goal_distance)):
            raise ValueError("planner constants must be positive and finite")

    def plan(self, env: RingExchangeEnv, hypotheses: Iterable[CirculationHypothesis] | None = None,
             require_success: bool = True) -> ExpertPlan:
        if env.done:
            raise ValueError("cannot plan from terminated environment")
        seeds = tuple(all_circulation_hypotheses() if hypotheses is None else hypotheses)
        if not seeds:
            raise ValueError("at least one circulation hypothesis is required")
        started = time.perf_counter()
        candidates = [self._plan_one(env, hypothesis) for hypothesis in seeds]
        elapsed = time.perf_counter() - started
        successful = [candidate for candidate in candidates if candidate.success]
        if require_success and not successful:
            raise RuntimeError("centralized ring expert found no collision-free circulation candidate")
        # Equal-cost direction ties are resolved deterministically *in the
        # planner only*.  Datasets can deliberately request either hypothesis.
        pool = successful or candidates
        selected = min(pool, key=lambda p: (p.episode_steps, p.path_length, p.hypothesis.direction))
        return ExpertPlan(**{**selected.__dict__, "candidate_count": len(seeds), "solve_time_seconds": elapsed})

    def plan_hypothesis(self, env: RingExchangeEnv, hypothesis: CirculationHypothesis) -> ExpertPlan:
        return self.plan(env, (hypothesis,))

    def _plan_one(self, source: RingExchangeEnv, hypothesis: CirculationHypothesis) -> ExpertPlan:
        cfg = source.config
        if self.route_speed > cfg.max_speed + 1e-12:
            raise ValueError("route_speed exceeds environment speed")
        clearance = self.route_radius - cfg.obstacle_radius - cfg.agent_radius
        outer_clearance = cfg.outer_radius - cfg.agent_radius - self.route_radius
        if clearance <= cfg.collision_margin or outer_clearance <= cfg.collision_margin:
            raise ValueError("route circle must lie safely in annulus")
        paths = [_TimedPath(start) for start in source.positions]
        for i, path in enumerate(paths):
            theta0, theta1 = _angle(source.positions[i]), _angle(source.goals[i])
            goal_error = float(np.linalg.norm(source.goals[i] - source.positions[i]))
            if goal_error <= cfg.goal_tolerance:
                # A joint episode may still be running while this agent has
                # completed.  A recovery expert must hold it at goal rather
                # than send it back through the annulus.
                continue
            if goal_error <= self.direct_goal_distance and self._direct_goal_segment_safe(source.positions[i], source.goals[i], cfg):
                # Near goal, choose by true line-segment feasibility, not a
                # brittle angular threshold.  This covers a small tangential
                # overshoot: even if current velocity still implies CW/CCW,
                # recovery must converge directly rather than relaunch a near
                # full circulation around the disk.
                path.move([source.goals[i]], self.route_speed)
                continue
            delta = _directed_delta(theta0, theta1, hypothesis.sign)
            # A polyline approximation to a radial-in / arc / radial-out route.
            count = max(2, int(math.ceil(abs(delta) / self.arc_increment)))
            arc = [self.route_radius * np.array((math.cos(theta0 + delta * k / count),
                                                 math.sin(theta0 + delta * k / count)))
                   for k in range(1, count + 1)]
            goal_radius = float(np.linalg.norm(source.goals[i]))
            exit_point = goal_radius * np.array((math.cos(theta1), math.sin(theta1)))
            entry = self.route_radius * np.array((math.cos(theta0), math.sin(theta0)))
            path.move([entry, *arc, exit_point, source.goals[i]], self.route_speed)
        final_time = max(path.time for path in paths)
        steps = min(cfg.max_steps, int(math.ceil(final_time / cfg.dt)) + 2)
        times = np.arange(steps + 1) * cfg.dt
        reference = np.stack([[path.at(t) for path in paths] for t in times])
        actions = np.diff(reference, axis=0) / cfg.dt
        if np.linalg.norm(actions, axis=-1).max(initial=0.0) > cfg.max_speed + 1e-8:
            raise AssertionError("internal route violated speed bound")
        return self._validate(source, hypothesis, actions)

    @staticmethod
    def _direct_goal_segment_safe(start: np.ndarray, goal: np.ndarray, cfg) -> bool:
        """Exact annular geometric feasibility for a short point-to-goal chord."""
        start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        inner = float(point_segment_distance(np.zeros(2), start, goal))
        outer = max(float(np.linalg.norm(start)), float(np.linalg.norm(goal)))
        return bool(inner > cfg.obstacle_radius + cfg.agent_radius + cfg.collision_margin and
                    outer < cfg.outer_radius - cfg.agent_radius - cfg.collision_margin)

    @staticmethod
    def _validate(source: RingExchangeEnv, hypothesis: CirculationHypothesis, scheduled_actions: np.ndarray) -> ExpertPlan:
        validation = RingExchangeEnv(source.config)
        validation.restore_augmented_state(source.augmented_state())
        positions, observations, actions = [validation.positions.copy()], [validation.observation().copy()], []
        min_pair = min_obstacle = min_outer = math.inf
        info = None
        for action in scheduled_actions:
            observation, _, done, info = validation.step(action)
            positions.append(validation.positions.copy())
            observations.append(observation.copy())
            actions.append(np.asarray(action, dtype=np.float64).copy())
            min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
            min_obstacle = min(min_obstacle, float(np.min(info["swept_obstacle_distance"])))
            min_outer = min(min_outer, float(np.min(info["swept_outer_distance"])))
            if done:
                break
        if info is None:
            raise AssertionError("expert produced no actions")
        result = validation.summary()
        actions_array = np.asarray(actions, dtype=np.float64)
        return ExpertPlan(hypothesis=hypothesis, positions=np.asarray(positions), actions=actions_array,
                          observations=np.asarray(observations, dtype=np.float32), terminal_reason=str(info["termination"]),
                          success=bool(result["collision_free_success"]),
                          collision=bool(result["obstacle_collision"] or result["outer_collision"] or result["agent_collision"]),
                          timeout=str(info["termination"]) == "timeout", episode_steps=int(result["episode_steps"]),
                          solve_time_seconds=0.0, path_length=float(np.linalg.norm(actions_array, axis=-1).sum() * validation.config.dt),
                          min_inter_agent_surface_distance=min_pair, min_obstacle_clearance=min_obstacle,
                          min_outer_clearance=min_outer)


def circulation_signature(positions: np.ndarray) -> str:
    """Analysis-only realized trajectory mode: CW, CCW, mixed, or stationary."""
    data = np.asarray(positions, dtype=np.float64)
    if data.ndim != 3 or data.shape[1:] != (4, 2):
        raise ValueError("positions must be [T,4,2]")
    angles = np.unwrap(np.arctan2(data[..., 1], data[..., 0]), axis=0)
    displacement = angles[-1] - angles[0]
    positive, negative = (displacement > 0.15).sum(), (displacement < -0.15).sum()
    if positive and negative:
        return "mixed"
    if positive:
        return "ccw"
    if negative:
        return "cw"
    return "stationary"


def rollout_expert(env: RingExchangeEnv, hypothesis: CirculationHypothesis | None = None,
                   expert: CentralizedExpert | None = None) -> ExpertPlan:
    return (expert or CentralizedExpert()).plan(env, None if hypothesis is None else (hypothesis,))


__all__ = ("CirculationHypothesis", "ExpertPlan", "CentralizedExpert", "all_circulation_hypotheses",
           "circulation_signature", "infer_circulation_hypothesis", "rollout_expert")

"""Eight-agent coupled dual-intersection exchange plant.

The free space is the union of one horizontal corridor and two vertical
corridors.  The intersections are deliberately wider waiting/conflict regions
than the bridge between them.  The plant mirrors the current four-agent
single-integrator semantics: it observes swept collisions but never repairs
the supplied action itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
import math
from typing import Any

import numpy as np


AGENT_NAMES = tuple(f"A{i}" for i in range(1, 9))
NUM_AGENTS = 8
PAIR_INDICES = tuple(combinations(range(NUM_AGENTS), 2))


@dataclass(frozen=True)
class StressConfig:
    """Fixed physical and monitor contract for the stress-test plant.

    ``max_steps`` is longer than Double-Bottleneck because an oracle is allowed
    to clear agents sequentially; the velocity, radius, time step, CBF gains,
    and monitor thresholds retain the existing scenario's scale.
    """

    schema: str = "eta_coupled_dual_intersection_exchange_v1"
    dt: float = 0.05
    max_steps: int = 3000
    max_speed: float = 0.5
    agent_radius: float = 0.12
    wall_radius: float = 4 / 600
    wall_collision_margin: float = 0.005
    agent_collision_margin: float = 0.0
    goal_tolerance: float = 0.08
    horizontal_half_length: float = 4.6
    horizontal_corridor_half_length: float = 2.7
    vertical_half_length: float = 3.6
    corridor_half_width: float = 0.25
    staging_half_height: float = 1.0
    junction_half_width: float = 0.34
    junction_x: float = 1.5
    progress_window_seconds: float = 2.0
    deadlock_hold_seconds: float = 5.0
    progress_epsilon: float = 0.01
    speed_epsilon_fraction: float = 0.05
    terminate_on_collision: bool = True
    terminate_on_success: bool = True
    terminate_on_deadlock: bool = True

    def __post_init__(self) -> None:
        if self.schema != "eta_coupled_dual_intersection_exchange_v1":
            raise ValueError("unknown stress-test schema")
        values = (
            self.dt, self.max_steps, self.max_speed, self.agent_radius,
            self.wall_radius, self.wall_collision_margin, self.goal_tolerance,
            self.horizontal_half_length, self.horizontal_corridor_half_length,
            self.vertical_half_length, self.corridor_half_width,
            self.staging_half_height, self.junction_half_width,
            self.junction_x, self.progress_window_seconds,
            self.deadlock_hold_seconds, self.progress_epsilon,
            self.speed_epsilon_fraction,
        )
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in values):
            raise ValueError("all physical and monitor constants must be positive finite")
        if self.junction_half_width < self.corridor_half_width:
            raise ValueError("junctions must not be narrower than their approaches")
        if not 0 < self.horizontal_corridor_half_length < self.horizontal_half_length:
            raise ValueError("corridor must lie strictly between the two staging rooms")
        if 2 * (self.agent_radius + self.wall_radius + self.wall_collision_margin) >= 2 * self.corridor_half_width:
            raise ValueError("corridor cannot fit a collision-free agent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    delta = b - a
    fraction = np.clip(
        np.sum((p - a) * delta, axis=-1) / np.maximum(np.sum(delta * delta, axis=-1), 1e-30),
        0.0,
        1.0,
    )
    return np.linalg.norm(p - (a + fraction[..., None] * delta), axis=-1)


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Exact 2-D segment distance, vectorized over leading dimensions."""
    a, b, c, d = map(np.asarray, (a, b, c, d))
    r, s, q = b - a, d - c, c - a

    def cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    denominator = cross(r, s)
    safe = np.where(np.abs(denominator) > 1e-15, denominator, 1.0)
    first, second = cross(q, s) / safe, cross(q, r) / safe
    intersects = (
        (np.abs(denominator) > 1e-15)
        & (first >= 0) & (first <= 1) & (second >= 0) & (second <= 1)
    )
    distance = np.minimum.reduce((
        point_segment_distance(a, c, d), point_segment_distance(b, c, d),
        point_segment_distance(c, a, b), point_segment_distance(d, a, b),
    ))
    return np.where(intersects, 0.0, distance)


def _union_boundary(rectangles: tuple[tuple[float, float, float, float], ...]) -> np.ndarray:
    """Return exposed segments of an axis-aligned union of rectangles.

    A small rectilinear cell decomposition avoids hand-maintained interior wall
    segments at the two junction overlaps.
    """
    xs = np.asarray(sorted({value for x0, x1, _, _ in rectangles for value in (x0, x1)}))
    ys = np.asarray(sorted({value for _, _, y0, y1 in rectangles for value in (y0, y1)}))
    filled = np.zeros((len(xs) - 1, len(ys) - 1), dtype=bool)
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            x, y = (xs[ix] + xs[ix + 1]) / 2, (ys[iy] + ys[iy + 1]) / 2
            filled[ix, iy] = any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in rectangles)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for ix in range(filled.shape[0]):
        for iy in range(filled.shape[1]):
            if not filled[ix, iy]:
                continue
            x0, x1, y0, y1 = xs[ix], xs[ix + 1], ys[iy], ys[iy + 1]
            if ix == 0 or not filled[ix - 1, iy]:
                segments.append(((x0, y0), (x0, y1)))
            if ix == filled.shape[0] - 1 or not filled[ix + 1, iy]:
                segments.append(((x1, y0), (x1, y1)))
            if iy == 0 or not filled[ix, iy - 1]:
                segments.append(((x0, y0), (x1, y0)))
            if iy == filled.shape[1] - 1 or not filled[ix, iy + 1]:
                segments.append(((x0, y1), (x1, y1)))
    return np.asarray(segments, dtype=np.float64)


def build_layout(config: StressConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build walls and fixed goal disks for the coupled exchange map."""
    h, hc, v, c, j, staging = (
        config.horizontal_half_length, config.horizontal_corridor_half_length,
        config.vertical_half_length, config.corridor_half_width,
        config.junction_half_width, config.staging_half_height,
    )
    rectangles = (
        (-hc, hc, -c, c),
        (-h, -hc, -staging, staging),
        (hc, h, -staging, staging),
        (-config.junction_x - j, -config.junction_x + j, -v, v),
        (config.junction_x - j, config.junction_x + j, -v, v),
    )
    walls = _union_boundary(rectangles)
    goals = np.asarray((
        (4.20, -0.75), (3.80, 0.75),
        (-4.20, 0.35), (-3.80, -0.35),
        (-1.68, -3.05), (-1.32, 3.05),
        (1.68, -3.05), (1.32, 3.05),
    ), dtype=np.float64)
    return walls, goals


def default_positions() -> np.ndarray:
    """Collision-free queue starts; benchmark generation perturbs these only."""
    return np.asarray((
        (-2.90, -0.75), (-3.30, 0.75),
        (2.90, 0.35), (3.30, -0.35),
        (-1.68, 0.90), (-1.32, -0.90),
        (1.68, 0.90), (1.32, -0.90),
    ), dtype=np.float64)


class CoupledDualIntersectionEnv:
    """Eight-agent deterministic plant with the repository's first-event monitor."""

    num_agents = NUM_AGENTS
    action_shape = (NUM_AGENTS, 2)
    pair_indices = PAIR_INDICES

    def __init__(self, config: StressConfig | None = None):
        self.config = config or StressConfig()
        self.walls, self.goals = build_layout(self.config)
        self.reset()

    def observation(self) -> np.ndarray:
        """Full state observation, one row per agent, shape ``[8, 34]``."""
        rows = []
        for agent in range(self.num_agents):
            relative = []
            for other in range(self.num_agents):
                if agent != other:
                    relative.extend((
                        self.positions[other] - self.positions[agent],
                        self.velocities[other] - self.velocities[agent],
                    ))
            rows.append(np.concatenate((
                self.positions[agent], self.velocities[agent],
                self.goals[agent] - self.positions[agent], *relative,
            )))
        return np.asarray(rows, dtype=np.float32)

    def _rectangles(self) -> tuple[tuple[float, float, float, float], ...]:
        cfg = self.config
        return (
            (-cfg.horizontal_corridor_half_length, cfg.horizontal_corridor_half_length, -cfg.corridor_half_width, cfg.corridor_half_width),
            (-cfg.horizontal_half_length, -cfg.horizontal_corridor_half_length, -cfg.staging_half_height, cfg.staging_half_height),
            (cfg.horizontal_corridor_half_length, cfg.horizontal_half_length, -cfg.staging_half_height, cfg.staging_half_height),
            (-cfg.junction_x - cfg.junction_half_width, -cfg.junction_x + cfg.junction_half_width, -cfg.vertical_half_length, cfg.vertical_half_length),
            (cfg.junction_x - cfg.junction_half_width, cfg.junction_x + cfg.junction_half_width, -cfg.vertical_half_length, cfg.vertical_half_length),
        )

    def outside(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape [N,2]")
        inside = np.zeros(len(positions), dtype=bool)
        for x0, x1, y0, y1 in self._rectangles():
            inside |= (
                (positions[:, 0] >= x0) & (positions[:, 0] <= x1)
                & (positions[:, 1] >= y0) & (positions[:, 1] <= y1)
            )
        return ~inside

    def distances(self, positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        p = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        if p.shape != self.action_shape:
            raise ValueError("positions must have shape [8,2]")
        wall = (
            point_segment_distance(p[:, None], self.walls[None, :, 0], self.walls[None, :, 1])
            - self.config.agent_radius - self.config.wall_radius
        )
        pair = np.asarray([
            np.linalg.norm(p[first] - p[second]) - 2 * self.config.agent_radius
            for first, second in self.pair_indices
        ], dtype=np.float64)
        return wall, pair

    def reset(self, positions: np.ndarray | None = None) -> np.ndarray:
        self.positions = np.array(default_positions() if positions is None else positions, dtype=np.float64, copy=True)
        if self.positions.shape != self.action_shape or not np.isfinite(self.positions).all():
            raise ValueError("invalid initial positions")
        wall, pair = self.distances(self.positions)
        if self.outside(self.positions).any() or wall.min() <= self.config.wall_collision_margin or pair.min() <= self.config.agent_collision_margin:
            raise ValueError("initial condition must be collision-free with margin")
        self.velocities = np.zeros(self.action_shape, dtype=np.float64)
        self.step_count = 0
        self.distance_history = [np.linalg.norm(self.goals - self.positions, axis=-1)]
        self.candidate_since: int | None = None
        self.stuck_timer = self.max_stuck_timer = 0.0
        self.ever_candidate_deadlock = False
        self.first_success_step = self.first_deadlock_step = None
        self.first_wall_collision_step = self.first_agent_collision_step = None
        self.done = False
        return self.observation()

    def snapshot(self) -> dict[str, Any]:
        return {
            "positions": self.positions.copy(),
            "last_applied_velocity": self.velocities.copy(),
            "goals": self.goals.copy(),
            "walls": self.walls.copy(),
            "pair_indices": np.asarray(self.pair_indices, dtype=np.int64),
            "config": self.config.to_dict(),
            "step": self.step_count,
        }

    def junction_occupancy(self, positions: np.ndarray | None = None) -> np.ndarray:
        """Boolean ``[8,2]`` membership in the two explicit conflict squares."""
        p = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        cfg = self.config
        result = []
        for center in (-cfg.junction_x, cfg.junction_x):
            result.append((np.abs(p[:, 0] - center) <= cfg.junction_half_width) & (np.abs(p[:, 1]) <= cfg.junction_half_width))
        return np.stack(result, axis=1)

    def bridge_occupancy(self, positions: np.ndarray | None = None) -> np.ndarray:
        p = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        return (np.abs(p[:, 0]) < self.config.junction_x - self.config.junction_half_width) & (np.abs(p[:, 1]) <= self.config.corridor_half_width)

    def step(self, executed_velocity: np.ndarray):
        if self.done:
            raise RuntimeError("reset a terminated episode before stepping")
        action = np.array(executed_velocity, dtype=np.float64, copy=True)
        if action.shape != self.action_shape or not np.isfinite(action).all():
            raise ValueError("expected finite action [8,2]")
        if np.linalg.norm(action, axis=-1).max() > self.config.max_speed + 1e-9:
            raise ValueError("action exceeds maximum speed")
        before = self.positions.copy()
        previous_errors = np.linalg.norm(self.goals - before, axis=-1)
        self.positions = before + self.config.dt * action
        self.velocities = action.copy()
        self.step_count += 1
        wall, pair = self.distances()
        swept_wall = (
            segment_distance(before[:, None], self.positions[:, None], self.walls[None, :, 0], self.walls[None, :, 1])
            - self.config.agent_radius - self.config.wall_radius
        )
        swept_pair = np.asarray([
            point_segment_distance(np.zeros(2), before[first] - before[second], self.positions[first] - self.positions[second])
            - 2 * self.config.agent_radius
            for first, second in self.pair_indices
        ], dtype=np.float64)
        outside = self.outside(self.positions)
        wall_collision = bool((swept_wall <= self.config.wall_collision_margin).any() or outside.any())
        agent_collision = bool((swept_pair <= self.config.agent_collision_margin).any())
        errors = np.linalg.norm(self.goals - self.positions, axis=-1)
        speeds = np.linalg.norm(action, axis=-1)
        success = bool((errors <= self.config.goal_tolerance).all())
        self.distance_history.append(errors.copy())
        window = self.config.progress_window_seconds / self.config.dt
        start = self.step_count - window
        ready = start >= -1e-10
        progress = np.full(self.num_agents, np.nan, dtype=np.float64)
        if ready:
            start = max(0.0, start)
            lower, upper = int(math.floor(start)), int(math.ceil(start))
            alpha = start - lower
            past = (1 - alpha) * self.distance_history[lower] + alpha * self.distance_history[upper]
            progress = past - errors
        candidate = bool(
            ready and not success and np.max(np.abs(progress)) < self.config.progress_epsilon
            and float(speeds.max()) < self.config.max_speed * self.config.speed_epsilon_fraction
        )
        if candidate:
            self.ever_candidate_deadlock = True
            if self.candidate_since is None:
                self.candidate_since = self.step_count
            self.stuck_timer = (self.step_count - self.candidate_since) * self.config.dt
        else:
            self.candidate_since, self.stuck_timer = None, 0.0
        self.max_stuck_timer = max(self.max_stuck_timer, self.stuck_timer)
        deadlock = bool(candidate and self.stuck_timer >= self.config.deadlock_hold_seconds - 1e-10)
        for happened, attr in (
            (success, "first_success_step"), (deadlock, "first_deadlock_step"),
            (wall_collision, "first_wall_collision_step"), (agent_collision, "first_agent_collision_step"),
        ):
            if happened and getattr(self, attr) is None:
                setattr(self, attr, self.step_count)
        termination = "running"
        if (wall_collision or agent_collision) and self.config.terminate_on_collision:
            termination = "collision"
        elif success and self.config.terminate_on_success:
            termination = "success"
        elif deadlock and self.config.terminate_on_deadlock:
            termination = "deadlock"
        elif self.step_count >= self.config.max_steps:
            termination = "timeout"
        self.done = termination != "running"
        info = {
            "step": self.step_count, "time": self.step_count * self.config.dt,
            "goal_errors": errors.copy(), "speeds": speeds.copy(),
            "wall_collision": wall_collision, "agent_collision": agent_collision,
            "min_swept_wall_distance": float(swept_wall.min()),
            "min_swept_agent_distance": float(swept_pair.min()),
            "task_success": success, "candidate_deadlock": candidate,
            "deadlock": deadlock, "stuck_timer": float(self.stuck_timer),
            "window_progress": progress.copy(), "termination": termination,
            "junction_occupancy": self.junction_occupancy(),
            "bridge_occupancy": self.bridge_occupancy(),
            "outside_workspace": outside.copy(),
        }
        reward = np.asarray(previous_errors.sum() - errors.sum(), dtype=np.float32)
        return self.observation(), reward, self.done, info

    def summary(self) -> dict[str, Any]:
        success = self.first_success_step is not None
        return {
            "success": success,
            "final_goal_success": bool((np.linalg.norm(self.goals - self.positions, axis=-1) <= self.config.goal_tolerance).all()),
            "collision_free_success": success and self.first_wall_collision_step is None and self.first_agent_collision_step is None,
            "wall_collision": self.first_wall_collision_step is not None,
            "agent_collision": self.first_agent_collision_step is not None,
            "deadlock": self.first_deadlock_step is not None,
            "episode_steps": self.step_count,
            "max_stuck_timer": float(self.max_stuck_timer),
        }

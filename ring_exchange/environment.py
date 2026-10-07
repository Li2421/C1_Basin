"""Physical environment for the four-agent Ring Exchange benchmark.

Four discs exchange approximately opposite locations in a *wide* annulus.
There is deliberately no lane, one-at-a-time gate, right-of-way rule, or
CW/CCW bit in this plant.  The shared central obstacle makes incompatible
lateral decisions create head-on and cyclic conflicts rather than reducing the
task to a left-vs-right bottleneck.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


AGENT_NAMES = ("A", "B", "C", "D")
PAIR_INDICES = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
SPLITS = ("train", "development", "test")


@dataclass(frozen=True)
class Config:
    schema: str = "ring_exchange_single_integrator_v1"
    dt: float = 0.05
    max_steps: int = 700
    max_speed: float = 0.52
    agent_radius: float = 0.16
    obstacle_radius: float = 0.90
    outer_radius: float = 3.25
    collision_margin: float = 0.005
    agent_collision_margin: float = 0.0
    goal_tolerance: float = 0.09
    terminate_on_collision: bool = True
    terminate_on_success: bool = True

    def __post_init__(self) -> None:
        if self.schema not in {"ring_exchange_single_integrator_v1", "ring_exchange_single_integrator_v2_local_frame"}:
            raise ValueError("Unknown dynamics schema")
        values = (self.dt, self.max_steps, self.max_speed, self.agent_radius,
                  self.obstacle_radius, self.outer_radius, self.goal_tolerance)
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
            raise ValueError("All physical constants must be positive and finite")
        # A radial free width of 2.35 is many agent diameters, expressly ruling
        # out a disguised single-file obstacle passage.
        if self.outer_radius - self.obstacle_radius < 6 * self.agent_radius:
            raise ValueError("annulus must remain broadly passable")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RingInstance:
    """A complete immutable initial-state draw, including the four goals."""
    positions: np.ndarray
    velocities: np.ndarray
    goals: np.ndarray
    split: str
    seed: int
    global_angle: float

    def copy(self) -> "RingInstance":
        return RingInstance(self.positions.copy(), self.velocities.copy(), self.goals.copy(),
                            self.split, self.seed, self.global_angle)


def _angle_vector(theta: np.ndarray) -> np.ndarray:
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1)


def _split_seed(split: str, seed: int) -> np.random.SeedSequence:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    # Separate streams make split construction independent, not a filtered
    # subset of a shared collection of failures.
    tags = {"train": 11939, "development": 27183, "test": 41771}
    return np.random.SeedSequence([tags[split], int(seed)])


def sample_instance(seed: int, split: str = "train", config: Config | None = None) -> RingInstance:
    """Draw one broad, safe, independently reproducible benchmark state.

    A global rotation, per-agent angular/radial variation, velocity, and goal
    angle all vary.  Adjacent start angles retain a generous minimum gap, so
    invalid collision states are never used as nominal training data.
    """
    cfg = config or Config()
    rng = np.random.default_rng(_split_seed(split, seed))
    global_angle = float(rng.uniform(-math.pi, math.pi))
    anchors = global_angle + np.arange(4) * (math.pi / 2)
    start_theta = anchors + rng.uniform(-0.20, 0.20, size=4)
    start_radius = rng.uniform(2.30, 2.68, size=4)
    # Opposite-side goals with independent small angular and radial offsets.
    goal_theta = start_theta + math.pi + rng.uniform(-0.18, 0.18, size=4)
    goal_radius = rng.uniform(2.30, 2.68, size=4)
    positions = start_radius[:, None] * _angle_vector(start_theta)
    goals = goal_radius[:, None] * _angle_vector(goal_theta)
    headings = rng.uniform(-math.pi, math.pi, size=4)
    speeds = rng.uniform(0.0, 0.11, size=4)
    velocities = speeds[:, None] * _angle_vector(headings)
    return RingInstance(positions.astype(np.float64), velocities.astype(np.float64),
                        goals.astype(np.float64), split, int(seed), global_angle)


def sample_initial_state(split: str, seed: int, config: Config | None = None) -> RingInstance:
    """Pipeline-friendly alias with ``(split, seed)`` argument order."""
    return sample_instance(seed=seed, split=split, config=config)


def point_segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    point, first, second = map(np.asarray, (point, first, second))
    direction = second - first
    fraction = np.clip(np.sum((point - first) * direction, axis=-1) /
                       np.maximum(np.sum(direction * direction, axis=-1), 1e-30), 0.0, 1.0)
    return np.linalg.norm(point - (first + fraction[..., None] * direction), axis=-1)


class AgentView:
    def __init__(self, env: "RingExchangeEnv", index: int):
        self.env, self.index = env, index

    @property
    def name(self) -> str:
        return AGENT_NAMES[self.index]

    @property
    def position(self) -> np.ndarray:
        return self.env.positions[self.index]

    @property
    def velocity(self) -> np.ndarray:
        return self.env.velocities[self.index]

    @property
    def goal(self) -> np.ndarray:
        return self.env.goals[self.index]


class RingExchangeEnv:
    """A joint four-disc first-order plant with swept collision detection."""
    num_agents = 4
    action_shape = (4, 2)
    pair_indices = PAIR_INDICES
    agent_order = AGENT_NAMES

    def __init__(self, config: Config | None = None, instance: RingInstance | None = None):
        self.config = config or Config()
        # Named geometry is exposed for audit logs and a future safety layer;
        # it is not included as a handcrafted policy mode.
        self.wall_names = ("central_obstacle", "outer_boundary")
        self.obstacle_center = np.zeros(2, dtype=np.float64)
        self.agents = [AgentView(self, i) for i in range(self.num_agents)]
        self.reset(instance=instance)

    def observation(self) -> np.ndarray:
        """Full state observation, shape ``[4, 23]``.

        Rows follow canonical order ``A,B,C,D`` and contain own position,
        velocity and goal displacement; three ascending-index relative
        position/velocity blocks; then fixed geometry ``[obstacle centre
        relative to agent (2), obstacle radius, outer radius, obstacle signed
        clearance]``.  It contains no circulation-mode or priority label.
        """
        rows = []
        for i in range(4):
            relative = []
            for j in range(4):
                if i != j:
                    relative.extend((self.positions[j] - self.positions[i],
                                     self.velocities[j] - self.velocities[i]))
            radius = float(np.linalg.norm(self.positions[i]))
            geometry = np.array((-self.positions[i, 0], -self.positions[i, 1],
                                 self.config.obstacle_radius, self.config.outer_radius,
                                 radius - self.config.obstacle_radius - self.config.agent_radius))
            rows.append(np.concatenate((self.positions[i], self.velocities[i],
                                        self.goals[i] - self.positions[i], *relative, geometry)))
        return np.asarray(rows, dtype=np.float32)

    def joint_observation(self) -> np.ndarray:
        return self.observation().reshape(-1)

    def outside(self, positions: np.ndarray) -> np.ndarray:
        radii = np.linalg.norm(np.asarray(positions, dtype=np.float64), axis=-1)
        return ((radii <= self.config.obstacle_radius + self.config.agent_radius + self.config.collision_margin) |
                (radii >= self.config.outer_radius - self.config.agent_radius - self.config.collision_margin))

    def distances(self, positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        p = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        if p.shape != self.action_shape:
            raise ValueError("positions must have shape [4,2]")
        radii = np.linalg.norm(p, axis=-1)
        # First column is central obstacle clearance; second is outer-boundary
        # clearance.  This makes obstacle identity auditable downstream.
        wall = np.stack((radii - self.config.obstacle_radius - self.config.agent_radius,
                         self.config.outer_radius - self.config.agent_radius - radii), axis=-1)
        pair = np.asarray([np.linalg.norm(p[i] - p[j]) - 2 * self.config.agent_radius
                           for i, j in PAIR_INDICES], dtype=np.float64)
        return wall, pair

    def obstacle_clearance(self, positions: np.ndarray | None = None) -> np.ndarray:
        """Signed centre-disk clearance used by obstacle safety constraints."""
        p = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        return np.linalg.norm(p, axis=-1) - self.config.obstacle_radius - self.config.agent_radius

    def obstacle_normals(self, positions: np.ndarray | None = None) -> np.ndarray:
        """Outward unit normals of the central disk (safe for zero input)."""
        p = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        return p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)

    def reset(self, positions: np.ndarray | None = None, *, velocities: np.ndarray | None = None,
              goals: np.ndarray | None = None, instance: RingInstance | None = None) -> np.ndarray:
        if instance is not None:
            if positions is not None or velocities is not None or goals is not None:
                raise ValueError("pass either instance or explicit state, not both")
            positions, velocities, goals = instance.positions, instance.velocities, instance.goals
            self.instance = instance.copy()
        else:
            if positions is None:
                instance = sample_instance(0, "train", self.config)
                positions, velocities, goals = instance.positions, instance.velocities, instance.goals
                self.instance = instance
            else:
                self.instance = None
        self.positions = np.asarray(positions, dtype=np.float64).copy()
        self.velocities = (np.zeros(self.action_shape) if velocities is None else
                           np.asarray(velocities, dtype=np.float64).copy())
        if goals is None:
            raise ValueError("goals are required with explicit positions")
        self.goals = np.asarray(goals, dtype=np.float64).copy()
        if any(x.shape != self.action_shape or not np.isfinite(x).all()
               for x in (self.positions, self.velocities, self.goals)):
            raise ValueError("state arrays must be finite [4,2]")
        if np.linalg.norm(self.velocities, axis=-1).max(initial=0.0) > self.config.max_speed:
            raise ValueError("initial velocity exceeds speed limit")
        wall, pair = self.distances()
        if self.outside(self.positions).any() or wall.min() <= self.config.collision_margin or pair.min() <= 0.0:
            raise ValueError("initial state must be safe and inside the annulus")
        self.step_count = 0
        self.done = False
        self.first_success_step = self.first_obstacle_collision_step = None
        self.first_outer_collision_step = self.first_agent_collision_step = None
        return self.observation()

    def snapshot(self) -> dict[str, Any]:
        return {"positions": self.positions.copy(), "last_applied_velocity": self.velocities.copy(),
                "goals": self.goals.copy(), "obstacle_center": self.obstacle_center.copy(),
                "obstacle_radius": self.config.obstacle_radius, "outer_radius": self.config.outer_radius,
                "wall_names": self.wall_names,
                "pair_indices": np.asarray(PAIR_INDICES, dtype=np.int64), "config": self.config.to_dict(),
                "step": self.step_count}

    def augmented_state(self) -> dict[str, Any]:
        return {**self.snapshot(), "done": self.done, "first_success_step": self.first_success_step,
                "first_obstacle_collision_step": self.first_obstacle_collision_step,
                "first_outer_collision_step": self.first_outer_collision_step,
                "first_agent_collision_step": self.first_agent_collision_step}

    def restore_augmented_state(self, state: dict[str, Any]) -> np.ndarray:
        if state.get("config") != self.config.to_dict():
            raise ValueError("state/config mismatch")
        self.positions = np.asarray(state["positions"], dtype=np.float64).copy()
        self.velocities = np.asarray(state["last_applied_velocity"], dtype=np.float64).copy()
        self.goals = np.asarray(state["goals"], dtype=np.float64).copy()
        if any(x.shape != self.action_shape for x in (self.positions, self.velocities, self.goals)):
            raise ValueError("malformed state")
        self.step_count = int(state["step"])
        self.done = bool(state["done"])
        for field in ("first_success_step", "first_obstacle_collision_step",
                      "first_outer_collision_step", "first_agent_collision_step"):
            setattr(self, field, state[field])
        return self.observation()

    def step(self, executed_velocity: np.ndarray):
        if self.done:
            raise RuntimeError("reset a terminated episode before stepping")
        u = np.asarray(executed_velocity, dtype=np.float64)
        if u.shape != self.action_shape or not np.isfinite(u).all():
            raise ValueError("expected finite action [4,2]")
        if np.linalg.norm(u, axis=-1).max(initial=0.0) > self.config.max_speed + 1e-9:
            raise ValueError("action exceeds speed bound; plant never clips")
        before = self.positions.copy()
        previous_error = np.linalg.norm(self.goals - before, axis=-1)
        after = before + self.config.dt * u
        self.positions, self.velocities = after, u.copy()
        self.step_count += 1
        wall, pair = self.distances()
        # Min distance from obstacle centre over each swept segment catches a
        # disk crossing between endpoints.  Outer boundary is convex, so its
        # maximum radius over a segment occurs at an endpoint.
        swept_radius_min = point_segment_distance(np.zeros(2), before, after)
        swept_obstacle = swept_radius_min - self.config.obstacle_radius - self.config.agent_radius
        swept_outer = np.minimum(self.config.outer_radius - self.config.agent_radius - np.linalg.norm(before, axis=-1),
                                 self.config.outer_radius - self.config.agent_radius - np.linalg.norm(after, axis=-1))
        swept_pairs = np.asarray([point_segment_distance(np.zeros(2), before[i] - before[j], after[i] - after[j]) -
                                  2 * self.config.agent_radius for i, j in PAIR_INDICES])
        obstacle_collision = bool((swept_obstacle <= self.config.collision_margin).any())
        outer_collision = bool((swept_outer <= self.config.collision_margin).any())
        agent_collision = bool((swept_pairs <= self.config.agent_collision_margin).any())
        errors = np.linalg.norm(self.goals - after, axis=-1)
        success = bool((errors <= self.config.goal_tolerance).all())
        for flag, field in ((success, "first_success_step"), (obstacle_collision, "first_obstacle_collision_step"),
                            (outer_collision, "first_outer_collision_step"), (agent_collision, "first_agent_collision_step")):
            if flag and getattr(self, field) is None:
                setattr(self, field, self.step_count)
        if (obstacle_collision or outer_collision or agent_collision) and self.config.terminate_on_collision:
            termination = "collision"
        elif success and self.config.terminate_on_success:
            termination = "success"
        elif self.step_count >= self.config.max_steps:
            termination = "timeout"
        else:
            termination = "running"
        self.done = termination != "running"
        info = {"step": self.step_count, "time": self.step_count * self.config.dt,
                "positions": after.copy(), "velocities": u.copy(), "goal_errors": errors,
                "speeds": np.linalg.norm(u, axis=-1), "wall_distances": wall,
                "wall_names": ("central_obstacle", "outer_boundary"), "obstacle_collision": obstacle_collision,
                "outer_collision": outer_collision, "agent_collision": agent_collision,
                "agent_pair_indices": np.asarray(PAIR_INDICES, dtype=np.int64),
                "agent_surface_distances": pair, "swept_obstacle_distance": swept_obstacle,
                "swept_outer_distance": swept_outer, "swept_agent_distances": swept_pairs,
                "min_swept_agent_distance": float(swept_pairs.min()), "task_success": success,
                "termination": termination,
                "integration_residual": float(np.abs((after - before) - self.config.dt * u).max())}
        reward = np.float32((previous_error - errors).sum() + 0.01 * success - obstacle_collision - outer_collision - agent_collision)
        return self.observation(), reward, self.done, info

    def summary(self) -> dict[str, Any]:
        return {"success": self.first_success_step is not None,
                "final_goal_success": bool((np.linalg.norm(self.goals - self.positions, axis=-1) <= self.config.goal_tolerance).all()),
                "collision_free_success": self.first_success_step is not None and self.first_obstacle_collision_step is None and
                                          self.first_outer_collision_step is None and self.first_agent_collision_step is None,
                "obstacle_collision": self.first_obstacle_collision_step is not None,
                "outer_collision": self.first_outer_collision_step is not None,
                "wall_collision": self.first_outer_collision_step is not None,
                "agent_collision": self.first_agent_collision_step is not None,
                "episode_steps": self.step_count, "first_success_step": self.first_success_step,
                "first_obstacle_collision_step": self.first_obstacle_collision_step,
                "first_outer_collision_step": self.first_outer_collision_step,
                "first_agent_collision_step": self.first_agent_collision_step}


Env = RingExchangeEnv


@dataclass(frozen=True)
class LocalFrameConfig(Config):
    """v6 data/policy representation fingerprint; physical plant is unchanged."""
    schema: str = "ring_exchange_single_integrator_v2_local_frame"

    def __post_init__(self) -> None:
        super().__post_init__()
        # The local chart has the same physical constants as v1, so allowing a
        # caller to override this tag to v1 would make a local observation/
        # action dataset fingerprint-identical to a world-frame one.  Refuse
        # that representation mismatch at construction time.
        if self.schema != "ring_exchange_single_integrator_v2_local_frame":
            raise ValueError("LocalFrameConfig requires the v2 local-frame schema")

__all__ = ("AGENT_NAMES", "PAIR_INDICES", "SPLITS", "Config", "LocalFrameConfig", "RingInstance", "sample_instance", "sample_initial_state",
           "point_segment_distance", "RingExchangeEnv", "Env")

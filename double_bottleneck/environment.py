"""Four-agent, two-bottleneck deterministic single-integrator environment.

Dynamics and event semantics intentionally mirror Toy Give-Way:

    p[k+1] = p[k] + dt * u[k]

Collisions are observed (including swept motion) and never alter the motion.
The environment contains no policy, eta corrector, or hard-safety projector.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .scenario import AGENT_NAMES, INITIAL_REGIMES, build_layout, initial_positions


PAIR_INDICES = tuple((i, j) for i in range(4) for j in range(i + 1, 4))


@dataclass(frozen=True)
class Config:
    schema: str = "double_bottleneck_single_integrator_v1"
    dt: float = 0.05
    # Preserve the Toy Give-Way physical timeout (850 * 0.05 = 42.5 s).
    max_steps: int = 850
    max_speed: float = 0.5
    agent_radius: float = 0.16
    world_half_length: float = 4.0
    outer_half_height: float = 0.82
    chamber_half_length: float = 0.95
    chamber_half_height: float = 0.82
    bottleneck_length: float = 0.90
    bottleneck_width: float = 0.50
    wall_radius: float = 4 / 600
    wall_collision_margin: float = 0.005
    agent_collision_margin: float = 0.0
    agent_reward_margin: float = 0.005
    goal_tolerance: float = 0.08
    detection_protocol: str = "window_progress_v2"
    progress_window_seconds: float = 2.0
    deadlock_hold_seconds: float = 5.0
    progress_epsilon: float = 0.01
    speed_epsilon_fraction: float = 0.05
    initial_regime: str = "weakly_asymmetric"
    terminate_on_collision: bool = True
    terminate_on_success: bool = True
    terminate_on_deadlock: bool = True

    def __post_init__(self):
        if self.schema != "double_bottleneck_single_integrator_v1":
            raise ValueError("Unknown dynamics schema")
        if self.detection_protocol != "window_progress_v2":
            raise ValueError("Unknown detection protocol")
        if self.initial_regime not in INITIAL_REGIMES:
            raise ValueError(f"Unknown initial regime {self.initial_regime!r}")
        positive = (
            self.dt,
            self.max_steps,
            self.max_speed,
            self.agent_radius,
            self.world_half_length,
            self.outer_half_height,
            self.chamber_half_length,
            self.chamber_half_height,
            self.bottleneck_length,
            self.bottleneck_width,
            self.progress_window_seconds,
            self.deadlock_hold_seconds,
            self.progress_epsilon,
            self.speed_epsilon_fraction,
        )
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in positive):
            raise ValueError("Dynamics, geometry, horizon, and monitor constants must be positive and finite")
        if self.chamber_half_length + self.bottleneck_length >= self.world_half_length:
            raise ValueError("Bottlenecks must leave nonempty outer staging regions")
        if self.bottleneck_width / 2 >= min(self.chamber_half_height, self.outer_half_height):
            raise ValueError("Bottlenecks must be narrower than chamber and staging regions")
        disk_diameter = 2 * (self.agent_radius + self.wall_radius + self.wall_collision_margin)
        if self.bottleneck_width <= disk_diameter:
            raise ValueError("Bottleneck is too narrow for even one collision-free agent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def point_segment_distance(p, a, b):
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    d = b - a
    t = np.clip(np.sum((p - a) * d, axis=-1) / np.maximum(np.sum(d * d, axis=-1), 1e-30), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * d), axis=-1)


def segment_distance(a, b, c, d):
    """Exact 2-D segment distance, vectorized over leading dimensions."""
    a, b, c, d = map(np.asarray, (a, b, c, d))
    r, s, q = b - a, d - c, c - a

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    denom = cross(r, s)
    safe = np.where(np.abs(denom) > 1e-15, denom, 1.0)
    t, u = cross(q, s) / safe, cross(q, r) / safe
    intersect = (np.abs(denom) > 1e-15) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    distance = np.minimum.reduce(
        (
            point_segment_distance(a, c, d),
            point_segment_distance(b, c, d),
            point_segment_distance(c, a, b),
            point_segment_distance(d, a, b),
        )
    )
    return np.where(intersect, 0.0, distance)


class AgentView:
    def __init__(self, env: "DoubleBottleneckEnv", index: int):
        self.env, self.index = env, index

    @property
    def name(self):
        return AGENT_NAMES[self.index]

    @property
    def position(self):
        return self.env.positions[self.index]

    @property
    def velocity(self):
        return self.env.velocities[self.index]

    @property
    def goal(self):
        return self.env.goals[self.index]


class DoubleBottleneckEnv:
    """Four-agent plant and event monitor, independent of controller choice."""

    num_agents = 4
    action_shape = (4, 2)
    pair_indices = PAIR_INDICES

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        layout = build_layout(self.config)
        self.walls = layout.walls
        self.wall_names = layout.wall_names
        self.goals = layout.goals
        self.agents = [AgentView(self, i) for i in range(self.num_agents)]
        self.reset()

    def observation(self) -> np.ndarray:
        """Full four-agent observation, shape ``[4,18]``.

        Each row is ``[p_i, v_i, goal_i-p_i, (p_j-p_i, v_j-v_i) for j!=i]``
        with other-agent blocks in ascending global index order.  Frozen
        two-agent Flow-BC must use :meth:`pair_observation`, not this tensor.
        """
        rows = []
        for i in range(self.num_agents):
            relative = []
            for j in range(self.num_agents):
                if i != j:
                    relative.extend((self.positions[j] - self.positions[i], self.velocities[j] - self.velocities[i]))
            rows.append(
                np.concatenate(
                    (self.positions[i], self.velocities[i], self.goals[i] - self.positions[i], *relative)
                )
            )
        return np.asarray(rows, dtype=np.float32)

    def pair_observation(self, first: int, second: int) -> np.ndarray:
        """Return the exact legacy Flow-BC primitive for one ordered pair.

        The output has shape ``[2,10]``.  Its first row is the view of
        ``first`` relative to ``second`` and the second row is the reciprocal
        view.  How multiple pair-policy outputs are aggregated is deliberately
        outside the environment.
        """
        if first == second or not (0 <= first < self.num_agents and 0 <= second < self.num_agents):
            raise ValueError("Expected two distinct valid agent indices")
        rows = []
        for i, j in ((first, second), (second, first)):
            rows.append(
                np.concatenate(
                    (
                        self.positions[i],
                        self.velocities[i],
                        self.goals[i] - self.positions[i],
                        self.positions[j] - self.positions[i],
                        self.velocities[j] - self.velocities[i],
                    )
                )
            )
        return np.asarray(rows, dtype=np.float32)

    def all_pair_observations(self) -> np.ndarray:
        """All six unordered-pair Flow primitives, shape ``[6,2,10]``."""
        return np.stack([self.pair_observation(i, j) for i, j in self.pair_indices])

    def outside(self, positions) -> np.ndarray:
        p = np.asarray(positions)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("positions must have shape [N,2]")
        length = self.config.world_half_length
        inner = self.config.chamber_half_length
        outer_gate = inner + self.config.bottleneck_length
        narrow = self.config.bottleneck_width / 2
        in_left = (p[:, 0] >= -length) & (p[:, 0] <= -outer_gate) & (np.abs(p[:, 1]) <= self.config.outer_half_height)
        in_left_gate = (p[:, 0] >= -outer_gate) & (p[:, 0] <= -inner) & (np.abs(p[:, 1]) <= narrow)
        in_chamber = (np.abs(p[:, 0]) <= inner) & (np.abs(p[:, 1]) <= self.config.chamber_half_height)
        in_right_gate = (p[:, 0] >= inner) & (p[:, 0] <= outer_gate) & (np.abs(p[:, 1]) <= narrow)
        in_right = (p[:, 0] >= outer_gate) & (p[:, 0] <= length) & (np.abs(p[:, 1]) <= self.config.outer_half_height)
        return ~(in_left | in_left_gate | in_chamber | in_right_gate | in_right)

    def distances(self, positions=None) -> tuple[np.ndarray, np.ndarray]:
        p = self.positions if positions is None else np.asarray(positions)
        if p.shape != self.action_shape:
            raise ValueError("positions must have shape [4,2]")
        wall = (
            point_segment_distance(p[:, None, :], self.walls[None, :, 0, :], self.walls[None, :, 1, :])
            - self.config.agent_radius
            - self.config.wall_radius
        )
        pair = np.asarray(
            [np.linalg.norm(p[i] - p[j]) - 2 * self.config.agent_radius for i, j in self.pair_indices],
            dtype=np.float64,
        )
        return wall, pair

    def reset(self, positions=None, regime: str | None = None) -> np.ndarray:
        regime = self.config.initial_regime if regime is None else regime
        start = initial_positions(self.config, regime) if positions is None else positions
        self.positions = np.array(start, dtype=np.float64, copy=True)
        if self.positions.shape != self.action_shape or not np.isfinite(self.positions).all():
            raise ValueError("Invalid initial positions")
        wall, pair = self.distances()
        if (
            self.outside(self.positions).any()
            or wall.min() <= self.config.wall_collision_margin
            or pair.min() <= self.config.agent_collision_margin
        ):
            raise ValueError("Initial condition must be safe")
        self.initial_regime = regime
        self.velocities = np.zeros(self.action_shape, dtype=np.float64)
        self.step_count = 0
        self.distance_history = [np.linalg.norm(self.goals - self.positions, axis=-1)]
        self.candidate_since = None
        self.stuck_timer = self.max_stuck_timer = 0.0
        self.ever_candidate_deadlock = False
        self.first_success_step = self.first_deadlock_step = None
        self.first_wall_collision_step = self.first_agent_collision_step = None
        self.done = False
        return self.observation()

    def snapshot(self) -> dict[str, Any]:
        """Detached physical state for post-hoc controller/safety code."""
        return dict(
            positions=self.positions.copy(),
            last_applied_velocity=self.velocities.copy(),
            goals=self.goals.copy(),
            walls=self.walls.copy(),
            pair_indices=np.asarray(self.pair_indices, dtype=np.int64),
            config=self.config.to_dict(),
            step=self.step_count,
        )

    def augmented_state(self) -> dict[str, Any]:
        """Detached exact-restoration state, including strict-monitor memory."""
        return dict(
            **self.snapshot(),
            distance_history=np.asarray(self.distance_history, dtype=np.float64).copy(),
            candidate_since=self.candidate_since,
            stuck_timer=float(self.stuck_timer),
            max_stuck_timer=float(self.max_stuck_timer),
            ever_candidate_deadlock=bool(self.ever_candidate_deadlock),
            first_success_step=self.first_success_step,
            first_deadlock_step=self.first_deadlock_step,
            first_wall_collision_step=self.first_wall_collision_step,
            first_agent_collision_step=self.first_agent_collision_step,
            done=bool(self.done),
            initial_regime=self.initial_regime,
        )

    def restore_augmented_state(self, state: dict[str, Any]) -> np.ndarray:
        """Restore a state produced by :meth:`augmented_state` exactly."""
        if state.get("config") != self.config.to_dict():
            raise ValueError("State/config mismatch")
        positions = np.asarray(state["positions"], dtype=np.float64)
        velocities = np.asarray(state["last_applied_velocity"], dtype=np.float64)
        history = np.asarray(state["distance_history"], dtype=np.float64)
        step = int(state["step"])
        if positions.shape != self.action_shape or velocities.shape != self.action_shape:
            raise ValueError("Malformed physical state")
        if history.shape != (step + 1, self.num_agents):
            raise ValueError("Malformed monitor history")
        if not all(np.isfinite(x).all() for x in (positions, velocities, history)):
            raise ValueError("State must be finite")
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.distance_history = [row.copy() for row in history]
        self.step_count = step
        self.candidate_since = state["candidate_since"]
        self.stuck_timer = float(state["stuck_timer"])
        self.max_stuck_timer = float(state["max_stuck_timer"])
        self.ever_candidate_deadlock = bool(state["ever_candidate_deadlock"])
        for name in (
            "first_success_step",
            "first_deadlock_step",
            "first_wall_collision_step",
            "first_agent_collision_step",
        ):
            setattr(self, name, state[name])
        self.done = bool(state["done"])
        self.initial_regime = str(state["initial_regime"])
        return self.observation()

    def diagnostics(self) -> dict[str, Any]:
        """Current geometry and strict-monitor state without advancing time."""
        wall, pair = self.distances()
        return dict(
            step=self.step_count,
            time=self.step_count * self.config.dt,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            goal_errors=np.linalg.norm(self.goals - self.positions, axis=-1),
            wall_distances=wall,
            agent_pair_indices=np.asarray(self.pair_indices, dtype=np.int64),
            agent_surface_distances=pair,
            min_agent_surface_distance=float(pair.min()),
            candidate_deadlock=self.candidate_since is not None,
            stuck_timer=float(self.stuck_timer),
            max_stuck_timer=float(self.max_stuck_timer),
            done=bool(self.done),
        )

    def step(self, executed_velocity):
        if self.done:
            raise RuntimeError("Reset a terminated episode before stepping")
        u = np.array(executed_velocity, dtype=np.float64, copy=True)
        if u.shape != self.action_shape or not np.isfinite(u).all():
            raise ValueError("Invalid executed velocity; expected finite [4,2]")
        if np.any(np.linalg.norm(u, axis=-1) > self.config.max_speed + 1e-9):
            raise ValueError("Executed action exceeds speed bound; no hidden plant clipping is allowed")

        before = self.positions.copy()
        previous_errors = np.linalg.norm(self.goals - before, axis=-1)
        self.positions = before + self.config.dt * u
        self.velocities = u.copy()
        self.step_count += 1

        wall, pair_distances = self.distances()
        swept_wall = (
            segment_distance(
                before[:, None, :],
                self.positions[:, None, :],
                self.walls[None, :, 0, :],
                self.walls[None, :, 1, :],
            )
            - self.config.agent_radius
            - self.config.wall_radius
        )
        swept_pairs = []
        for i, j in self.pair_indices:
            relative_before = before[i] - before[j]
            relative_after = self.positions[i] - self.positions[j]
            swept_pairs.append(
                float(point_segment_distance(np.zeros(2), relative_before, relative_after) - 2 * self.config.agent_radius)
            )
        swept_pairs = np.asarray(swept_pairs, dtype=np.float64)

        outside = self.outside(self.positions)
        endpoint_wall = bool((wall <= self.config.wall_collision_margin).any() or outside.any())
        endpoint_agent = bool((pair_distances <= self.config.agent_collision_margin).any())
        wall_collision = bool((swept_wall <= self.config.wall_collision_margin).any() or outside.any())
        agent_collision = bool((swept_pairs <= self.config.agent_collision_margin).any())

        errors = np.linalg.norm(self.goals - self.positions, axis=-1)
        speeds = np.linalg.norm(u, axis=-1)
        success = bool((errors <= self.config.goal_tolerance).all())
        self.distance_history.append(errors.copy())

        window_start = self.step_count - self.config.progress_window_seconds / self.config.dt
        window_ready = window_start >= -1e-10
        window_progress = np.full(self.num_agents, np.nan)
        if window_ready:
            window_start = max(0.0, window_start)
            lo, hi = int(math.floor(window_start)), int(math.ceil(window_start))
            alpha = window_start - lo
            past = (1 - alpha) * self.distance_history[lo] + alpha * self.distance_history[hi]
            window_progress = past - errors
        max_speed = float(speeds.max())
        candidate = bool(
            window_ready
            and not success
            and np.max(np.abs(window_progress)) < self.config.progress_epsilon
            and max_speed < self.config.max_speed * self.config.speed_epsilon_fraction
        )
        if candidate:
            self.ever_candidate_deadlock = True
            if self.candidate_since is None:
                self.candidate_since = self.step_count
            self.stuck_timer = (self.step_count - self.candidate_since) * self.config.dt
        else:
            self.candidate_since = None
            self.stuck_timer = 0.0
        self.max_stuck_timer = max(self.max_stuck_timer, self.stuck_timer)
        deadlock = bool(candidate and self.stuck_timer >= self.config.deadlock_hold_seconds - 1e-10)

        for flag, attr in (
            (success, "first_success_step"),
            (deadlock, "first_deadlock_step"),
            (wall_collision, "first_wall_collision_step"),
            (agent_collision, "first_agent_collision_step"),
        ):
            if flag and getattr(self, attr) is None:
                setattr(self, attr, self.step_count)

        # Preserve Toy Give-Way first-event precedence exactly.
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

        info = dict(
            step=self.step_count,
            time=self.step_count * self.config.dt,
            positions=self.positions.copy(),
            velocities=u.copy(),
            goal_errors=errors,
            speeds=speeds,
            wall_distances=wall,
            min_swept_wall_distance=float(swept_wall.min()),
            agent_pair_indices=np.asarray(self.pair_indices, dtype=np.int64),
            agent_surface_distances=pair_distances,
            agent_surface_distance=float(pair_distances.min()),
            swept_agent_distances=swept_pairs,
            min_swept_agent_distance=float(swept_pairs.min()),
            wall_collision=wall_collision,
            agent_collision=agent_collision,
            endpoint_wall_collision=endpoint_wall,
            endpoint_agent_collision=endpoint_agent,
            outside_workspace=outside,
            task_success=success,
            window_ready=window_ready,
            window_progress=window_progress,
            candidate_deadlock=candidate,
            stuck_timer=self.stuck_timer,
            max_speed=max_speed,
            deadlock_trigger_timestep=self.first_deadlock_step if self.first_deadlock_step is not None else -1,
            deadlock=deadlock,
            termination=termination,
            tracking_error=float(np.abs(self.velocities - u).max()),
            integration_residual=float(np.abs((self.positions - before) - self.config.dt * u).max()),
        )
        progress = previous_errors - errors
        reward = np.asarray(
            progress.sum()
            + 0.01 * success
            - (wall <= self.config.wall_collision_margin).sum()
            - (pair_distances <= self.config.agent_reward_margin).sum(),
            dtype=np.float32,
        )
        return self.observation(), reward, self.done, info

    def summary(self) -> dict[str, Any]:
        success = self.first_success_step is not None
        return dict(
            ever_candidate_deadlock=self.ever_candidate_deadlock,
            max_stuck_timer=self.max_stuck_timer,
            deadlock_trigger_timestep=self.first_deadlock_step,
            success=success,
            final_goal_success=bool((np.linalg.norm(self.goals - self.positions, axis=-1) <= self.config.goal_tolerance).all()),
            collision_free_success=success
            and self.first_wall_collision_step is None
            and self.first_agent_collision_step is None,
            wall_collision=self.first_wall_collision_step is not None,
            agent_collision=self.first_agent_collision_step is not None,
            deadlock=self.first_deadlock_step is not None,
            episode_steps=self.step_count,
            first_success_step=self.first_success_step,
            first_deadlock_step=self.first_deadlock_step,
            first_wall_collision_step=self.first_wall_collision_step,
            first_agent_collision_step=self.first_agent_collision_step,
        )


# Concise alias matching the original scenario naming style.
Env = DoubleBottleneckEnv

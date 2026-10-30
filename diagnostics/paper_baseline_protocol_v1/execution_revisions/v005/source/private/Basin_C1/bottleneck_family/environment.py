"""N-agent disk dynamics with swept collisions and local stall diagnostics.

No learned controller, priority assignment, or deadlock-resolution oracle is
embedded in the plant. Local stall candidates are not certified deadlocks.
"""
from dataclasses import asdict
import hashlib
import json
import math
import numpy as np

from single_integrator.environment import point_segment_distance, segment_distance
from .scenario import Config, build_instance


class BottleneckEnv:
    def __init__(self, config=None):
        self.instance = build_instance(config)
        self.config = self.instance.config
        self.num_agents = self.config.num_agents
        self.action_shape = (self.num_agents, 2)
        self.walls = self.instance.geometry.walls
        self.pair_indices = tuple(zip(*np.triu_indices(self.num_agents, 1)))
        self._i, self._j = np.triu_indices(self.num_agents, 1)
        self.reset()

    def reset(self, positions=None, goals=None):
        p = np.array(self.instance.positions if positions is None else positions, dtype=float, copy=True)
        g = np.array(self.instance.goals if goals is None else goals, dtype=float, copy=True)
        for points in (p,g):
            if points.shape != self.action_shape or not self.instance.geometry.valid_points(points).all():
                raise ValueError('invalid start/goal geometry')
            sep = np.linalg.norm(points[self._i]-points[self._j], axis=1)
            if np.any(sep <= 2*self.config.agent_radius+self.config.agent_collision_margin):
                raise ValueError('overlapping start/goal disks')
        self.positions, self.goals = p, g
        self.velocities = np.zeros(self.action_shape)
        self.step_count = 0
        self.done = False
        self.termination = 'running'
        self.collided = False
        self.history = [p.copy()]
        return self.observation()

    def snapshot(self):
        return dict(positions=self.positions.copy(), goals=self.goals.copy(),
                    last_applied_velocity=self.velocities.copy(), walls=self.walls.copy(),
                    pair_indices=self.pair_indices, config=asdict(self.config), step=self.step_count)

    def initial_state_sha256(self):
        """Identity of the full reset state, including velocity and clock."""
        if self.step_count != 0:
            raise RuntimeError('initial-state hash must be taken before the first step')
        state = dict(config=asdict(self.config), positions=self.positions.tolist(),
                     goals=self.goals.tolist(), velocities=self.velocities.tolist(),
                     step_count=self.step_count)
        return hashlib.sha256(json.dumps(state, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()

    def observation(self):
        # Set-shaped physical inputs; no priority, route, agent ID or mode labels.
        return dict(agents=np.concatenate((self.positions, self.velocities,
                                           self.goals-self.positions), axis=1),
                    walls=self.walls.copy(),
                    rectangles=self.instance.geometry.rectangles.copy())

    def step(self, velocity, *, diagnose_stalls=True):
        if self.done:
            raise RuntimeError('episode already terminated; reset before stepping')
        u = np.asarray(velocity, dtype=float)
        if u.shape != self.action_shape or not np.isfinite(u).all():
            raise ValueError('velocity must be finite [N,2]')
        if np.max(np.linalg.norm(u, axis=1)) > self.config.max_speed + 1e-10:
            raise ValueError('speed exceeds max_speed')
        c = self.config
        before = self.positions.copy()
        after = before+c.dt*u
        wall_clearance = segment_distance(before[:,None], after[:,None],
            self.walls[None,:,0], self.walls[None,:,1]).min()-self.instance.geometry.margin
        relative_before = before[self._i]-before[self._j]
        relative_after = after[self._i]-after[self._j]
        pair_clearance = point_segment_distance(np.zeros(2), relative_before, relative_after).min()
        pair_clearance -= 2*c.agent_radius+c.agent_collision_margin
        collision = bool(wall_clearance <= 0 or pair_clearance <= 0 or
                         not self.instance.geometry.valid_points(after).all())
        self.positions, self.velocities = after, u.copy()
        self.step_count += 1
        self.history.append(after.copy())
        self.collided |= collision
        success = bool(np.all(np.linalg.norm(self.goals-after, axis=1) <= c.goal_tolerance))
        self.termination = ('collision' if collision else 'success' if success else
                            'timeout' if self.step_count >= c.max_steps else 'running')
        self.done = self.termination != 'running'
        info = dict(step=self.step_count, termination=self.termination,
                    collision=collision, task_success=success and not self.collided,
                    swept_clearance=float(min(wall_clearance,pair_clearance)),
                    min_swept_wall_clearance=float(wall_clearance),
                    min_swept_agent_clearance=float(pair_clearance),
                    positions=after.copy(), velocities=u.copy(),
                    # This is deliberately not called a deadlock event.
                    stalled_groups=self.stalled_groups() if diagnose_stalls else [])
        return self.observation(), self.done, info

    def stalled_groups(self, window_seconds=5., displacement_epsilon=.02, interaction_distance=.8):
        """Local persistent low-motion clusters; queues and deliberate waits may qualify.

        A moving robot elsewhere cannot hide the cluster. A full displacement
        window avoids mistaking a periodic motion's coincident endpoints for rest.
        This diagnostic supplies candidates for a separate deadlock adjudication.
        """
        if (not all(math.isfinite(v) and v > 0 for v in
                    (window_seconds, displacement_epsilon, interaction_distance))):
            raise ValueError('stall diagnostic thresholds must be finite and positive')
        count = max(1, math.ceil(window_seconds/self.config.dt))
        if len(self.history) <= count:
            return []
        window = np.asarray(self.history[-count-1:])
        motion = np.linalg.norm(window-window[0], axis=2).max(axis=0)
        quiet = motion <= displacement_epsilon
        pending = np.linalg.norm(self.goals-self.positions, axis=1) > self.config.goal_tolerance
        separation = np.linalg.norm(self.positions[:,None]-self.positions[None,:], axis=2)
        edges = (separation <= interaction_distance) & quiet[:,None] & quiet[None,:]
        remaining = set(np.flatnonzero(quiet).tolist()); groups = []
        while remaining:
            seed = remaining.pop(); group = {seed}; stack = [seed]
            while stack:
                i = stack.pop()
                connected = set(np.flatnonzero(edges[i]).tolist()) & remaining
                remaining -= connected; group |= connected; stack.extend(connected)
            if len(group) >= 2 and any(pending[list(group)]):
                groups.append(sorted(group))
        return sorted(groups)

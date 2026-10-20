"""User-authorized single-robot seam; never used in multi-robot evidence."""
from dataclasses import asdict
import numpy as np
from single_integrator.environment import segment_distance
from bottleneck_family.scenario import Config
from geometry import GeneratedEnv

class SingleAgentConfig(Config):
    def __post_init__(self):
        if self.num_agents != 1:
            raise ValueError('single-agent environment only')
        values=asdict(self); values['num_agents']=2
        Config(**values)

class SingleAgentGeneratedEnv(GeneratedEnv):
    def __init__(self, geometry, positions, goals, config=None):
        values=asdict(config or geometry.config);values['num_agents']=1
        super().__init__(geometry, positions, goals, SingleAgentConfig(**values))

    def stalled_groups(self, *args, **kwargs):
        return []

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
        collision = bool(wall_clearance <= 0 or
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
                    swept_clearance=float(wall_clearance),
                    min_swept_wall_clearance=float(wall_clearance),
                    positions=after.copy(), velocities=u.copy(),
                    # This is deliberately not called a deadlock event.
                    stalled_groups=self.stalled_groups() if diagnose_stalls else [])
        return self.observation(), self.done, info

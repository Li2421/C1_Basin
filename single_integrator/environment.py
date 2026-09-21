"""Deterministic p[k+1] = p[k] + dt*u[k]; collisions never modify motion."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    schema: str = 'giveway_single_integrator_v1'
    dt: float = 0.05
    max_steps: int = 850
    max_speed: float = 0.5
    agent_radius: float = 0.16
    corridor_width: float = 0.4
    bay_top: float = 0.68
    corridor_half_length: float = 2.5
    wall_layout_scale: float = 1.0
    # Retain the legacy effective wall geometry and reporting thresholds.
    wall_radius: float = 4 / 600
    wall_collision_margin: float = 0.005
    agent_collision_margin: float = 0.0
    agent_reward_margin: float = 0.005
    goal_tolerance: float = 0.08
    # Deprecated fields retained only to read historical metadata.
    slow_speed: float = 0.05
    congestion_goal_error: float = 0.10
    deadlock_seconds: float = 1.0
    detection_protocol: str = "window_progress_v2"
    progress_window_seconds: float = 2.0
    deadlock_hold_seconds: float = 5.0
    progress_epsilon: float = 0.01
    speed_epsilon_fraction: float = 0.05
    terminate_on_collision: bool = True
    terminate_on_success: bool = True
    terminate_on_deadlock: bool = True

    def __post_init__(self):
        if self.schema != 'giveway_single_integrator_v1':
            raise ValueError('Unknown dynamics schema')
        if not 0 < self.corridor_width < .64 or not self.bay_top > self.corridor_width/2:
            raise ValueError('Geometry must preserve the give-way bottleneck')
        if not math.isfinite(self.corridor_half_length) or self.corridor_half_length < 1.3:
            raise ValueError('Corridor half length must be finite and >= 1.3 m')
        if min(self.dt, self.max_speed, self.agent_radius, self.max_steps, self.deadlock_seconds) <= 0:
            raise ValueError('Dynamics and horizon must be positive')

        if not math.isfinite(self.wall_layout_scale) or self.wall_layout_scale < 1.:
            raise ValueError("Wall layout scale must be finite and >= 1")
        if self.detection_protocol != "window_progress_v2":
            raise ValueError("Unknown detection protocol")
        if not all(math.isfinite(x) and x > 0 for x in [self.progress_window_seconds, self.deadlock_hold_seconds, self.progress_epsilon, self.speed_epsilon_fraction]):
            raise ValueError("Deadlock parameters must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def bounded_nominal(action, max_speed):
    """Policy interface projection, performed BEFORE a post-hoc safety filter."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2, 2) or not np.isfinite(action).all():
        raise ValueError('Expected finite joint velocity [2,2]')
    norm = np.linalg.norm(action, axis=-1, keepdims=True)
    return action * np.minimum(1., max_speed / np.maximum(norm, 1e-30))


def point_segment_distance(p, a, b):
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    d = b - a
    t = np.clip(np.sum((p-a)*d, axis=-1) / np.maximum(np.sum(d*d, axis=-1), 1e-30), 0., 1.)
    return np.linalg.norm(p - (a + t[..., None]*d), axis=-1)


def segment_distance(a, b, c, d):
    """Exact 2-D distance, vectorized over leading axes, including intersections."""
    a, b, c, d = map(np.asarray, (a, b, c, d))
    r, s, q = b-a, d-c, c-a
    cross = lambda x, y: x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denom = cross(r, s)
    safe = np.where(np.abs(denom)>1e-15, denom, 1.)
    t, u = cross(q, s)/safe, cross(q, r)/safe
    intersect = (np.abs(denom)>1e-15) & (t>=0) & (t<=1) & (u>=0) & (u<=1)
    dist = np.minimum.reduce([point_segment_distance(a,c,d), point_segment_distance(b,c,d), point_segment_distance(c,a,b), point_segment_distance(d,a,b)])
    return np.where(intersect, 0., dist)


class AgentView:
    def __init__(self, env, index):
        self.env, self.index = env, index

    @property
    def position(self):
        return self.env.positions[self.index]

    @property
    def velocity(self):
        return self.env.velocities[self.index]

    @property
    def goal(self):
        return self.env.goals[self.index]


class GiveWayEnv:
    def __init__(self, config=None):
        self.config = config or Config()
        w = self.config.corridor_width/2
        top = self.config.bay_top
        length = self.config.corridor_half_length
        self.wall_names = ('left_end', 'right_end', 'floor', 'left_ceiling', 'right_ceiling', 'bay_left', 'bay_right', 'bay_top')
        self.walls = np.array([
            [[-length,-w],[-length,w]], [[length,-w],[length,w]],
            [[-length,-w],[length,-w]], [[-length,w],[-w,w]], [[w,w],[length,w]],
            [[-w,w],[-w,top]], [[w,w],[w,top]], [[-w,top],[w,top]],
        ], dtype=np.float64)
        self.walls *= self.config.wall_layout_scale
        self.goals = np.array([[length-.21,0.],[-length+.21,0.]])
        self.agents = [AgentView(self,i) for i in range(2)]
        self.reset()

    def observation(self):
        rows = []
        for i in range(2):
            j = 1-i
            rows.append(np.concatenate([self.positions[i], self.velocities[i], self.goals[i]-self.positions[i], self.positions[j]-self.positions[i], self.velocities[j]-self.velocities[i]]))
        return np.asarray(rows, dtype=np.float32)

    def outside(self, positions):
        p = np.asarray(positions);scale = self.config.wall_layout_scale;w = self.config.corridor_width/2*scale
        corridor = (np.abs(p[:,0])<=self.config.corridor_half_length*scale)&(p[:,1]>=-w)&(p[:,1]<=w)
        bay = (np.abs(p[:,0])<=w)&(p[:,1]>=w)&(p[:,1]<=self.config.bay_top*scale)
        return ~(corridor|bay)

    def distances(self, positions=None):
        p = self.positions if positions is None else np.asarray(positions)
        wall = point_segment_distance(p[:,None,:], self.walls[None,:,0,:], self.walls[None,:,1,:])-self.config.agent_radius-self.config.wall_radius
        agent = np.linalg.norm(p[0]-p[1])-2*self.config.agent_radius
        return wall, float(agent)

    def reset(self, positions=None):
        start=min(2.,self.config.corridor_half_length-.45)
        self.positions = np.array([[-start,0.],[start,0.]] if positions is None else positions, dtype=np.float64, copy=True)
        if self.positions.shape!=(2,2) or not np.isfinite(self.positions).all():
            raise ValueError('Invalid initial positions')
        wall, agent = self.distances()
        if self.outside(self.positions).any() or wall.min()<=self.config.wall_collision_margin or agent<=self.config.agent_collision_margin:
            raise ValueError('Initial condition must be safe')
        self.velocities = np.zeros((2,2), dtype=np.float64)
        self.step_count = 0
        self.distance_history = [np.linalg.norm(self.goals-self.positions, axis=-1)]
        self.candidate_since = None
        self.stuck_timer = self.max_stuck_timer = 0.0
        self.ever_candidate_deadlock = False
        self.first_success_step = self.first_deadlock_step = None
        self.first_wall_collision_step = self.first_agent_collision_step = None
        self.done = False
        return self.observation()

    def snapshot(self):
        """Detached state for post-hoc filters; no mutable plant object is passed."""
        return dict(positions=self.positions.copy(), last_applied_velocity=self.velocities.copy(), goals=self.goals.copy(), walls=self.walls.copy(), config=self.config.to_dict(), step=self.step_count)

    def step(self, executed_velocity):
        if self.done:
            raise RuntimeError('Reset a terminated episode before stepping')
        u = np.array(executed_velocity, dtype=np.float64, copy=True)
        if u.shape!=(2,2) or not np.isfinite(u).all():
            raise ValueError('Invalid executed velocity')
        if np.any(np.linalg.norm(u,axis=-1)>self.config.max_speed+1e-9):
            raise ValueError('Executed action exceeds speed bound; no hidden plant clipping is allowed')
        before = self.positions.copy()
        previous_errors = np.linalg.norm(self.goals-before,axis=-1)
        self.positions = before + self.config.dt*u
        self.velocities = u.copy()  # Bookkeeping of applied velocity, not an inertial state.
        self.step_count += 1
        wall, agent_distance = self.distances()
        swept_wall = segment_distance(before[:,None,:],self.positions[:,None,:],self.walls[None,:,0,:],self.walls[None,:,1,:])-self.config.agent_radius-self.config.wall_radius
        rel_before = before[0]-before[1];rel_after = self.positions[0]-self.positions[1]
        swept_agent = float(point_segment_distance(np.zeros(2),rel_before,rel_after)-2*self.config.agent_radius)
        outside = self.outside(self.positions)
        endpoint_wall = bool((wall<=self.config.wall_collision_margin).any() or outside.any())
        endpoint_agent = bool(agent_distance<=self.config.agent_collision_margin)
        wall_collision = bool((swept_wall<=self.config.wall_collision_margin).any() or outside.any())
        agent_collision = bool(swept_agent<=self.config.agent_collision_margin)
        errors = np.linalg.norm(self.goals-self.positions,axis=-1)
        speeds = np.linalg.norm(u,axis=-1)
        success = bool((errors<=self.config.goal_tolerance).all())
        self.distance_history.append(errors.copy())
        # Interpolate distance samples for an exact Tw, including noninteger Tw/dt.
        window_start = self.step_count - self.config.progress_window_seconds/self.config.dt
        window_ready = window_start >= -1e-10
        window_progress = np.full(2, np.nan)
        if window_ready:
            window_start = max(0., window_start)
            lo = int(math.floor(window_start)); hi = int(math.ceil(window_start))
            alpha = window_start-lo
            past = (1-alpha)*self.distance_history[lo]+alpha*self.distance_history[hi]
            window_progress = past-errors
        max_speed = float(speeds.max())
        candidate = bool(window_ready and not success
                         and np.max(np.abs(window_progress)) < self.config.progress_epsilon
                         and max_speed < self.config.max_speed*self.config.speed_epsilon_fraction)
        if candidate:
            self.ever_candidate_deadlock = True
            if self.candidate_since is None:
                self.candidate_since = self.step_count
            self.stuck_timer = (self.step_count-self.candidate_since)*self.config.dt
        else:
            self.candidate_since = None
            self.stuck_timer = 0.
        self.max_stuck_timer = max(self.max_stuck_timer, self.stuck_timer)
        deadlock = candidate and self.stuck_timer >= self.config.deadlock_hold_seconds-1e-10
        for flag, attr in [(success,'first_success_step'),(deadlock,'first_deadlock_step'),(wall_collision,'first_wall_collision_step'),(agent_collision,'first_agent_collision_step')]:
            if flag and getattr(self,attr) is None:
                setattr(self,attr,self.step_count)
        termination = 'running'
        if (wall_collision or agent_collision) and self.config.terminate_on_collision:termination='collision'
        elif success and self.config.terminate_on_success:termination='success'
        elif deadlock and self.config.terminate_on_deadlock:termination='deadlock'
        elif self.step_count>=self.config.max_steps:termination='timeout'
        self.done = termination!='running'
        info = dict(step=self.step_count,time=self.step_count*self.config.dt,positions=self.positions.copy(),velocities=u.copy(),goal_errors=errors,speeds=speeds,wall_distances=wall,min_swept_wall_distance=float(swept_wall.min()),agent_surface_distance=agent_distance,min_swept_agent_distance=swept_agent,wall_collision=wall_collision,agent_collision=agent_collision,endpoint_wall_collision=endpoint_wall,endpoint_agent_collision=endpoint_agent,outside_workspace=outside,task_success=success,window_ready=window_ready,window_progress=window_progress,candidate_deadlock=candidate,stuck_timer=self.stuck_timer,max_speed=max_speed,deadlock_trigger_timestep=self.first_deadlock_step if self.first_deadlock_step is not None else -1,deadlock=bool(deadlock),termination=termination,tracking_error=float(np.abs(self.velocities-u).max()),integration_residual=float(np.abs((self.positions-before)-self.config.dt*u).max()))
        # Preserve the legacy shared progress and per-agent geometric penalties.
        progress = previous_errors-errors
        reward = np.asarray(progress.sum()+.01*success-(wall<=self.config.wall_collision_margin).sum(axis=1)-(agent_distance<=self.config.agent_reward_margin), dtype=np.float32)
        return self.observation(), reward, self.done, info

    def summary(self):
        success = self.first_success_step is not None
        return dict(ever_candidate_deadlock=self.ever_candidate_deadlock,max_stuck_timer=self.max_stuck_timer,deadlock_trigger_timestep=self.first_deadlock_step,success=success,final_goal_success=bool((np.linalg.norm(self.goals-self.positions,axis=-1)<=self.config.goal_tolerance).all()),collision_free_success=success and self.first_wall_collision_step is None and self.first_agent_collision_step is None,wall_collision=self.first_wall_collision_step is not None,agent_collision=self.first_agent_collision_step is not None,deadlock=self.first_deadlock_step is not None,episode_steps=self.step_count,first_success_step=self.first_success_step,first_deadlock_step=self.first_deadlock_step,first_wall_collision_step=self.first_wall_collision_step,first_agent_collision_step=self.first_agent_collision_step)

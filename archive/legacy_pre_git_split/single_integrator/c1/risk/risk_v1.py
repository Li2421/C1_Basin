"""R_risk_v1: soft TRO geometry plus persistent task-stall risk for GiveWay.

The cone/activity calculation stays in :mod:`soft_activity`.  This module only
adds a modular task potential and combines its windowed progress signal with
that already-validated geometric component.
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from .soft_activity import SoftRiskConfig
from .risk_function import trajectory_risk


@dataclass(frozen=True)
class RiskV1Config(SoftRiskConfig):
    # v1_1 follows the environment's per-agent goal criterion. The old sum
    # criterion is retained solely for replaying explicitly versioned v1 runs.
    unfinished_mode: str = 'per_agent'
    window_seconds: float = 4.0
    delta_prog: float = .02
    tau_prog: float = .005
    task_distance_epsilon: float = 1e-5
    task_mask_temperature: float = .02
    goal_tolerance: float = .08

    def __post_init__(self):
        super().__post_init__()
        if self.unfinished_mode not in ('per_agent', 'joint_sum'):
            raise ValueError('unfinished_mode must be per_agent or joint_sum')
        values = (self.window_seconds, self.delta_prog, self.tau_prog,
                  self.task_distance_epsilon, self.task_mask_temperature, self.goal_tolerance)
        if any(not np.isfinite(x) or x <= 0 for x in values):
            raise ValueError('v1 progress parameters must be finite and positive')

    def window_steps(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        steps = int(round(self.window_seconds / dt))
        if steps < 1 or not np.isclose(steps * dt, self.window_seconds, rtol=0., atol=1e-10):
            raise ValueError('window_seconds must be an integer multiple of dt')
        return steps


def task_potential(positions, goals, initial_positions, config=None):
    """Smooth normalized joint goal-distance potential, shape ``[...,]``.

    ``sqrt(||p-g||^2+e^2)-e`` is smooth at a goal while retaining zero there.
    The fixed initial denominator anchors the trajectory at approximately one.
    """
    config = config or RiskV1Config()
    positions, goals, initial_positions = map(jnp.asarray, (positions, goals, initial_positions))
    def distance(x):
        return jnp.sqrt(jnp.sum((x-goals)**2, axis=-1)+config.task_distance_epsilon**2)-config.task_distance_epsilon
    total = jnp.sum(distance(positions), axis=-1)
    initial = jnp.sum(distance(initial_positions), axis=-1)
    return total / (initial + config.task_distance_epsilon)


@partial(jax.jit, static_argnames=('config', 'window_steps'))
def trajectory_risk_v1(cone_risk_t, positions, goals, config=None, window_steps=None):
    """Compute all v1 components for a batch of equal-length trajectories.

    ``positions`` has T+1 states and ``cone_risk_t`` has T controls/states.
    Only indices t >= W contribute to the trajectory aggregate.
    """
    config = config or RiskV1Config()
    c = jnp.asarray(cone_risk_t)
    positions, goals = jnp.asarray(positions), jnp.asarray(goals)
    if c.ndim == 1:
        c = c[None]
    if positions.ndim == 3:
        positions = positions[None]
    if c.ndim != 2 or positions.ndim != 4 or positions.shape[:2] != (c.shape[0], c.shape[1]+1):
        raise ValueError('expected cone risk [B,T] and positions [B,T+1,2,2]')
    W = config.window_steps(.05) if window_steps is None else window_steps
    if not isinstance(W, int) or W < 1 or W >= c.shape[1]:
        raise ValueError('window must have 1 <= W < trajectory horizon')
    initial = positions[:, 0]
    potential = task_potential(positions, goals, initial[:, None], config)
    # The first W entries have no complete physical-time history and are masked.
    potential_t = potential[:, :-1]
    progress = potential_t[:, :-W] - potential_t[:, W:]
    distances = jnp.sqrt(jnp.sum((positions[:, W:-1]-goals)**2, axis=-1)+config.task_distance_epsilon**2)-config.task_distance_epsilon
    if config.unfinished_mode == 'joint_sum':
        unfinished_mask = jax.nn.sigmoid((jnp.sum(distances, axis=-1)-2.*config.goal_tolerance)/config.task_mask_temperature)
    else:
        # Smooth OR of per-agent unfinished indicators: one agent reaching
        # its goal cannot hide the other agent's remaining goal error.
        unfinished = jax.nn.sigmoid((distances-config.goal_tolerance)/config.task_mask_temperature)
        a, b = unfinished[..., 0], unfinished[..., 1]
        unfinished_mask = a+b-a*b
    stall = unfinished_mask * jax.nn.sigmoid((config.delta_prog-progress)/config.tau_prog)
    c_valid = c[:, W:]
    total = c_valid+stall-c_valid*stall
    aggregate = trajectory_risk(total, config)
    # Keep full-length fields aligned with control time for logging; warmup is NaN.
    pad = jnp.full((c.shape[0], W), jnp.nan, c.dtype)
    return dict(trajectory_risk=aggregate, cone_risk_t=c, task_potential=potential_t,
                progress_delta=jnp.concatenate((pad, progress), axis=-1),
                unfinished_mask=jnp.concatenate((pad, unfinished_mask), axis=-1),
                stall_risk_t=jnp.concatenate((pad, stall), axis=-1),
                total_risk_t=jnp.concatenate((pad, total), axis=-1), valid_mask=jnp.arange(c.shape[1])[None] >= W)

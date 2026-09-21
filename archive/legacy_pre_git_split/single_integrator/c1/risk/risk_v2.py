"""Candidate full-episode risk: interval-aligned progress and terminal liveness.

This is a versioned research revision, not a proof of liveness. The cone
calculation and constrained optimization remain unchanged. Legacy v1 risk
definitions remain independently replayable.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from .risk_v1 import RiskV1Config, task_potential
from .risk_function import trajectory_risk


@dataclass(frozen=True)
class RiskV2Config(RiskV1Config):
    terminal_distance_scale: float = .1
    terminal_weight: float = .75
    terminal_goal_weight: float = .75
    terminal_failure_floor: float = .5
    composition_revision: str = 'balanced_terminal_v3'

    def __post_init__(self):
        super().__post_init__()
        if self.unfinished_mode != 'per_agent':
            raise ValueError('v2 requires per-agent unfinished semantics')
        if not np.isfinite(self.terminal_distance_scale) or self.terminal_distance_scale<=0:
            raise ValueError('terminal_distance_scale must be positive')
        if not np.isfinite(self.terminal_weight) or not 0<self.terminal_weight<1:
            raise ValueError('terminal_weight must lie strictly between zero and one')
        if not np.isfinite(self.terminal_goal_weight) or not 0<self.terminal_goal_weight<1:
            raise ValueError('terminal_goal_weight must lie strictly between zero and one')
        if not np.isfinite(self.terminal_failure_floor) or not 0<self.terminal_failure_floor<1:
            raise ValueError('terminal_failure_floor must lie strictly between zero and one')
        if self.terminal_weight*self.terminal_goal_weight*self.terminal_failure_floor<=1-self.terminal_weight:
            raise ValueError('terminal failure floor must exceed the maximum successful exposure cost')
        if self.composition_revision!='balanced_terminal_v3':
            raise ValueError('unsupported terminal risk composition')


def trajectory_risk_v2(cone_risk_t, positions, goals, config, step_mask, terminal_codes, dt):
    c=jnp.asarray(cone_risk_t)
    positions,goals=jnp.asarray(positions),jnp.asarray(goals)
    mask=jax.lax.stop_gradient(jnp.asarray(step_mask,bool))
    codes=jax.lax.stop_gradient(jnp.asarray(terminal_codes))
    if c.ndim!=2 or positions.shape!=(c.shape[0],c.shape[1]+1,2,2) or mask.shape!=c.shape or codes.shape!=(len(c),):
        raise ValueError('expected [B,T] risk/mask, [B,T+1,2,2] positions and [B] terminal codes')
    W=config.window_steps(dt)
    potential=task_potential(positions,goals,positions[:,0,None],config)
    endpoints=jnp.arange(c.shape[1])+1
    past=jnp.maximum(endpoints-W,0)
    progress=potential[:,past]-potential[:,1:]
    distance=jnp.sqrt(jnp.sum((positions[:,1:]-goals)**2,axis=-1)+config.task_distance_epsilon**2)-config.task_distance_epsilon
    unfinished_agent=jax.nn.sigmoid((distance-config.goal_tolerance)/config.task_mask_temperature)
    a,b=unfinished_agent[...,0],unfinished_agent[...,1]
    unfinished=a+b-a*b
    progress_mask=mask & (endpoints[None]>=W)
    stall=jnp.where(progress_mask,unfinished*jax.nn.sigmoid((config.delta_prog-progress)/config.tau_prog),0.)
    total=c+stall-c*stall
    exposure=trajectory_risk(total,config,step_mask=mask)
    lengths=jnp.sum(mask,axis=1).astype(jnp.int32)
    rows=jnp.arange(len(c));last=jnp.maximum(lengths-1,0)
    final_errors=distance[rows,last]
    excess=jnp.max(jax.nn.relu(final_errors-config.goal_tolerance),axis=-1)
    goal_risk=excess/(excess+config.terminal_distance_scale)
    failed=(codes>=2)
    # Keep a failing final window from being diluted by a long easy prefix.
    terminal_stall=jnp.where(failed,stall[rows,last],0.)
    # A terminal failure is an actual discrete environment event. The floor
    # prevents near-tolerance timeouts from looking better than successes;
    # the remaining distance term retains a useful piecewise state gradient.
    terminal_goal=jnp.where(failed,config.terminal_failure_floor+
                            (1-config.terminal_failure_floor)*goal_risk,0.)
    # OR composition at either level can erase all goal derivatives when the
    # other component saturates. Convex combinations preserve both signals.
    terminal=config.terminal_goal_weight*terminal_goal+(1-config.terminal_goal_weight)*terminal_stall
    terminal=jnp.where((codes==2)|(codes==3),1.,terminal)
    result=(1-config.terminal_weight)*exposure+config.terminal_weight*terminal
    result=jnp.where((codes==2)|(codes==3),1.,result)
    return dict(trajectory_risk=result, exposure_risk=exposure,
                terminal_risk=terminal,terminal_goal_risk=terminal_goal,
                terminal_stall_risk=terminal_stall,cone_risk_t=c,
                task_potential=potential[:,1:],progress_delta=progress,
                unfinished_mask=unfinished,stall_risk_t=stall,total_risk_t=total,
                valid_mask=mask,progress_valid_mask=progress_mask,
                episode_lengths=lengths,terminal_codes=codes,censored=codes==0)

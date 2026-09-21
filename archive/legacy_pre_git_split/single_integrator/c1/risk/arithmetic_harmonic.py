"""Experimental sign-sound arithmetic/harmonic temporal event robustness.

Own candidate, inspired by average-based temporal robustness. No controller or
event changes. This module is not integrated into shared-policy training.
"""
import math
import jax.numpy as jnp


def conjunction(values, mask=None, axis=-1):
    """Positive harmonic mean if all predicates positive; negative mean otherwise.

    Zero is the event boundary. Empty reductions return zero. Invalid entries
    are excluded from both the sign decision and normalization.
    """
    values=jnp.asarray(values)
    mask=jnp.ones_like(values,dtype=bool) if mask is None else jnp.broadcast_to(mask,values.shape)
    count=jnp.sum(mask,axis=axis)
    all_positive=jnp.all(~mask | (values>0),axis=axis)
    safe=jnp.where(mask & (values>0),values,1.)
    minimum=jnp.min(jnp.where(mask,safe,jnp.inf),axis=axis,keepdims=True)
    any_valid=jnp.any(mask,axis=axis,keepdims=True)
    minimum=jnp.where(any_valid,minimum,1.)
    denominator=jnp.sum(jnp.where(mask,minimum/safe,0.),axis=axis)
    harmonic=jnp.maximum(count,1)*jnp.squeeze(minimum,axis=axis)/jnp.where(denominator>0,denominator,1.)
    negative=jnp.sum(jnp.where(mask,jnp.minimum(values,0.),0.),axis=axis)/jnp.maximum(count,1)
    return jnp.where(count>0,jnp.where(all_positive,harmonic,negative),0.)


def disjunction(values, mask=None, axis=-1):
    return -conjunction(-jnp.asarray(values),mask,axis)


def event_upper(robustness,scale=.02):
    """R>=1 iff eta>=0; positive events have R>1, negative events R<1.

    A C1 scalar generating function; aggregation and physical rollout remain
    only piecewise differentiable. No independently substituted backward rule.
    """
    if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite scale required')
    z=jnp.asarray(robustness)/scale
    return jnp.where(z>=0,1+z,jnp.exp(jnp.minimum(z,0.)))


def temporal_event(margins,valid,*,hold_samples,scale=.02):
    """Conjunction across all predicates/hold samples, disjunction over windows.

    Positive robustness iff some complete window has all raw margins positive.
    Finite margins are normalized monotonically to (-1,1), preserving signs.
    """
    margins=jnp.asarray(margins);valid=jnp.asarray(valid,dtype=bool)
    if margins.ndim!=2 or valid.shape!=margins.shape[:1]:raise ValueError('invalid temporal shapes')
    if not isinstance(hold_samples,int) or hold_samples<1:raise ValueError('positive hold_samples required')
    if len(margins)<hold_samples:
        zero=jnp.sum(margins)*0.
        return dict(robustness=zero,J_live=zero,eligible=jnp.array(False),complete_windows=jnp.array(0))
    normalized=margins/(1+jnp.abs(margins))
    indices=jnp.arange(len(margins)-hold_samples+1)[:,None]+jnp.arange(hold_samples)
    enabled=jnp.all(valid[indices],axis=1)
    window_score=conjunction(normalized[indices],axis=(1,2))
    eta=disjunction(window_score,enabled)
    eligible=jnp.any(enabled)
    return dict(robustness=eta,J_live=jnp.where(eligible,event_upper(eta,scale),0.),
                eligible=eligible,complete_windows=enabled.sum())


def trajectory(before,after,applied,goals,alive,*,terminal_timeout,dt,max_speed,
               goal_tolerance,hold_seconds,progress_window_seconds,
               progress_epsilon,speed_epsilon_fraction,scale=.02):
    """Original strict deadlock OR original terminal stalled-timeout event.

    The first-event mask is supplied by the unchanged physical evaluator.
    All comparisons and window endpoints match the existing event protocol.
    """
    constants=(dt,max_speed,goal_tolerance,hold_seconds,progress_window_seconds,
               progress_epsilon,speed_epsilon_fraction)
    if not all(math.isfinite(v) and v>0 for v in constants):raise ValueError('positive detector parameters required')
    before,after,applied,goals,alive=map(jnp.asarray,(before,after,applied,goals,alive))
    if before.ndim!=3 or before.shape!=after.shape or goals.shape!=before.shape[1:]:
        raise ValueError('expected matching [time,agents,dimension] trajectories')
    if len(before)==0 or alive.shape!=(len(before),):raise ValueError('nonempty trace and matching alive required')
    norm=lambda x:jnp.sqrt(jnp.maximum(jnp.sum(x*x,axis=-1),1e-30))
    distances=norm(jnp.concatenate([before[:1],after],axis=0)-goals)
    end=jnp.arange(1,len(before)+1)
    start=jnp.maximum(0.,end-progress_window_seconds/dt)
    low,high=jnp.floor(start).astype(int),jnp.ceil(start).astype(int)
    past=(1-(start-low))[:,None]*distances[low]+(start-low)[:,None]*distances[high]
    speeds=norm(applied.reshape(after.shape))
    margins=jnp.stack([(jnp.max(distances[1:],axis=-1)-goal_tolerance)/goal_tolerance,
        1-jnp.max(jnp.abs(past-distances[1:]),axis=-1)/progress_epsilon,
        1-jnp.max(speeds,axis=-1)/(max_speed*speed_epsilon_fraction)],axis=-1)
    valid=alive & (end>=progress_window_seconds/dt-1e-10)
    strict=temporal_event(margins,valid,
        hold_samples=int(math.ceil(hold_seconds/dt-1e-10))+1,scale=scale)
    steps=int(round(2./dt));count=jnp.sum(alive)
    last=jnp.clip(count-1,0,len(after)-1);first=jnp.clip(count-steps,0,len(after)-1)
    selected=alive & (jnp.arange(len(after))>=count-steps)
    speed_margins=(1-speeds/.05).reshape(-1)
    progress_margins=1-jnp.abs(distances[1:][last]-distances[1:][first])/.02
    stall_margins=jnp.concatenate([speed_margins,progress_margins])
    stall_mask=jnp.concatenate([jnp.repeat(selected,after.shape[1]),jnp.ones(after.shape[1],bool)])
    stalled=conjunction(stall_margins/(1+jnp.abs(stall_margins)),stall_mask)
    stall_enabled=jnp.asarray(terminal_timeout,dtype=bool) & (count>=steps)
    enabled=jnp.stack([strict['eligible'],stall_enabled])
    eta=disjunction(jnp.stack([strict['robustness'],stalled]),enabled)
    eligible=jnp.any(enabled)
    return dict(J_live=jnp.where(eligible,event_upper(eta,scale),0.),robustness=eta,
        eligible=eligible,strict_robustness=strict['robustness'],stalled_robustness=stalled,
        strict_eligible=strict['eligible'],stalled_eligible=stall_enabled)

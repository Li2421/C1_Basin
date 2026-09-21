"""First-event-equivalent margin on a fixed-horizon, uncensored model trajectory.

Historical success guards prevent events after task completion from counting;
historical deadlock witnesses cannot be erased by later model recovery.
"""
import math
import jax
import jax.numpy as jnp


def event_margin(unfinished,stagnant,slow,ready,terminal_stall,*,hold_samples):
    unfinished,stagnant,slow,ready=map(jnp.asarray,(unfinished,stagnant,slow,ready))
    n=len(unfinished)
    if not n or any(x.shape!=(n,) for x in (unfinished,stagnant,slow,ready)):
        raise ValueError('matching nonempty one-dimensional predicate traces required')
    if not isinstance(hold_samples,int) or hold_samples<1:raise ValueError('positive hold_samples required')
    history=jax.lax.associative_scan(jnp.minimum,unfinished)
    if n>=hold_samples:
        indices=jnp.arange(n-hold_samples+1)[:,None]+jnp.arange(hold_samples)
        windows=jnp.min(jnp.stack([unfinished,stagnant,slow],axis=-1)[indices],axis=(1,2))
        enabled=jnp.all(ready[indices],axis=1)
        strict=jnp.max(jnp.where(enabled,jnp.minimum(windows,history[hold_samples-1:]),-jnp.inf))
    else:
        strict=jnp.array(-jnp.inf)
    stalled=jnp.minimum(terminal_stall,history[-1])
    return dict(robustness=jnp.maximum(strict,stalled),strict_margin=strict,stalled_margin=stalled)


def trajectory(before,after,applied,goals,*,dt,max_speed,goal_tolerance,hold_seconds,
               progress_window_seconds,progress_epsilon,speed_epsilon_fraction,temperature=.1):
    constants=(dt,max_speed,goal_tolerance,hold_seconds,progress_window_seconds,
               progress_epsilon,speed_epsilon_fraction,temperature)
    if not all(math.isfinite(x) and x>0 for x in constants):raise ValueError('positive finite configuration required')
    before,after,applied,goals=map(jnp.asarray,(before,after,applied,goals))
    if before.ndim!=3 or before.shape!=after.shape or goals.shape!=before.shape[1:]:
        raise ValueError('matching full model trajectories required')
    n=len(before)
    if not n:raise ValueError('empty trajectory')
    norm=lambda x:jnp.sqrt(jnp.maximum(jnp.sum(x*x,axis=-1),1e-30))
    errors=norm(jnp.concatenate([before[:1],after],axis=0)-goals)
    speed=norm(applied.reshape(after.shape))
    end=jnp.arange(1,n+1);start=jnp.maximum(0.,end-progress_window_seconds/dt)
    lo,hi=jnp.floor(start).astype(int),jnp.ceil(start).astype(int)
    past=(1-(start-lo))[:,None]*errors[lo]+(start-lo)[:,None]*errors[hi]
    unfinished=(jnp.max(errors[1:],axis=-1)-goal_tolerance)/goal_tolerance
    stagnant=1-jnp.max(jnp.abs(past-errors[1:]),axis=-1)/progress_epsilon
    slow=1-jnp.max(speed,axis=-1)/(max_speed*speed_epsilon_fraction)
    last_steps=int(round(2./dt))
    terminal_stall=(jnp.minimum(jnp.min(1-speed[-last_steps:]/.05),
        jnp.min(1-jnp.abs(errors[-1]-errors[-last_steps])/.02))
        if n>=last_steps else jnp.array(-jnp.inf))
    result=event_margin(unfinished,stagnant,slow,end>=progress_window_seconds/dt-1e-10,
        terminal_stall,hold_samples=int(math.ceil(hold_seconds/dt-1e-10))+1)
    eta=result['robustness'];eligible=jnp.isfinite(eta)
    safe_eta=jnp.where(eligible,eta,0.)
    return dict(**result,J_live=jnp.where(eligible,jax.nn.softplus(safe_eta/temperature)/math.log(2.),0.),
                eligible=eligible)

"""Experimental temporal guard against motion without terminal progress.

Original deadlock labels and upper bound are unchanged. No spatial potential,
waypoint, escape direction or new control rule is constructed. The extra term
uses signed temporal goal-error change, the existing progress signal, only at
an original timeout or deadlock. It is not a completion certificate or a trained solution.
"""
import jax
import jax.numpy as jnp
from .ordered_guidance import trajectory as base_trajectory


def combine(base, margin, eligible, weight=.5):
    if not 0 < weight < 1:
        raise ValueError('guard weight must be strictly between zero and one')
    # Smooth bounded increasing map. Unlike a sharp sigmoid its derivative
    # near unit stalled margin does not vanish exponentially.
    bounded=.5*(1+margin/jnp.hypot(1.,margin))
    guard=jnp.where(eligible,weight*bounded,0.)
    return base+jnp.abs(1-base)*guard,guard


def trajectory(before,after,applied,goals,alive,*,terminal_timeout,weight=.5,**kwargs):
    base=base_trajectory(before,after,applied,goals,alive,
                         terminal_timeout=terminal_timeout,**kwargs)
    steps=int(round(kwargs['hold_seconds']/kwargs['dt']))
    count=alive.sum();end=jnp.clip(count-1,0,len(after)-1)
    begin=jnp.clip(count-steps,0,len(before)-1)
    norm=lambda x:jnp.sqrt(jnp.maximum(jnp.sum(x*x,axis=-1),1e-30))
    progress=norm(before[begin]-goals)-norm(after[end]-goals)
    # Reuse the declared progress-rate threshold, over the declared hold time.
    threshold=kwargs['progress_epsilon']*steps*kwargs['dt']/kwargs['progress_window_seconds']
    margin=1-jnp.max(progress)/threshold
    eligible=jax.lax.stop_gradient((terminal_timeout | (base['robustness_hard']>0)) & (count>=steps))
    risk,guard=combine(base['J_live'],margin,eligible,weight)
    return dict(J_live=risk,base_risk=base['J_live'],terminal_progress_margin=margin,
                terminal_progress_guard=guard,robustness_hard=base['robustness_hard'])

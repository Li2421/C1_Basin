"""Detector robustness plus deadline debt: a second, versioned candidate.

No outcome labels are used. This is piecewise differentiable, not smooth or a
calibrated probability. Deadline debt still needs a useful navigation potential.
"""
import math
import jax
import jax.numpy as jnp
from single_integrator.c1.risk.progress_debt import trajectory as debt_trajectory


def trajectory(before, after, applied, goals, alive, *, dt=.05,
               max_speed=.5, goal_tolerance=.08, target_rate=.4,
               hold_seconds=5., progress_window_seconds=2.,
               progress_epsilon=.01, speed_epsilon_fraction=.05):
    base = debt_trajectory(before, after, goals, alive, dt=dt,
        max_speed=max_speed, goal_tolerance=goal_tolerance, target_rate=target_rate,
        hold_seconds=hold_seconds, speed_epsilon_fraction=speed_epsilon_fraction)
    n = len(after)
    if applied.shape == (n, 4):
        applied = applied.reshape(n, 2, 2)
    if applied.shape != after.shape:
        raise ValueError('applied must contain both robots velocities for every action')
    norm = lambda v: jnp.sqrt(jnp.maximum(jnp.sum(v*v, axis=-1), 1e-30))
    distances = norm(jnp.concatenate([before[:1], after])-goals)
    end = jnp.arange(1, n+1)
    start = jnp.maximum(0., end-progress_window_seconds/dt)
    lo, hi = jnp.floor(start).astype(int), jnp.ceil(start).astype(int)
    past = (1-(start-lo))[:, None]*distances[lo]+(start-lo)[:, None]*distances[hi]
    progress = past-distances[1:]
    speed_limit = max_speed*speed_epsilon_fraction
    # Every atomic margin is positive exactly when its detector condition is
    # true (up to equality conventions). Normalization uses physical thresholds.
    margins = jnp.stack([
        (jnp.max(distances[1:], axis=-1)-goal_tolerance)/goal_tolerance,
        1-jnp.max(jnp.abs(progress), axis=-1)/progress_epsilon,
        1-jnp.max(norm(applied), axis=-1)/speed_limit], axis=-1)
    candidate = jnp.min(margins, axis=-1)
    ready = end >= progress_window_seconds/dt-1e-10
    candidate = jnp.where(alive & ready, candidate, -jnp.inf)
    count = int(math.ceil(hold_seconds/dt-1e-10))+1
    persistent = jax.lax.reduce_window(candidate, jnp.inf, jax.lax.min,
                                       (count,), (1,), 'VALID')
    robustness = jnp.max(persistent)
    deadlock = jnp.maximum(1+robustness, 0.)
    return {**base, 'J_live': jnp.maximum(base['timeout_bound'], deadlock),
            'deadlock_bound': deadlock, 'deadlock_robustness': robustness}

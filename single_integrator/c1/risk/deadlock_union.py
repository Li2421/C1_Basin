"""Strict temporal deadlock plus the original terminal-stall detector.

Both are explicit existing outcome definitions. Generic timeout is not
relabeled as deadlock. Summing nonnegative bounds preserves both gradients.
"""
import math
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from single_integrator.c1.risk.deadlock_primary import trajectory as strict_trajectory


def terminal_stall_upper(errors, speeds, alive, timeout, *, dt,
                         window_seconds=2., speed_threshold=.05,
                         progress_threshold=.02, temperature=.02):
    """Bound the original stalled-timeout predicate with no scene geometry.

    The historical detector uses round(window/dt) post-action samples. Its
    error difference therefore spans (samples-1)*dt, which is preserved here.
    timeout is the authoritative first-event timeout code, not a guessed label.
    """
    values = (dt, window_seconds, speed_threshold, progress_threshold, temperature)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('positive finite detector parameters required')
    if errors.ndim != 2 or speeds.shape != errors.shape or alive.shape != errors.shape[:1]:
        raise ValueError('expected errors/speeds [time,agents] and alive [time]')
    steps = int(round(window_seconds/dt))
    if steps < 1 or len(errors) < 1:
        raise ValueError('nonempty terminal window required')
    count = alive.sum()
    end = jnp.clip(count-1, 0, len(errors)-1)
    begin = jnp.clip(count-steps, 0, len(errors)-1)
    selected = alive & (jnp.arange(len(errors)) >= count-steps)
    velocity_margin = 1-speeds/speed_threshold
    progress_margin = 1-jnp.abs(errors[end]-errors[begin])/progress_threshold
    margins = jnp.concatenate([velocity_margin.reshape(-1), progress_margin])
    valid = jnp.concatenate([jnp.broadcast_to(selected[:, None], speeds.shape).reshape(-1),
                             jnp.ones(errors.shape[1], bool)])
    size = valid.sum()
    upper = -temperature*(logsumexp(jnp.where(valid, -margins/temperature, -jnp.inf))-jnp.log(size))
    hard = jnp.min(jnp.where(valid, margins, jnp.inf))
    eligible = jax.lax.stop_gradient(timeout & (count >= steps))
    return dict(stalled_bound=jnp.where(eligible, jax.nn.relu(1+upper), 0.),
                stalled_hard_risk=jnp.where(eligible, jax.nn.relu(1+hard), 0.),
                stalled_eligible=eligible)


def trajectory(before, after, applied, goals, alive, *, terminal_timeout,
               terminal_window_seconds=2., terminal_speed_threshold=.05,
               terminal_progress_threshold=.02, **kwargs):
    strict = strict_trajectory(before, after, applied, goals, alive, **kwargs)
    norm = lambda x: jnp.sqrt(jnp.maximum(jnp.sum(x*x, axis=-1), 1e-30))
    stalled = terminal_stall_upper(norm(after-goals), norm(applied.reshape(after.shape)),
        alive, terminal_timeout, dt=kwargs['dt'], window_seconds=terminal_window_seconds,
        speed_threshold=terminal_speed_threshold, progress_threshold=terminal_progress_threshold,
        temperature=kwargs.get('temperature', .02))
    return {**strict, **stalled, 'strict_bound': strict['J_live'],
        'J_live': strict['J_live']+stalled['stalled_bound'],
        'deadlock_bound': strict['J_live']+stalled['stalled_bound'],
        'hard_risk': strict['hard_risk']+stalled['stalled_hard_risk']}

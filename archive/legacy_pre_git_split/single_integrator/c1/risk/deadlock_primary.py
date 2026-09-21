"""Scene-independent temporal deadlock certificate from task/speed predicates.

The risk contains no roadmap, action recommendation, failure label, or timeout
term. Configuration describes the detector, not the scene. Smooth reductions
are upper approximations, with an explicit deterministic error bound.
"""
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def temporal_upper(margins, valid, *, hold_samples, temperature=.02):
    """Upper-bound max_window min_(time,predicate) margins on complete windows.

    Positive margins mean all predicates hold. For M predicates, L hold
    samples and K valid windows, 0 <= smooth-hard <= tau*log(L*M*K).
    ReLU(1+smooth) >= 1 whenever a complete deadlock window exists.
    No complete window gives exactly zero risk and gradient.
    """
    if margins.ndim != 2 or valid.shape != margins.shape[:1]:
        raise ValueError('expected margins [time,predicates], valid [time]')
    if not isinstance(hold_samples, int) or hold_samples < 1:
        raise ValueError('hold_samples must be a positive integer')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    n, predicates = margins.shape
    if not n or not predicates:
        raise ValueError('empty predicate trace')
    if n < hold_samples:
        zero = jnp.sum(margins)*0.
        return dict(J_live=zero, hard_risk=zero, robustness_upper=jnp.array(-jnp.inf),
                    robustness_hard=jnp.array(-jnp.inf), approximation_bound=zero,
                    complete_windows=jnp.array(0))
    indices = jnp.arange(n-hold_samples+1)[:, None]+jnp.arange(hold_samples)
    windows = margins[indices]
    enabled = jnp.all(valid[indices], axis=1)
    # -tau log(mean exp(-x/tau)) is an UPPER approximation of min(x).
    minimum = -temperature*(logsumexp(-windows/temperature, axis=(1, 2))
                            -math.log(hold_samples*predicates))
    hard = jnp.min(windows, axis=(1, 2))
    count = enabled.sum()
    any_window = count > 0
    # A finite dummy branch avoids NaN derivatives of all-negative-infinity LSE.
    logits = jnp.where(enabled, minimum/temperature, -jnp.inf)
    logits = jnp.where(any_window, logits, jnp.zeros_like(logits))
    upper = temperature*logsumexp(logits)
    exact = jnp.max(jnp.where(enabled, hard, -jnp.inf))
    risk = jnp.where(any_window, jax.nn.relu(1+upper), 0.)
    return dict(J_live=risk, hard_risk=jnp.maximum(1+exact, 0.),
        robustness_upper=jnp.where(any_window, upper, -jnp.inf),
        robustness_hard=exact, complete_windows=count,
        approximation_bound=jnp.where(any_window,
            temperature*jnp.log(hold_samples*predicates*jnp.maximum(count, 1)), 0.))


def trajectory(before, after, applied, goals, alive, *, dt, max_speed,
               goal_tolerance, hold_seconds, progress_window_seconds,
               progress_epsilon, speed_epsilon_fraction, temperature=.02):
    """Adapter for N-agent, D-dimensional goal-distance stagnation detectors.

    Config and goals are problem inputs. The adapter is not a universal
    definition of deadlock for every task. Other tasks supply their own margins
    to temporal_upper. Gradients at exact symmetric rest may remain zero.
    """
    scalars = (dt, max_speed, goal_tolerance, hold_seconds,
               progress_window_seconds, progress_epsilon, speed_epsilon_fraction)
    if not all(math.isfinite(v) and v > 0 for v in scalars):
        raise ValueError('detector parameters must be finite and positive')
    if before.ndim != 3 or after.shape != before.shape or goals.shape != before.shape[1:]:
        raise ValueError('expected positions [time,agents,dimension] and matching goals')
    n = len(before)
    applied = jnp.reshape(applied, before.shape)
    norm = lambda x: jnp.sqrt(jnp.maximum(jnp.sum(x*x, axis=-1), 1e-30))
    distances = norm(jnp.concatenate([before[:1], after], axis=0)-goals)
    end = jnp.arange(1, n+1)
    start = jnp.maximum(0., end-progress_window_seconds/dt)
    low, high = jnp.floor(start).astype(int), jnp.ceil(start).astype(int)
    past = (1-(start-low))[:, None]*distances[low]+(start-low)[:, None]*distances[high]
    margins = jnp.stack([
        (jnp.max(distances[1:], axis=-1)-goal_tolerance)/goal_tolerance,
        1-jnp.max(jnp.abs(past-distances[1:]), axis=-1)/progress_epsilon,
        1-jnp.max(norm(applied), axis=-1)/(max_speed*speed_epsilon_fraction)], axis=-1)
    valid = alive & (end >= progress_window_seconds/dt-1e-10)
    count = int(math.ceil(hold_seconds/dt-1e-10))+1
    result = temporal_upper(margins, valid, hold_samples=count, temperature=temperature)
    return {**result, 'deadlock_bound': result['J_live']}

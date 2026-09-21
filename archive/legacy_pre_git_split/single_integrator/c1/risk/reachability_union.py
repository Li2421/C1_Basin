"""Experimental bounded union certificate; no controller or trainer changes."""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .deadlock_primary import trajectory as strict_trajectory
from .reachability_margin import probability_upper


def trajectory(before, after, applied, goals, alive, *, terminal_timeout,
               margin_width=.5, tail_budget=.01, temperature=.02, **kwargs):
    strict = strict_trajectory(before, after, applied, goals, alive,
                               temperature=temperature, **kwargs)
    norm = lambda x: jnp.sqrt(jnp.maximum(jnp.sum(x*x, axis=-1), 1e-30))
    errors, speeds = norm(after-goals), norm(applied.reshape(after.shape))
    count = alive.sum()
    steps = int(round(2./kwargs['dt']))
    end = jnp.clip(count-1, 0, len(after)-1)
    begin = jnp.clip(count-steps, 0, len(after)-1)
    selected = alive & (jnp.arange(len(after)) >= count-steps)
    margins = jnp.concatenate([(1-speeds/.05).reshape(-1),
                              1-jnp.abs(errors[end]-errors[begin])/.02])
    valid = jnp.concatenate([jnp.broadcast_to(selected[:, None], speeds.shape).reshape(-1),
                             jnp.ones(errors.shape[1], bool)])
    stalled_upper = -temperature*(logsumexp(jnp.where(valid, -margins/temperature, -jnp.inf))
                                  -jnp.log(valid.sum()))
    stalled_hard = jnp.min(jnp.where(valid, margins, jnp.inf))
    enabled = jax.lax.stop_gradient(jnp.array([
        strict['complete_windows'] > 0, terminal_timeout & (count >= steps)]))
    upper = jnp.array([strict['robustness_upper'], stalled_upper])
    hard = jnp.array([strict['robustness_hard'], stalled_hard])
    logits = jnp.where(enabled, upper/temperature, -jnp.inf)
    logits = jnp.where(enabled.any(), logits, jnp.zeros_like(logits))
    rho = temperature*logsumexp(logits)
    score = probability_upper(rho, margin_width=margin_width, tail_budget=tail_budget)
    return dict(J_live=jnp.where(enabled.any(), score, 0.),
                robustness_upper=jnp.where(enabled.any(), rho, 0.),
                robustness_hard=jnp.where(enabled.any(), jnp.max(jnp.where(enabled, hard, -jnp.inf)), 0.))

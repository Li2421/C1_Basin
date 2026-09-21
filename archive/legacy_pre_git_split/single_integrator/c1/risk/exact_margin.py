"""Piecewise differentiable exact temporal margin, with no smoothing bias.

For the supplied finite-horizon deadlock predicates, rho > 0 means every
predicate of at least one eligible persistent event is strictly positive.
R = relu(1+rho) separates negative and positive event margins at 1 and bounds
the event indicator above. At ties it uses JAX min/max subgradients; it is
not an everywhere-smooth risk or a calibrated individual probability.
"""
import jax
import jax.numpy as jnp
from .deadlock_primary import trajectory as strict_trajectory


def trajectory(before, after, applied, goals, alive, *, terminal_timeout, **kwargs):
    strict = strict_trajectory(before, after, applied, goals, alive, **kwargs)
    norm = lambda x: jnp.sqrt(jnp.maximum(jnp.sum(x*x, axis=-1), 1e-30))
    errors, speeds = norm(after-goals), norm(applied.reshape(after.shape))
    steps = int(round(2./kwargs['dt']))
    count = alive.sum()
    end = jnp.clip(count-1, 0, len(after)-1)
    begin = jnp.clip(count-steps, 0, len(after)-1)
    selected = alive & (jnp.arange(len(after)) >= count-steps)
    speed_margin = jnp.min(jnp.where(selected[:,None], 1-speeds/.05, jnp.inf))
    progress_margin = jnp.min(1-jnp.abs(errors[end]-errors[begin])/.02)
    stall = jnp.minimum(speed_margin, progress_margin)
    enabled = jax.lax.stop_gradient(terminal_timeout & (count >= steps))
    rho = jnp.maximum(strict['robustness_hard'], jnp.where(enabled, stall, -jnp.inf))
    valid = (strict['complete_windows'] > 0) | enabled
    finite_rho = jnp.where(valid, rho, 0.)
    return dict(J_live=jnp.where(valid, jax.nn.relu(1+finite_rho), 0.),
                robustness_hard=finite_rho, eligible=valid)

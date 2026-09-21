"""Success-gated deadline certificate plus temporal deadlock certificate.

The completion gate is discrete and detached, as in existing first-event C1.
This preserves a pathwise failure upper bound but NOT an unbiased pathwise
gradient of the population event probability. That limitation is explicit.
"""
import jax
import jax.numpy as jnp
from single_integrator.c1.risk.temporal_certificate import trajectory as temporal


def trajectory(before, after, applied, goals, alive, **kwargs):
    result = temporal(before, after, applied, goals, alive, **kwargs)
    tolerance = kwargs.get('goal_tolerance', .08)
    # Success must be absorbing. Reaching the goal and then leaving would
    # require an explicit historical-success gate instead of this endpoint.
    squared = jnp.sum((after[-1]-goals)**2, axis=-1)
    failure = jax.lax.stop_gradient(jnp.any(squared > tolerance*tolerance))
    deadline = jnp.where(failure, result['timeout_bound'], 0.)
    return {**result, 'unconditional_deadline_bound': result['timeout_bound'],
            'timeout_bound': deadline,
            'J_live': jnp.maximum(deadline, result['deadlock_bound'])}

"""Experimental bounded event upper bound, NOT wired into any trainer.

Input must be an upper approximation of an event robustness: the event
implies robustness >= 0. This module does not manufacture a geometric
deadlock detector or guarantee a nonzero full-policy gradient.
"""
import math

import jax
import jax.numpy as jnp

from single_integrator.c1.risk.deadlock_primary import temporal_upper


def probability_upper(robustness_upper, *, margin_width, tail_budget):
    """R=(1+eta)*sigmoid(log(1/eta)*(1+2*rho_upper/delta)).

    For 0<eta<1, delta>0: R(0)=1, 0<=R<=1+eta, and
    rho_upper<=-delta implies R<=eta. A statistical probability
    interpretation requires an expectation over the actual rollout law.
    """
    if not math.isfinite(margin_width) or margin_width <= 0:
        raise ValueError('positive finite margin_width required')
    if not math.isfinite(tail_budget) or not 0 < tail_budget < 1:
        raise ValueError('tail_budget must be in (0,1)')
    k = -math.log(tail_budget)
    return (1 + tail_budget) * jax.nn.sigmoid(
        k * (1 + 2 * jnp.asarray(robustness_upper) / margin_width))


def temporal_probability_upper(margins, valid, *, hold_samples,
                               temperature, margin_width, tail_budget):
    """Bound a persistent conjunction, reusing audited temporal semantics.

    This adapter covers only the event described by the supplied margins.
    Terminal stalled-deadlock union wiring is deliberately not implicit.
    No valid event window gives exact zero risk and finite zero derivative.
    """
    result = temporal_upper(margins, valid, hold_samples=hold_samples,
                            temperature=temperature)
    enabled = result['complete_windows'] > 0
    finite_input = jnp.where(enabled, result['robustness_upper'], 0.)
    score = probability_upper(finite_input, margin_width=margin_width,
                              tail_budget=tail_budget)
    return {**result, 'J_live': jnp.where(enabled, score, 0.),
            'false_positive_band_width': margin_width + result['approximation_bound']}

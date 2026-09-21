"""Experimental moment-coupled upper bound; no controller/trainer integration.

Population mean/variance give the proved bound. Plug-in sample moments do
not automatically give a population upper confidence bound.
"""
import math
import jax
import jax.numpy as jnp


def from_moments(mean, variance, *, variance_floor=1e-6):
    """Cantelli bound plus smooth positive slack to retain a mean gradient.

    Event is rho>0, r=E[rho]/sqrt(Var[rho]+floor).
    R=1/(1+relu(-r)^2)+softplus(r) dominates the Cantelli bound.
    Requires finite population moments and nonnegative variance.
    """
    if not math.isfinite(variance_floor) or variance_floor<=0:
        raise ValueError('positive finite variance floor required')
    mean,variance=jnp.asarray(mean),jnp.asarray(variance)
    r=mean/jnp.sqrt(variance+variance_floor)
    cantelli=1/(1+jax.nn.relu(-r)**2)
    return dict(J_live=cantelli+jax.nn.softplus(r),cantelli=cantelli,
                standardized_mean=r,mean=mean,variance=variance)


def from_samples(margins, *, variance_floor=1e-6):
    margins=jnp.asarray(margins)
    if margins.ndim!=1 or len(margins)<2:
        raise ValueError('at least two scalar rollout margins required')
    mean=jnp.mean(margins)
    return from_moments(mean,jnp.mean((margins-mean)**2),variance_floor=variance_floor)

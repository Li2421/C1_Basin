"""One-sided event probability envelope with defensive Gaussian quadrature.

The proposal changes integration only, never the executed controller. Proposal
centres are held fixed while differentiating the target expectation.
"""
import math
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def event_envelope(margin, width=.1):
    if not math.isfinite(width) or width <= 0:
        raise ValueError('positive finite boundary width required')
    z = jnp.clip(1 + jnp.asarray(margin) / width, 0., 1.)
    return z * z * (3 - 2 * z)


def proposal_centres(margins, noise_gradients, width=.1, max_norm=2.):
    """Independent pilot linearization toward the middle of the boundary band."""
    gradients = jnp.asarray(noise_gradients)
    flat = gradients.reshape((len(gradients), -1))
    square = jnp.sum(flat * flat, axis=1)
    scale = (-width / 2 - jnp.asarray(margins)) / jnp.maximum(square, 1e-30)
    shift = scale[:, None] * flat
    norm = jnp.linalg.norm(shift, axis=1)
    shift = shift * jnp.minimum(1., max_norm / jnp.maximum(norm, 1e-30))[:, None]
    return shift.reshape(gradients.shape)


def importance_weights(draws, centres):
    """p / (.5 p + .5 mean_j N(centre_j,I)), hence 0 < weight <= 2."""
    draws, centres = jnp.asarray(draws), jnp.asarray(centres)
    x, mu = draws.reshape((len(draws), -1)), centres.reshape((len(centres), -1))
    ratio = x @ mu.T - .5 * jnp.sum(mu * mu, axis=1)
    mixture = jnp.concatenate((jnp.full((len(x), 1), math.log(.5)),
                               ratio + math.log(.5 / len(mu))), axis=1)
    return jnp.exp(-logsumexp(mixture, axis=1))

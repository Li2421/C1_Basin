"""One-gate Gaussian residual policy for Direction A.

The network retains the existing residual input convention ``(h_s, s, X_t,
E)``.  Its three output heads are the only change to that interface.  The
sampling function below is shared by operational rollouts and deployment
adapters; there is no deterministic evaluation policy.
"""
from dataclasses import dataclass
import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DirectionADistribution:
    """Fixed physical scale limits; no experiment defaults are supplied."""

    sigma_min: float
    sigma_max: float

    def __post_init__(self):
        if not (math.isfinite(self.sigma_min) and math.isfinite(self.sigma_max)
                and 0.0 < self.sigma_min < self.sigma_max):
            raise ValueError("require 0 < sigma_min < sigma_max < infinity")


def initialization_logits(p_init, sigma_init, config):
    """Convert approved initial physical values to head biases exactly."""
    if not (math.isfinite(p_init) and 0.0 < p_init < 1.0):
        raise ValueError("p_init must be strictly between zero and one")
    if not (math.isfinite(sigma_init)
            and config.sigma_min < sigma_init < config.sigma_max):
        raise ValueError("sigma_init must lie strictly inside configured bounds")
    alpha = math.log(p_init/(1.0-p_init))
    fraction = ((sigma_init-config.sigma_min)
                /(config.sigma_max-config.sigma_min))
    beta = math.log(fraction/(1.0-fraction))
    return alpha, beta


class DirectionAResidual(nn.Module):
    """Return ``(mu[4], alpha, beta)`` from the frozen residual inputs.

    ``alpha_bias`` and ``beta_bias`` are required because the repository has no
    approved Direction A initialization.  Experiments must record them and the
    initialization seed.  The zero head kernels make the supplied initial
    values spatially constant while retaining all shared/head parameters in the
    trainable pytree.
    """

    alpha_bias: float
    beta_bias: float
    hidden_dims: Sequence[int] = (256, 256, 256)
    action_dim: int = 4
    observation_dim: int = 20
    environment_dim: int = 0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, h_s, s, X_t, E=None):
        h_s, s, X_t = map(jnp.asarray, (h_s, s, X_t))
        if h_s.ndim != 2 or h_s.shape[-1] != self.action_dim:
            raise ValueError(f"h_s must have shape [B,{self.action_dim}]")
        if s.shape != (h_s.shape[0], 1):
            raise ValueError("s must have shape [B,1]")
        if X_t.shape != (h_s.shape[0], self.observation_dim):
            raise ValueError(f"X_t must have shape [B,{self.observation_dim}]")
        if E is None:
            E = jnp.zeros((h_s.shape[0], self.environment_dim), dtype=h_s.dtype)
        else:
            E = jnp.asarray(E)
            if E.shape != (h_s.shape[0], self.environment_dim):
                raise ValueError(f"E must have shape [B,{self.environment_dim}]")
        x = jnp.concatenate((X_t, h_s, s, E), axis=-1)
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = nn.gelu(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
        zero = nn.initializers.zeros
        mu = nn.Dense(self.action_dim, kernel_init=zero, bias_init=zero,
                      name="mean_head")(x)
        alpha = nn.Dense(
            1, kernel_init=zero, bias_init=nn.initializers.constant(self.alpha_bias),
            name="gate_head")(x)[..., 0]
        beta = nn.Dense(
            1, kernel_init=zero, bias_init=nn.initializers.constant(self.beta_bias),
            name="scale_head")(x)[..., 0]
        return mu, alpha, beta


def policy_parameters(outputs, config):
    """Map raw heads to the frozen Bernoulli probability and physical sigma."""
    mu, alpha, beta = outputs
    mu, alpha, beta = jnp.asarray(mu), jnp.asarray(alpha), jnp.asarray(beta)
    if mu.shape[-1] != 4 or alpha.shape != mu.shape[:-1] or beta.shape != alpha.shape:
        raise ValueError("expected mu[...,4], alpha[...] and beta[...]")
    gate_probability = jax.nn.sigmoid(alpha)
    scale_fraction = jax.nn.sigmoid(beta)
    sigma = (config.sigma_min
             + (config.sigma_max-config.sigma_min)*scale_fraction)
    return mu, gate_probability, sigma


def sample_residual(outputs, gate_uniform, gaussian, config):
    """Sample the exact train/deploy law with one joint gate per timestep."""
    mu, probability, sigma = policy_parameters(outputs, config)
    gate_uniform, gaussian = jnp.asarray(gate_uniform), jnp.asarray(gaussian)
    if gate_uniform.shape != probability.shape or gaussian.shape != mu.shape:
        raise ValueError("random draws must have shapes p[...] and epsilon[...,4]")
    gate = gate_uniform < probability
    active = mu + sigma[..., None]*gaussian
    residual = jnp.where(gate[..., None], active, jnp.zeros_like(active))
    return gate, residual, dict(mu=mu, alpha=jnp.asarray(outputs[1]),
                                beta=jnp.asarray(outputs[2]), p=probability,
                                sigma=sigma, epsilon=gaussian)


def log_probability(outputs, gate, residual, config):
    """Exact mixed-law log density for fixed sampled ``(gate, residual)``."""
    mu, probability, sigma = policy_parameters(outputs, config)
    gate = jnp.asarray(gate, dtype=jnp.bool_)
    residual = jnp.asarray(residual)
    if gate.shape != probability.shape or residual.shape != mu.shape:
        raise ValueError("sample shapes do not match policy outputs")
    alpha = jnp.asarray(outputs[1])
    log_gate = jnp.where(gate, jax.nn.log_sigmoid(alpha),
                         jax.nn.log_sigmoid(-alpha))
    standardized_square = jnp.sum((residual-mu)**2, axis=-1)/(sigma*sigma)
    log_gaussian = (-2.0*jnp.log(2.0*jnp.pi) - 4.0*jnp.log(sigma)
                    - 0.5*standardized_square)
    return log_gate + jnp.where(gate, log_gaussian, 0.0)


def log_probability_components(outputs, gate, residual, config):
    """Three scalar terms whose parameter gradients are the frozen scores.

    The cross-head quantities are stopped only to separate the gate, mean and
    scale score components.  Their sum has the same parameter gradient as
    :func:`log_probability` for fixed samples and conditioning history.
    """
    mu, probability, sigma = policy_parameters(outputs, config)
    gate = jnp.asarray(gate, dtype=jnp.bool_)
    residual = jax.lax.stop_gradient(jnp.asarray(residual))
    if gate.shape != probability.shape or residual.shape != mu.shape:
        raise ValueError("sample shapes do not match policy outputs")
    gate_float = gate.astype(mu.dtype)
    alpha = jnp.asarray(outputs[1])
    gate_term = jnp.where(gate, jax.nn.log_sigmoid(alpha),
                          jax.nn.log_sigmoid(-alpha))
    sigma_fixed = jax.lax.stop_gradient(sigma)
    mu_fixed = jax.lax.stop_gradient(mu)
    mean_term = gate_float*(-0.5*jnp.sum((residual-mu)**2, axis=-1)
                            /(sigma_fixed*sigma_fixed))
    scale_term = gate_float*(-4.0*jnp.log(sigma)
                             -0.5*jnp.sum((residual-mu_fixed)**2, axis=-1)
                             /(sigma*sigma))
    return gate_term, mean_term, scale_term

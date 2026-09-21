"""Zero-initialized physical-control residual for post-Safety C1 V0."""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class ResidualCorrection(nn.Module):
    """Compute ``G_phi(h_s, s, X_t, E)`` for the joint two-agent controller.

    ``ActorVectorField`` receives ``(observations, actions, times)`` with
    shapes ``[B,20]``, ``[B,4]``, and ``[B,1]`` and returns ``[B,4]``.  This
    module retains those dimensions: ``X_t`` is the flattened observation and
    ``h_s`` receives the physical safe control; ``s`` is a constant zero.
    This network is never injected into the frozen Flow sampler. ``E`` is an optional,
    fixed-width environment conditioning vector.

    The final Dense layer has zero kernel and bias, hence its initial output is
    exactly zero while earlier layers remain trainable once it is updated.
    """
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
        return nn.Dense(self.action_dim,
                        kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros,
                        name="zero_initialized_output")(x)

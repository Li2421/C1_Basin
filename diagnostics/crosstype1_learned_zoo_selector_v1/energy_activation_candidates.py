"""Smooth 3-by-256 energy MLPs for the documented synthetic diagnosis."""
import flax.linen as nn
import jax.numpy as jnp


class SmoothEnergy(nn.Module):
    activation: str

    @nn.compact
    def __call__(self,obs,action):
        value=jnp.concatenate((obs,action),axis=-1)
        activate={'tanh':jnp.tanh,'silu':nn.silu,'softplus':nn.softplus}[self.activation]
        for _ in range(3):value=activate(nn.Dense(256)(value))
        return nn.Dense(1)(value)[...,0]

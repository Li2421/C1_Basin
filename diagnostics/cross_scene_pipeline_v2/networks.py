"""Copied numerical distributions/critic; specified one-adapter generator and new ablations."""
import bootstrap
import math
from typing import Any
import hashlib
import numpy as np
import flax.linen as nn
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
LOW=np.asarray([0.,-1.5,0.],np.float32)
HIGH=np.asarray([2.,1.5,1.5],np.float32)
CENTER=(LOW+HIGH)/2
RADIUS=(HIGH-LOW)/2
SEEDS=(17,23,41)


class Generator(nn.Module):
    """One q(eta|h,c), with native-input adapters and a shared trunk."""

    def setup(self):
        self.state_adapter = nn.Dense(96)
        self.context = nn.Dense(32)
        self.shared1 = nn.Dense(128)
        self.shared2 = nn.Dense(64)
        self.out = nn.Dense(6)

    def branch(self, h, c, adapter):
        z = nn.silu(adapter(h))
        ec = nn.silu(self.context(c))
        x = jnp.concatenate((z, ec), axis=-1)
        x = nn.silu(self.shared1(x))
        x = nn.silu(self.shared2(x))
        return self.out(x)

    def __call__(self, h, c):
        return self.branch(h, c, self.state_adapter)


class Critic(nn.Module):
    use_state: bool=True
    use_controller: bool=True

    def setup(self):
        if self.use_state:
            self.ring_adapter=nn.Dense(96);self.context=nn.Dense(32)
        self.action_encoder=nn.Dense(32)
        if self.use_controller:self.controller_encoder=nn.Dense(32)
        self.shared1=nn.Dense(128);self.shared2=nn.Dense(64);self.out=nn.Dense(1)

    def __call__(self,h,c,controller,action):
        parts=[]
        if self.use_state:parts.extend([nn.silu(self.ring_adapter(h)),nn.silu(self.context(c))])
        parts.append(nn.silu(self.action_encoder(action)))
        if self.use_controller:parts.append(nn.silu(self.controller_encoder(controller)))
        x=jnp.concatenate(parts,axis=-1)
        x=nn.silu(self.shared1(x));x=nn.silu(self.shared2(x))
        return self.out(x)[...,0]

def stable_int(*parts: Any) -> int:
    text = "\0".join(map(str, parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

def dist_params(raw):
    return raw[..., :3], 0.025 + 0.275 * jax.nn.sigmoid(raw[..., 3:])

def eta_mean(raw):
    return jnp.asarray(CENTER) + jnp.asarray(RADIUS) * jnp.tanh(raw[..., :3])

def eta_from_noise(raw, noise):
    mu, sigma = dist_params(raw)
    return jnp.asarray(CENTER) + jnp.asarray(RADIUS) * jnp.tanh(mu + sigma * noise)

def eta_normalized(eta):
    return (eta - jnp.asarray(CENTER)) / jnp.asarray(RADIUS)

def log_prob(raw, eta):
    u = jnp.clip(eta_normalized(eta), -0.999999, 0.999999)
    z = jnp.arctanh(u)
    mu, sigma = dist_params(raw)
    gaussian = -0.5 * (((z - mu) / sigma) ** 2 + 2 * jnp.log(sigma) + math.log(2 * math.pi))
    jac = jnp.log(jnp.asarray(RADIUS) * (1 - u * u) + 1e-9)
    return jnp.sum(gaussian - jac, axis=-1)

class NoControllerCritic(Critic):
    use_controller: bool=False

class SceneOnlyCritic(Critic):
    use_state: bool=False
    def setup(self):
        super().setup()
        self.context=nn.Dense(32)
    def __call__(self,h,c,controller,action):
        parts=[nn.silu(self.context(c)),nn.silu(self.action_encoder(action))]
        if self.use_controller:parts.append(nn.silu(self.controller_encoder(controller)))
        x=jnp.concatenate(parts,axis=-1)
        x=nn.silu(self.shared1(x));x=nn.silu(self.shared2(x))
        return self.out(x)[...,0]

def critic(mode):
    return {'full':Critic,'no_controller':NoControllerCritic,'scene_only':SceneOnlyCritic}[mode]()

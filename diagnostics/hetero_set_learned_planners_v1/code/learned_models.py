"""Three learned planner families (Flax), trained by imitation of the prioritized A* planner:
  mlp        - behavior cloning, 3x256 multilayer perceptron, mean-squared error
  attention  - Transformer over the robot's own token + 8 neighbour tokens (masked), 2 blocks, width 64
  diffusion  - diffusion policy (DDPM, 20 steps, cosine schedule), denoiser conditioned on the observation
Actions are velocity commands divided by max speed (norm <= 1)."""
import json, math
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax import serialization
from learned_obs import K, RAYS, OBS_DIM

EGO = 8 + RAYS
T_DIFF = 20


def cosine_schedule(T=T_DIFF, s=0.008):
    x = np.linspace(0, T, T + 1)
    ab = np.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.5)
    alphas = 1 - betas
    return betas.astype(np.float32), alphas.astype(np.float32), np.cumprod(alphas).astype(np.float32)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            x = nn.gelu(nn.Dense(256)(x))
        return nn.Dense(2)(x)


class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=64)(h, h, mask=mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(64)(nn.gelu(nn.Dense(128)(h)))
        return x + h


class AttentionPolicy(nn.Module):
    @nn.compact
    def __call__(self, xn, valid):
        ego = nn.Dense(64)(xn[:, :EGO])[:, None]
        nb = xn[:, EGO:].reshape(xn.shape[0], K, 6)[..., :5]
        tok = jnp.concatenate([ego, nn.Dense(64)(nb)], 1)
        m = jnp.concatenate([jnp.ones((xn.shape[0], 1), bool), valid], 1)
        mask = m[:, None, None, :] & m[:, None, :, None]
        for _ in range(2):
            tok = Block()(tok, mask)
        h = nn.gelu(nn.Dense(128)(nn.LayerNorm()(tok[:, 0])))
        return nn.Dense(2)(h)


class Denoiser(nn.Module):
    @nn.compact
    def __call__(self, xn, a, t):
        freqs = jnp.exp(jnp.linspace(0, math.log(100.0), 16))
        te = jnp.concatenate([jnp.sin(t[:, None] * freqs), jnp.cos(t[:, None] * freqs)], -1)
        h = jnp.concatenate([nn.gelu(nn.Dense(256)(xn)), a, nn.gelu(nn.Dense(64)(te))], -1)
        for _ in range(3):
            h = nn.gelu(nn.Dense(256)(h))
        return nn.Dense(2)(h)


def make(kind):
    return {'mlp': MLP(), 'attention': AttentionPolicy(), 'diffusion': Denoiser()}[kind]


def valid_of(obs):
    return obs[:, EGO:].reshape(obs.shape[0], K, 6)[..., 5] > 0.5


def apply_fn(kind, model):
    """Returns f(params, xn, valid, a, t) -> output (action for mlp/attention, predicted noise for diffusion)."""
    if kind == 'mlp':
        return lambda p, xn, valid, a, t: model.apply(p, xn)
    if kind == 'attention':
        return lambda p, xn, valid, a, t: model.apply(p, xn, valid)
    return lambda p, xn, valid, a, t: model.apply(p, xn, a, t)


class Policy:
    """Inference wrapper: obs (N,72) raw -> action (N,2) in max-speed units, norm <= 1."""

    def __init__(self, ckpt_dir):
        meta = json.load(open(f'{ckpt_dir}/meta.json'))
        self.kind = meta['kind']
        self.mean = np.asarray(meta['mean'], np.float32); self.std = np.asarray(meta['std'], np.float32)
        model = make(self.kind)
        dummy = (jnp.zeros((1, OBS_DIM)), jnp.zeros((1, K), bool), jnp.zeros((1, 2)), jnp.zeros((1,)))
        init = model.init(jax.random.PRNGKey(0), *(dummy[:1] if self.kind == 'mlp' else dummy[:2] if self.kind == 'attention' else (dummy[0], dummy[2], dummy[3])))
        self.params = serialization.from_bytes(init, open(f'{ckpt_dir}/params.msgpack', 'rb').read())
        f = apply_fn(self.kind, model)
        if self.kind != 'diffusion':
            self.fn = jax.jit(lambda p, xn, valid: f(p, xn, valid, None, None))
        else:
            betas, alphas, ab = cosine_schedule()
            ab_prev = np.concatenate([[1.0], ab[:-1]]).astype(np.float32)
            c0 = (np.sqrt(ab_prev) * betas / (1 - ab)).astype(np.float32)
            ct = (np.sqrt(alphas) * (1 - ab_prev) / (1 - ab)).astype(np.float32)
            sig = np.sqrt(betas * (1 - ab_prev) / (1 - ab)).astype(np.float32)

            def sample(p, xn, valid, noise):   # noise (T+1, N, 2): initial + per-step
                x = noise[0]
                for i in reversed(range(T_DIFF)):
                    t = jnp.full((xn.shape[0],), float(i))
                    eps = f(p, xn, valid, x, t)
                    x0 = jnp.clip((x - jnp.sqrt(1 - ab[i]) * eps) / jnp.sqrt(ab[i]), -1.5, 1.5)
                    x = c0[i] * x0 + ct[i] * x + (sig[i] * noise[i + 1] if i > 0 else 0.0)
                return x
            self.fn = jax.jit(sample)

    def __call__(self, obs, rng=None):
        xn = jnp.asarray((obs - self.mean) / self.std)
        valid = jnp.asarray(valid_of(obs))
        if self.kind == 'diffusion':
            noise = jnp.asarray(rng.standard_normal((T_DIFF + 1, obs.shape[0], 2)).astype(np.float32))
            a = np.asarray(self.fn(self.params, xn, valid, noise), np.float64)
        else:
            a = np.asarray(self.fn(self.params, xn, valid), np.float64)
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        return a * np.minimum(1.0, 1.0 / np.maximum(n, 1e-30))

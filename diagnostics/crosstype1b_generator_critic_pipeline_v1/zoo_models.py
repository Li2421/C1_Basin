"""New learned policies. No imports with experiment output side effects."""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class MLP(nn.Module):
    outputs: int
    layers: int = 3

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = nn.relu(nn.Dense(256)(x))
        return nn.Dense(self.outputs)(x)


class ChunkingPolicy(nn.Module):
    @nn.compact
    def __call__(self, obs):
        return MLP(64)(obs).reshape(*obs.shape[:-1], 8, 8)


class RecurrentPolicy(nn.Module):
    @nn.compact
    def __call__(self, obs, hidden):
        hidden, output = nn.GRUCell(features=256)(hidden, obs)
        result = MLP(16, layers=2)(output)
        return hidden, result[..., :8], jnp.clip(result[..., 8:], -5., 1.5)


class ConditionalVAE(nn.Module):
    def setup(self):
        self.encoder = MLP(8)
        self.decoder = MLP(8)

    def __call__(self, obs, target, key):
        moments = self.encoder(jnp.concatenate((obs, target), -1))
        mean, logs = moments[..., :4], jnp.clip(moments[..., 4:], -5., 1.5)
        latent = mean + jnp.exp(logs) * jax.random.normal(key, mean.shape)
        decoded = self.decode(obs, latent)
        kl = .5 * jnp.sum(jnp.square(mean) + jnp.exp(2 * logs) - 1 - 2 * logs, axis=-1)
        return decoded, kl

    def decode(self, obs, latent):
        return self.decoder(jnp.concatenate((obs, latent), -1))


class EnergyPolicy(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        return MLP(1)(jnp.concatenate((obs, action), -1))[..., 0]


def infonce(net,parameters,obs,target,key,action_mean,action_scale):
    raw=jax.random.uniform(key,(len(obs),256,8),minval=-1.,maxval=1.)
    negative=(raw-action_mean)/action_scale
    candidates=jnp.concatenate((target[:,None,:],negative),axis=1)
    condition=jnp.broadcast_to(obs[:,None,:],(*candidates.shape[:-1],obs.shape[-1]))
    scores=-net.apply(parameters,condition,candidates)
    return jnp.mean(jax.scipy.special.logsumexp(scores,axis=-1)-scores[:,0])


def energy_candidates(net, parameters, obs, key, action_mean, action_scale):
    """CEM in physical bounded action coordinates; best across all 3 rounds."""
    mean = jnp.zeros(8)
    std = jnp.ones(8)
    best = jnp.zeros(8)
    best_energy = jnp.inf
    for round_index in range(3):
        k = jax.random.fold_in(key, round_index)
        raw = jax.random.uniform(k, (1024, 8), minval=-1., maxval=1.) if round_index == 0 else (
              jnp.clip(mean + std * jax.random.normal(k, (1024, 8)), -1., 1.))
        normalized = (raw - action_mean) / action_scale
        energy = net.apply(parameters, jnp.broadcast_to(obs, (1024, obs.shape[-1])), normalized)
        index = jnp.argmin(energy)
        improve = energy[index] < best_energy
        best = jnp.where(improve, raw[index], best)
        best_energy = jnp.minimum(best_energy, energy[index])
        elite = raw[jnp.argsort(energy)[:103]]
        mean = elite.mean(0)
        std = jnp.maximum(elite.std(0), 1e-3)
    return (best - action_mean) / action_scale


def network(family):
    return {'chunking':ChunkingPolicy, 'recurrent':RecurrentPolicy,
            'cvae':ConditionalVAE, 'implicit':EnergyPolicy}[family]()


def initialize(net, family, key, batch=1):
    obs = jnp.zeros((batch, 92), jnp.float32)
    if family == 'recurrent':
        return net.init(key, obs, jnp.zeros((batch, 256), jnp.float32))
    if family == 'cvae':
        return net.init(key, obs, jnp.zeros((batch, 8), jnp.float32), key)
    if family == 'implicit':
        return net.init(key, obs, jnp.zeros((batch, 8), jnp.float32))
    return net.init(key, obs)


def decode(z, normal):
    raw = jnp.clip(jnp.asarray(normal['act_mean']) + jnp.asarray(normal['act_scale']) * z, -1., 1.)
    raw = raw.reshape(*raw.shape[:-1], 4, 2)
    return raw * jnp.minimum(1., normal['max_speed'] / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-30))


class PolicySession:
    """Episode-owned state; safety projection never gets stored sequence buffers."""
    def __init__(self, family, parameters, normal):
        self.family, self.parameters, self.normal = family, parameters, normal
        self.net = network(family)
        self.reset()
        self._call = jax.jit(self._sample)

    def reset(self):
        self.hidden = np.zeros((1, 256), np.float32)
        self.sequence = None
        self.sequence_start = None
        self.last_step = -1
        self.last_action = None

    def _sample(self, obs, key, hidden):
        obs = (obs.reshape(1,92) - jnp.asarray(self.normal['obs_mean'])) / jnp.asarray(self.normal['obs_scale'])
        if self.family == 'chunking':
            return self.net.apply(self.parameters, obs)[0], hidden
        if self.family == 'recurrent':
            h, mean, logs = self.net.apply(self.parameters, obs, hidden)
            z = mean + jnp.exp(logs) * jax.random.normal(key, mean.shape)
            return z[0], h
        if self.family == 'cvae':
            z = self.net.apply(self.parameters, obs, jax.random.normal(key, (1,4)), method=self.net.decode)
            return z[0], hidden
        z = energy_candidates(self.net, self.parameters, obs[0], key,
                              jnp.asarray(self.normal['act_mean']), jnp.asarray(self.normal['act_scale']))
        return z, hidden

    def action(self, obs, key, step):
        if step == self.last_step:
            return self.last_action.copy()
        if step != self.last_step + 1:
            raise ValueError('PolicySession needs every consecutive physical step; reset per episode')
        if self.family == 'chunking':
            if step % 4 == 0:
                sequence, _ = self._call(jnp.asarray(obs), key, jnp.asarray(self.hidden))
                self.sequence = np.asarray(decode(sequence, self.normal)).copy()
                self.sequence_start = step
            action = self.sequence[step-self.sequence_start].copy()
        else:
            z, hidden = self._call(jnp.asarray(obs), key, jnp.asarray(self.hidden))
            self.hidden = np.asarray(hidden).copy()
            action = np.asarray(decode(z, self.normal)).copy()
        self.last_step, self.last_action = step, action.copy()
        return action.copy()


def sequence_examples(trajectories, normal, family):
    """Uniform transition anchors. Padded actions repeat last, masked in loss."""
    length = 8 if family == 'chunking' else 32
    observations, actions, masks = [], [], []
    for trajectory in trajectories:
        x = trajectory.observations[:-1].reshape(-1,92)
        y = trajectory.actions.reshape(-1,8)
        if family == 'chunking':
            indices = np.arange(len(y))[:,None] + np.arange(length)[None,:]
            masks.append(indices < len(y))
            observations.append(x)
            actions.append(y[np.minimum(indices,len(y)-1)])
        else:
            # Non-overlapping truncated sequences preserve temporal order,
            # and their recurrent initial state is provided by a prefix scan.
            for start in range(0,len(y),length):
                indices = start + np.arange(length)
                observations.append(x[np.minimum(indices,len(y)-1)][None])
                actions.append(y[np.minimum(indices,len(y)-1)][None])
                masks.append((indices < len(y))[None])
    x = np.concatenate(observations).astype(np.float32)
    y = np.concatenate(actions).astype(np.float32)
    x = (x-np.asarray(normal['obs_mean'],np.float32))/np.asarray(normal['obs_scale'],np.float32)
    y = (y-np.asarray(normal['act_mean'],np.float32))/np.asarray(normal['act_scale'],np.float32)
    return x, y, np.concatenate(masks)

"""Shared per-agent world-coordinate deterministic and mixture-density policies."""
from common import *
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
from observation import observe
jax.config.update('jax_enable_x64',False)

class Policy(nn.Module):
    family:str
    @nn.compact
    def __call__(self,x):
        for _ in range(3):x=nn.relu(nn.Dense(256)(x))
        z=nn.Dense(25 if self.family=='mixture' else 2)(x)
        if self.family=='mixture':return z[...,:5],z[...,5:15].reshape(*z.shape[:-1],5,2),jnp.clip(z[...,15:].reshape(*z.shape[:-1],5,2),-5.,1.5)
        return z

_cache={}
def load_policy(checkpoint):
    path=Path(checkpoint); key=(str(path),sha(path))
    if key in _cache:return _cache[key]
    cfg=read(path.parent/'CONFIG.json');normal=read(path.parent/'NORMALIZATION.json')
    net=Policy(cfg['family']);pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,35),jnp.float32)),path.read_bytes())
    @jax.jit
    def sample(observed,key):
        x=(observed-jnp.asarray(normal['obs_mean']))/jnp.asarray(normal['obs_scale'])
        if cfg['family']=='deterministic':z=net.apply(pa,x)
        else:
            logits,means,logs=net.apply(pa,x);a,b=jax.random.split(key)
            indices=jax.random.categorical(a,logits,axis=-1)
            selected=means[jnp.arange(len(x)),indices];scale=logs[jnp.arange(len(x)),indices]
            z=selected+jnp.exp(scale)*jax.random.normal(b,selected.shape)
        raw=jnp.clip(jnp.asarray(normal['act_mean'])+jnp.asarray(normal['act_scale'])*z,-1.,1.)
        return raw*jnp.minimum(1.,.5/jnp.maximum(jnp.linalg.norm(raw,axis=-1,keepdims=True),1e-30))
    def act(env,key):return np.asarray(sample(jnp.asarray(observe(env)),key),np.float64)
    _cache[key]=act;return act

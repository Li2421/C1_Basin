from common import *
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
class GaussianNet(nn.Module):
 @nn.compact
 def __call__(self,x):
  for width in [256,256,256]:x=nn.relu(nn.Dense(width)(x))
  z=nn.Dense(16)(x);return z[...,:8],jnp.clip(z[...,8:],-5.,1.5)
def sample(net,parameters,obs,key,cfg):
 x=(jnp.asarray(obs).reshape(-1)-jnp.asarray(cfg['obs_mean']))/jnp.asarray(cfg['obs_scale']);mean,logstd=net.apply(parameters,x);z=mean+jnp.exp(logstd)*jax.random.normal(key,(8,));raw=jnp.clip(jnp.asarray(cfg['act_mean'])+jnp.asarray(cfg['act_scale'])*z,-1.,1.).reshape(4,2);return raw*jnp.minimum(1.,cfg['max_speed']/jnp.maximum(jnp.linalg.norm(raw,axis=-1,keepdims=True),1e-30))

"""The locked six Transformer Gaussian policies; no flow-matching substitute."""
import flax.linen as nn
import jax,jax.numpy as jnp
class TransformerPolicy(nn.Module):
    layers:int
    @nn.compact
    def __call__(self,observations):
        x=observations.reshape(*observations.shape[:-1],4,23)
        x=nn.Dense(128,name='embedding')(x)
        for i in range(self.layers):
            y=nn.LayerNorm(name=f'attention_norm_{i}')(x)
            y=nn.SelfAttention(num_heads=4,qkv_features=128,out_features=128,dropout_rate=0.,name=f'attention_{i}')(y,deterministic=True);x=x+y
            y=nn.LayerNorm(name=f'feedforward_norm_{i}')(x);y=nn.Dense(256,name=f'feedforward_in_{i}')(y);y=nn.gelu(y);x=x+nn.Dense(128,name=f'feedforward_out_{i}')(y)
        values=nn.Dense(4,name='gaussian_head')(x)
        return values[...,:2].reshape(*observations.shape[:-1],8),jnp.clip(values[...,2:].reshape(*observations.shape[:-1],8),-5.,1.5)
def gaussian_nll(mean,logs,target):return jnp.sum(.5*jnp.square((target-mean)*jnp.exp(-logs))+logs+.5*jnp.log(2*jnp.pi),axis=-1)
def decode(z,normal):
    raw=jnp.clip(jnp.asarray(normal['act_mean'],jnp.float32)+jnp.asarray(normal['act_scale'],jnp.float32)*z,-1.,1.).reshape(*z.shape[:-1],4,2)
    return raw*jnp.minimum(1.,normal['max_speed']/jnp.maximum(jnp.linalg.norm(raw,axis=-1,keepdims=True),1e-30))
def sample(net,parameters,observation,key,normal):
    x=(jnp.asarray(observation,jnp.float32).reshape(1,92)-jnp.asarray(normal['obs_mean'],jnp.float32))/jnp.asarray(normal['obs_scale'],jnp.float32)
    mean,logs=net.apply(parameters,x);z=mean[0]+jnp.exp(logs[0])*jax.random.normal(key,(8,))
    return decode(z,normal)

"""Fixed critic recipes: unconstrained logits, no prior anchor or fallback."""
import bootstrap
import flax.linen as nn
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
class MLP(nn.Module):
 use_neighbours:bool=True
 @nn.compact
 def __call__(self,x,training=False):
  if not self.use_neighbours:x=x[...,:24]
  x=nn.silu(nn.Dense(128,name='dense1')(x));x=nn.Dropout(.1,name='drop1')(x,deterministic=not training)
  x=nn.silu(nn.Dense(128,name='dense2')(x));x=nn.Dropout(.1,name='drop2')(x,deterministic=not training)
  return nn.Dense(1,name='out')(x)[...,0]
class Original(nn.Module):
 def setup(self):
  self.ring_adapter=nn.Dense(96);self.context=nn.Dense(32);self.action_encoder=nn.Dense(32);self.controller_encoder=nn.Dense(32);self.shared1=nn.Dense(128);self.shared2=nn.Dense(64);self.out=nn.Dense(1)
 def __call__(self,x,training=False):
  c=x[...,:12];h=x[...,12:18];ct=x[...,18:21];e=x[...,21:24]
  parts=[nn.silu(self.ring_adapter(h)),nn.silu(self.context(c)),nn.silu(self.action_encoder(e)),nn.silu(self.controller_encoder(ct))]
  z=jnp.concatenate(parts,-1);z=nn.silu(self.shared1(z));z=nn.silu(self.shared2(z));return self.out(z)[...,0]
def model(kind):return Original()if kind=='original'else MLP(kind!='no_neighbour')

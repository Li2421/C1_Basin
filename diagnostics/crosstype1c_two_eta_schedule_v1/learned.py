"""Three policies with the exact source recipe's observation/action chart."""
from common import *
import jax,jax.numpy as jnp,flax.linen as nn
from flax import serialization
from legacy_controllers import Nominal,bound
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
jax.config.update('jax_enable_x64',False)
class Policy(nn.Module):
 family:str
 @nn.compact
 def __call__(self,obs,action=None,t=None):
  x=obs
  if self.family=='diffusion':
   frequency=jnp.exp(-jnp.arange(16)*jnp.log(10000.)/15)
   theta=t[...,None]*frequency[None,:];embedding=jnp.concatenate([jnp.sin(theta),jnp.cos(theta)],-1)
   x=jnp.concatenate([obs,action,embedding],-1)
  for _ in range(3):x=nn.relu(nn.Dense(256)(x))
  z=nn.Dense(85 if self.family=='mixture' else 8)(x)
  if self.family=='mixture':return z[...,:5],z[...,5:45].reshape(*z.shape[:-1],5,8),jnp.clip(z[...,45:].reshape(*z.shape[:-1],5,8),-5.,1.5)
  return z

def alpha_bar():return jnp.cumprod(1-jnp.linspace(.0001,.02,100))
def sample(net,params,obs,key,normal,family):
 x=(jnp.asarray(obs).reshape(1,92)-jnp.asarray(normal['obs_mean']))/jnp.asarray(normal['obs_scale'])
 if family=='deterministic':z=net.apply(params,x)[0]
 elif family=='mixture':
  logits,means,logs=net.apply(params,x);k1,k2=jax.random.split(key);index=jax.random.categorical(k1,logits[0]);z=means[0,index]+jnp.exp(logs[0,index])*jax.random.normal(k2,(8,))
 else:
  z=jax.random.normal(key,(1,8));ab=alpha_bar();ts=np.linspace(99,0,10,dtype=int)
  for i,t in enumerate(ts):
   epsilon=net.apply(params,x,z,jnp.full((1,),t,jnp.float32));clean=(z-jnp.sqrt(1-ab[t])*epsilon)/jnp.sqrt(ab[t]);previous=ab[ts[i+1]] if i+1<len(ts) else 1.;z=jnp.sqrt(previous)*clean+jnp.sqrt(1-previous)*epsilon
  z=z[0]
 raw=jnp.clip(jnp.asarray(normal['act_mean'])+jnp.asarray(normal['act_scale'])*z,-1.,1.).reshape(4,2)
 return raw*jnp.minimum(1.,normal['max_speed']/jnp.maximum(jnp.linalg.norm(raw,axis=-1,keepdims=True),1e-30))
class LearnedNominal(Nominal):
 def __init__(self,payload):
  manifest=read(payload['adapter_manifest_path']);super().__init__(payload['controller_class'],[],manifest);self.frame=payload['observation_frame'];self.checkpoint=Path(payload['flow_checkpoint_path']);assert sha(self.checkpoint)==payload['flow_checkpoint_sha256'];normal=read(self.checkpoint.parent/'NORMALIZATION.json');self.net=Policy(self.kind);args=(jnp.zeros((1,92),jnp.float32),jnp.zeros((1,8),jnp.float32),jnp.zeros((1,),jnp.float32)) if self.kind=='diffusion' else (jnp.zeros((1,92),jnp.float32),);self.parameters=serialization.from_bytes(self.net.init(jax.random.PRNGKey(0),*args),self.checkpoint.read_bytes());self.sample=jax.jit(lambda obs,key:sample(self.net,self.parameters,obs,key,normal,self.kind))
 def observation(self,env):return RingFlowAdapter.observation(self,env)
 def to_world(self,a,env):return RingFlowAdapter.to_world(self,a,env)
 def nominal(self,env,key,traced=False):
  assert not traced;obs=self.observation(env);raw=np.asarray(self.sample(jnp.asarray(obs),key),np.float64);return obs,bound(self.to_world(raw,env).copy(),self.config.max_speed),None

"""One shared official one-step actor, with observer-local public noise encoding."""
from core import *
from guard import guard
from data import teacher,LOCAL
import jax,jax.numpy as jnp,pickle
from new_benchmark_common.macflow import ActorVectorField
from poc.field_controller import Runtime
from ring_exchange.local_frame import local_observation,local_actions_to_world
guard()
NOISE_TAG=3006
def network():return ActorVectorField(hidden_dims=(512,512,512,512),action_dim=2,layer_norm=False,encoder=None)
def inputs(obs,mean,scale):
 ids=jnp.broadcast_to(jnp.eye(4,dtype=jnp.float32),(*obs.shape[:-2],4,4));return jnp.concatenate((ids,(obs-mean)/scale),-1)
def to_local(v,p):
 r=p/jnp.maximum(jnp.linalg.norm(p,axis=-1,keepdims=True),1e-12);t=jnp.stack((-r[...,1],r[...,0]),-1)
 return jnp.stack(((v*r).sum(-1),(v*t).sum(-1)),-1)
def joint_local_noise(obs,z):
 """Only each observer's radius and its relative-position slots enter.
 z has (...,4,2), each component expressed in source robot j's frame.
 """
 rows=[]
 for i in range(4):
  sources=[]
  for j in range(4):
   if j==i:q=jnp.stack((obs[...,i,0],jnp.zeros_like(obs[...,i,0])),-1)
   else:
    slot=[k for k in range(4) if k!=i].index(j);q=obs[...,i,6+4*slot:8+4*slot]+jnp.stack((obs[...,i,0],jnp.zeros_like(obs[...,i,0])),-1)
   r=q/jnp.maximum(jnp.linalg.norm(q,axis=-1,keepdims=True),1e-12);t=jnp.stack((-r[...,1],r[...,0]),-1)
   sources.append(z[...,j,:1]*r+z[...,j,1:]*t)
  rows.append(jnp.concatenate(sources,-1))
 return jnp.stack(rows,-2)
def joint_global_noise(z,positions):
 return jnp.stack([jnp.concatenate([to_local(z[...,j,:],positions[...,i,:]) for j in range(4)],-1) for i in range(4)],-2)
def encoding(index,variant,x,o,z):
 if variant=='V3':return joint_local_noise(x,z) if index in LOCAL else joint_global_noise(z,o[...,:2])
 return z if index in LOCAL else to_local(z,o[...,:2])
def target(index,u,o):return u if index in LOCAL else to_local(u,o[...,:2])
def public_key(key):return jax.random.fold_in(key,NOISE_TAG)
class Student(Runtime):
 def __init__(self,index,variant,checkpoint):
  base=teacher(index);self.__dict__.update({k:getattr(base,k) for k in ['scenario','config','config_cls','env_cls','manifest','basis','cbf']});self._base=base;self.index=index;self.variant=variant;self.checkpoint=Path(checkpoint)
  with self.checkpoint.open('rb') as f:ck=pickle.load(f)
  self.mean=ck['mean'];self.scale=ck['scale'];self.params=ck['params'];model=network();self.forward=jax.jit(lambda x,z:model.apply({'params':self.params},x,z));guard()
 def nominal(self,env,key,traced=False):
  assert not traced
  x=local_observation(env.positions,env.velocities,env.goals,env.config);o=inputs(jnp.asarray(x),self.mean,self.scale)
  if self.variant=='V3':
   z=jax.random.normal(public_key(key),(1,8),dtype=jnp.float32).reshape(4,2)
   noise=joint_local_noise(jnp.asarray(x),z) if self.index in LOCAL else joint_global_noise(z,jnp.asarray(env.positions,dtype=jnp.float32))
  else:noise=jnp.stack([jax.random.normal(jax.random.fold_in(key,i),(2,),dtype=jnp.float32) for i in range(4)])
  raw=np.stack([np.asarray(self.forward(o[i],noise[i])) for i in range(4)]);raw=np.clip(raw,-1.,1.);raw*=np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12))
  action=local_actions_to_world(np.asarray(raw,np.float64),env.positions).copy();action*=np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(action,axis=1,keepdims=True),1e-30));return x,action,None

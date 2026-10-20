"""Shared official one-step actor, with private observations and private noise."""
from core import *
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import optax
import pickle
from new_benchmark_common.macflow import ActorVectorField
from poc.field_controller import Runtime
from ring_exchange.local_frame import local_observation
jax.config.update('jax_enable_x64',False)

def network():return ActorVectorField(hidden_dims=(512,512,512,512),action_dim=2,layer_norm=False,encoder=None)
def inputs(observations,mean,scale):
    ids=jnp.broadcast_to(jnp.eye(4,dtype=jnp.float32),(*observations.shape[:-2],4,4))
    return jnp.concatenate((ids,(observations-mean)/scale),axis=-1)
class Student(Runtime):
    def __init__(self,index):
        from data import teacher
        base=teacher(index)
        self.scenario=base.scenario;self.config=base.config;self.config_cls=base.config_cls;self.env_cls=base.env_cls;self.manifest=base.manifest;self.basis=base.basis;self.cbf=base.cbf
        self._base=base;self.checkpoint=HERE/'students'/read(HERE/'TEACHERS.json')[index]['name']/'best.pkl'
        with self.checkpoint.open('rb') as f:ck=pickle.load(f)
        self.mean=ck['mean'];self.scale=ck['scale'];self.params=ck['params'];model=network()
        self.forward=jax.jit(lambda o,z:model.apply({'params':self.params},o,z))
        jax.config.update('jax_enable_x64',False)
    def nominal(self,env,key,traced=False):
        assert not traced,'one-step student has no ODE trace'
        obs=local_observation(env.positions,env.velocities,env.goals,env.config);o=np.asarray(inputs(jnp.asarray(obs),self.mean,self.scale));out=[]
        for i in range(4):
            z=jax.random.normal(jax.random.fold_in(key,i),(2,),dtype=jnp.float32)
            # Separate call: agent i has only row i and its own random stream.
            out.append(np.asarray(self.forward(o[i],z)))
        raw=np.clip(np.asarray(out),-1.,1.);raw=raw*np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12))
        action=self._base.to_world(np.asarray(raw,dtype=np.float64),env).copy();action*=np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(action,axis=1,keepdims=True),1e-30))
        return obs,action,None

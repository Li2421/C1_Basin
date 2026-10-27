"""Fresh BC pilots: original 3x256 joint architecture, explicit new-scene input.

World-frame observations and world-frame demonstrated velocities are paired.
Inactive slots in the single-robot capability query are encoder padding only;
they never enter plant geometry, task success, or the safety constraints.
"""
from experiment import *
from flow_scene import LiteratureEnv
from memo_planning import MemoNominal
from single_integrator.environment import point_segment_distance
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
from functools import lru_cache
jax.config.update('jax_enable_x64',False)

class PilotEnv(LiteratureEnv):
    def __init__(self,state,c=None):super().__init__(state,c);self.parent_agent=state.get('agent')

def observation(env,target_n):
    actual=len(env.positions);slots=list(range(actual)) if actual>1 else [int(env.parent_agent or 0)];rows=np.zeros((target_n,4*target_n+7),np.float32)
    for local,i in enumerate(slots):
        p=env.positions[local];v=env.velocities[local];relative=[]
        for j in range(target_n):
            if i==j:continue
            if j in slots:
                k=slots.index(j);relative.extend([*(env.positions[k]-p),*(env.velocities[k]-v)])
            else:relative.extend([0.,0.,0.,0.])
        if len(env.walls):
            a=env.walls[:,0];ab=env.walls[:,1]-a;t=np.clip(np.sum((p-a)*ab,-1)/np.sum(ab*ab,-1),0,1);delta=a+t[:,None]*ab-p;distance=np.linalg.norm(delta,axis=-1);j=int(distance.argmin());geometry=[*delta[j],env.config.agent_radius,11. if env.config.scene=='lit_circle_five' else 2. if env.config.scene=='lit_crowded_random' else 3.,distance[j]-env.instance.geometry.margin]
        else:geometry=[0.,0.,env.config.agent_radius,3.,0.]
        rows[i]=np.r_[p,v,env.goals[local]-p,relative,geometry]
    return rows.reshape(-1),slots

class Policy(nn.Module):
    family:str
    n:int
    @nn.compact
    def __call__(self,obs):
        x=obs
        for _ in range(3):x=nn.relu(nn.Dense(256)(x))
        d=2*self.n;z=nn.Dense(5+10*d if self.family=='mixture' else d)(x)
        if self.family=='mixture':return z[...,:5],z[...,5:5+5*d].reshape(*z.shape[:-1],5,d),jnp.clip(z[...,5+5*d:].reshape(*z.shape[:-1],5,d),-5.,1.5)
        return z

@lru_cache(None)
def load_policy(folder,expected):
    folder=Path(folder);assert sha(folder/'best.msgpack')==expected;cfg=read(folder/'CONFIG.json');normal=read(folder/'NORMALIZATION.json');net=Policy(cfg['family'],cfg['N']);dim=cfg['N']*(4*cfg['N']+7)
    pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,dim))), (folder/'best.msgpack').read_bytes())
    @jax.jit
    def sample(obs,key):
        x=(obs[None]-jnp.asarray(normal['obs_mean']))/jnp.asarray(normal['obs_scale'])
        if cfg['family']=='deterministic':z=net.apply(pa,x)[0]
        else:
            logits,means,logs=net.apply(pa,x);k1,k2=jax.random.split(key);index=jax.random.categorical(k1,logits[0]);z=means[0,index]+jnp.exp(logs[0,index])*jax.random.normal(k2,(2*cfg['N'],))
        raw=jnp.clip(jnp.asarray(normal['act_mean'])+jnp.asarray(normal['act_scale'])*z,-1.,1.).reshape(cfg['N'],2)
        return raw*jnp.minimum(1.,normal['max_speed']/jnp.maximum(jnp.linalg.norm(raw,axis=-1,keepdims=True),1e-30))
    return cfg,sample

class PilotNominal(MemoNominal):
    def action(self,env,key):
        if self.kind not in ('deterministic','mixture'):return super().action(env,key)
        cfg,fn=load_policy(self.parameters['folder'],self.parameters['checkpoint_sha256']);obs,slots=observation(env,cfg['N']);raw=np.asarray(fn(jnp.asarray(obs),key),np.float64)
        return raw[slots].copy()

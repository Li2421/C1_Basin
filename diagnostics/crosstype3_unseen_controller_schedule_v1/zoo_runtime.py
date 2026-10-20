"""Stateful new policies at the historical TT position, with per-episode reset."""
from common import *
from legacy_controllers import Nominal, bound
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from zoo_models import PolicySession, network, initialize
from flax import serialization
import jax
jax.config.update('jax_enable_x64', False)


class ZooNominal(Nominal):
    def __init__(self,payload):
        manifest=read(payload['adapter_manifest_path'])
        super().__init__(payload['controller_class'],[],manifest)
        self.frame=payload['observation_frame']
        self.checkpoint=Path(payload['flow_checkpoint_path'])
        assert sha(self.checkpoint)==payload['flow_checkpoint_sha256']
        self.normal=read(self.checkpoint.parent/'NORMALIZATION.json')
        net=network(self.kind)
        template=initialize(net,self.kind,jax.random.PRNGKey(0))
        self.parameters=serialization.from_bytes(template,self.checkpoint.read_bytes())
        self.session=PolicySession(self.kind,self.parameters,self.normal)
        self.episode_env=None

    def observation(self,env):
        return RingFlowAdapter.observation(self,env)

    def to_world(self,actions,env):
        return RingFlowAdapter.to_world(self,actions,env)

    def make_env(self,state):
        env=super().make_env(state)
        self.session.reset()
        self.episode_env=env
        return env

    def nominal(self,env,key,traced=False):
        assert not traced
        if env is not self.episode_env:
            if env.step_count!=0:
                raise ValueError('A new recurrent/chunking episode must start at t0')
            self.session.reset();self.episode_env=env
        obs=self.observation(env)
        raw=self.session.action(obs,key,env.step_count)
        action=bound(self.to_world(np.asarray(raw,np.float64),env).copy(),self.config.max_speed)
        return obs,action,None


def make(name,protocol,cache):
    kind=protocol['profiles'][name]['payload'].get('controller_class','Flow')
    if kind=='Flow' and name=='base_u_v10_local_macflow':
        if name not in cache:
            from poc.field_controller import Runtime
            cache[name]=Runtime('ring_exchange')
            assert sha(cache[name].checkpoint)==protocol['profiles'][name]['payload']['flow_checkpoint_sha256']
        jax.config.update('jax_enable_x64',False)
        return cache[name]
    if kind=='ring_ippo':
        if name not in cache:
            cache[name]=RingIPPONominal(protocol['profiles'][name]['payload'])
        return cache[name]
    if kind in ('chunking','recurrent','cvae','implicit'):
        if name not in cache:
            cache[name]=ZooNominal(protocol['profiles'][name]['payload'])
        return cache[name]
    from controllers import make as historical_make
    return historical_make(name,protocol,cache)


class RingIPPONominal(Nominal):
    def __init__(self,payload):
        from ippo import Actor,Critic,base_normalize,online_normalize,decode_action
        import jax.numpy as jnp
        manifest=read(payload['adapter_manifest_path'])
        super().__init__('ring_ippo',[],manifest)
        self.frame=payload['observation_frame']
        self.checkpoint=Path(payload['flow_checkpoint_path'])
        assert sha(self.checkpoint)==payload['flow_checkpoint_sha256']
        self.normal=read(self.checkpoint.parent/'NORMALIZATION.json')
        actor,critic=Actor(),Critic()
        template=dict(parameters=dict(actor=actor.init(jax.random.PRNGKey(0),jnp.zeros((1,23))),
                                      critic=critic.init(jax.random.PRNGKey(0),jnp.zeros((1,23)))),
                      statistics=dict(count=jnp.asarray(1e-4),mean=jnp.zeros(23),variance=jnp.ones(23)))
        loaded=serialization.from_bytes(template,self.checkpoint.read_bytes())
        def sample(obs):
            x=online_normalize(base_normalize(obs,self.normal),loaded['statistics'])
            mean,_=actor.apply(loaded['parameters']['actor'],x)
            return decode_action(mean,self.normal)
        self.sample=jax.jit(sample)

    def observation(self,env):
        return RingFlowAdapter.observation(self,env)

    def to_world(self,actions,env):
        return RingFlowAdapter.to_world(self,actions,env)

    def nominal(self,env,key,traced=False):
        assert not traced
        obs=self.observation(env)
        raw=np.asarray(self.sample(obs),np.float64)
        action=bound(self.to_world(raw,env).copy(),self.config.max_speed)
        action[np.linalg.norm(env.goals-env.positions,axis=-1)<=self.config.goal_tolerance]=0.
        return obs,action,None

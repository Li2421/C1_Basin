from core import *
import jax,jax.numpy as jnp
from poc.field_controller import Runtime
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
class FourFlowAdapter(Runtime):
 def __init__(self,checkpoint):
  super().__init__('four_way_intersection')
  jax.config.update('jax_enable_x64',False)
  self.checkpoint=Path(checkpoint)
  self.agent,_=load_checkpoint(self.checkpoint,expected_environment_fingerprint=self.manifest['environment_fingerprint'])
  c=self.agent.config
  self.scale=np.asarray(c['act_scale'],dtype=np.float64).reshape(-1,2);self.mean=np.asarray(c['act_mean'],dtype=np.float64).reshape(-1,2)
  @jax.jit
  def trace(obs,key):
   flat=obs.reshape(1,c['joint_obs_dim']);flat=(flat-jnp.asarray(c['obs_mean']))/jnp.asarray(c['obs_scale']);z=jax.random.normal(key,(1,c['joint_act_dim']));path=[z[0]]
   for k in range(self.steps):
    t=jnp.full((1,1),k/self.steps);z=z+self.agent.network.select('actor_bc_flow')(flat,z,t)/self.steps;path.append(z[0])
   return jnp.stack(path)
  @jax.jit
  def velocity(obs,z,s):
   flat=obs.reshape(1,c['joint_obs_dim']);flat=(flat-jnp.asarray(c['obs_mean']))/jnp.asarray(c['obs_scale']);return self.agent.network.select('actor_bc_flow')(flat,z.reshape(1,-1),s.reshape(1,1))[0]
  self.trace_fn,self.velocity_fn=trace,velocity;self.native_fn=jax.jit(lambda obs,key:sample_bounded_actions(self.agent,obs[None],key)[0])

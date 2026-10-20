"""Supply new nominal sources through the existing public episode cache.

No runtime loop or source-module global is modified. All physical execution,
projection, RNG and event decisions remain in the reproduced mixed runtime.
"""
from common import *
from mixed_execution import execute as existing_execute
from legacy_controllers import Nominal,bound
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from unseen_models import TransformerPolicy,sample
from flax import serialization
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
NEW_TYPES=('transformer_bc','gradient_mpc','conflict_based_search')
class TransformerNominal(Nominal):
    def __init__(self,payload):
        manifest=read(payload['adapter_manifest_path']);super().__init__('transformer_bc',[],manifest);self.frame=payload['observation_frame'];self.checkpoint=Path(payload['flow_checkpoint_path']);assert sha(self.checkpoint)==payload['flow_checkpoint_sha256'];self.normal=read(self.checkpoint.parent/'NORMALIZATION.json');self.net=TransformerPolicy(int(payload['transformer_layers']));template=self.net.init(jax.random.PRNGKey(0),jnp.zeros((1,92),jnp.float32));self.parameters=serialization.from_bytes(template,self.checkpoint.read_bytes());self.sample=jax.jit(lambda obs,key:sample(self.net,self.parameters,obs,key,self.normal))
    def observation(self,env):return RingFlowAdapter.observation(self,env)
    def to_world(self,a,env):return RingFlowAdapter.to_world(self,a,env)
    def nominal(self,env,key,traced=False):
        assert not traced;obs=self.observation(env);raw=np.asarray(self.sample(jnp.asarray(obs),key),np.float64);return obs,bound(self.to_world(raw,env).copy(),self.config.max_speed),None

def ensure(c,p,cache):
    name=c['base_controller_name'];v=p['profiles'][name]['payload'];kind=v.get('controller_class','Flow')
    if name not in cache and kind in NEW_TYPES:
        if kind=='transformer_bc':cache[name]=TransformerNominal(v)
        elif kind=='gradient_mpc':
            from gradient_mpc import GradientMPC
            cache[name]=GradientMPC(v['nominal_parameters'],p['native_manifest'])
        else:
            from conflict_based_search import ConflictBasedSearch
            cache[name]=ConflictBasedSearch(v['nominal_parameters'],p['native_manifest'])
    jax.config.update('jax_enable_x64',False)
def execute(c,p,cache,phase,probe=False):
    assert c.get('rhr_rule') is None,'Continuing right-hand rules are outside this experiment'
    ensure(c,p,cache);out=existing_execute(c,p,cache,phase,probe=probe);base=cache[c['base_controller_name']]
    if hasattr(base,'fallback_count'):out['controller_diagnostics']=clean(dict(cbs_fallback_count=base.fallback_count,cbs_planning_records=base.planning_records,replan_steps=base.replan_steps))
    elif hasattr(base,'optimizations'):out['controller_diagnostics']=dict(gradient_optimizations=base.optimizations)
    out['new_nominal_injection_sha256']=sha(__file__)
    return out

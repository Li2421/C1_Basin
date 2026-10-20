from common import *
from dataclasses import replace
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from validation_v2.adapters import FourRingRuntime
from ring_exchange.environment import Config,LocalFrameConfig,RingExchangeEnv
from poc.field_controller import Runtime
import jax
jax.config.update('jax_enable_x64',False)
class FactorAdapter(RingFlowAdapter):
 def __init__(self,checkpoint,manifest,frame,H,r):
  super().__init__(checkpoint,manifest,frame);self.config=replace(self.config,max_steps=int(H),obstacle_radius=float(r));jax.config.update('jax_enable_x64',False)
class FactorNative(FourRingRuntime):
 def __init__(self,H,r):
  super().__init__('ring_exchange');self.config=replace(self.config,max_steps=int(H),obstacle_radius=float(r));jax.config.update('jax_enable_x64',False)
def config_for(name,H,r):
 v=read(HERE/'ORIGINAL_PROFILES.json')[name]['payload']
 if name=='base_u_v10_local_macflow':base=read(HERE/'HISTORICAL_INVENTORY.json')['config'];cls=LocalFrameConfig
 else:base=read(v['adapter_manifest_path'])['scenario_config'];cls=LocalFrameConfig if v['observation_frame']=='local' else Config
 return replace(cls(**base),max_steps=int(H),obstacle_radius=float(r))
def make(name,H,r,cache):
 key=(name,H,r)
 if key not in cache:
  v=read(HERE/'ORIGINAL_PROFILES.json')[name]['payload'];cache[key]=FactorNative(H,r) if name=='base_u_v10_local_macflow' else FactorAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'],H,r);assert sha(cache[key].checkpoint)==v['flow_checkpoint_sha256']
 jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled;return cache[key]

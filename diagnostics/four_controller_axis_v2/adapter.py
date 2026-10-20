"""Round19b checkpoint-native Four adapters; no historical files are modified."""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2]
POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
sys.path[:0]=[str(ROOT),str(POC),str(POC/'source')]
import jax
import jax.numpy as jnp
import numpy as np
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
from new_benchmark_common.basin_dataset_v1 import TrainingRuntime
from new_benchmark_common.safety_eta3 import ScenarioRuntime,SCENARIOS
from poc.field_controller import Runtime

class FourFlowAdapter(Runtime):
    """E8 FF reference with each checkpoint's native normalizers/width."""
    def __init__(self,checkpoint):
        super().__init__('four_way_intersection')
        jax.config.update('jax_enable_x64',False)
        self.checkpoint=Path(checkpoint)
        self.agent,_=load_checkpoint(self.checkpoint,expected_environment_fingerprint=self.manifest['environment_fingerprint'])
        c=self.agent.config
        self.scale=np.asarray(c['act_scale'],dtype=np.float64).reshape(-1,2)
        self.mean=np.asarray(c['act_mean'],dtype=np.float64).reshape(-1,2)
        @jax.jit
        def trace(obs,key):
            flat=obs.reshape(1,c['joint_obs_dim'])
            flat=(flat-jnp.asarray(c['obs_mean']))/jnp.asarray(c['obs_scale'])
            z=jax.random.normal(key,(1,c['joint_act_dim']))
            path=[z[0]]
            for k in range(self.steps):
                t=jnp.full((1,1),k/self.steps)
                z=z+self.agent.network.select('actor_bc_flow')(flat,z,t)/self.steps
                path.append(z[0])
            return jnp.stack(path)
        @jax.jit
        def velocity(obs,z,s):
            flat=obs.reshape(1,c['joint_obs_dim'])
            flat=(flat-jnp.asarray(c['obs_mean']))/jnp.asarray(c['obs_scale'])
            return self.agent.network.select('actor_bc_flow')(flat,z.reshape(1,-1),s.reshape(1,1))[0]
        self.trace_fn,self.velocity_fn=trace,velocity
        self.native_fn=jax.jit(lambda obs,key:sample_bounded_actions(self.agent,obs[None],key)[0])
        jax.config.update('jax_enable_x64',False)
        assert not jax.config.x64_enabled

class UnregisteredTrainingRuntime(TrainingRuntime):
    """Historical physical runtime with its registry side effect disabled."""
    def _register_training_identities(self):
        pass

def intervention_runtime(checkpoint:Path):
    """Canonical v13 at t0, checkpoint-native alternate Flow thereafter."""
    jax.config.update('jax_enable_x64',False)
    rt=UnregisteredTrainingRuntime('four_way_intersection',[],parent=False)
    assert rt.checkpoint_sha != __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest()
    alternate,_=load_checkpoint(checkpoint,expected_environment_fingerprint=rt.manifest['environment_fingerprint'])
    base=rt.agent
    counter={'calls':0}
    def flow(env,key):
        rt.agent=base if counter['calls']==0 else alternate
        counter['calls']+=1
        return ScenarioRuntime.flow_world(rt,env,key)
    rt.flow_world=flow
    def reset_counter():
        counter['calls']=0
        rt.agent=base
    jax.config.update('jax_enable_x64',False)
    assert not jax.config.x64_enabled
    return rt,reset_counter

def historical_state(row:dict,index:int):
    """Rebuild the original intermediate state without changing its timers."""
    p=row['physical']
    timestep=int(round((p['max_seconds']-p['remaining_seconds'])/p['dt']))
    return {'uid':row['state_uid'],'alias':row['family'],'index':index,
            'source_group':row['family'],
            'physical':{'positions':p['positions'],'velocities':p['velocities'],
                        'goals':p['goals'],'timestep':timestep}}

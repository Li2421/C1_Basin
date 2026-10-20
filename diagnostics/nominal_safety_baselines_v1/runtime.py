"""Load frozen round27 implementation directly; add only baseline dispatch."""
from core import *
import importlib.util
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
# Frozen modules import our core solely for paths/serialization. This avoids any
# cache writes into the historical directory; controller functions are unchanged.
controllers=load('controllers',OLD/'controllers.py')
frozen=load('round27_frozen_run',OLD/'run.py')
import jax
from poc.field_controller import Runtime
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
# Importing dependencies can change this global setting: restore AFTER imports.
jax.config.update('jax_enable_x64',False)
def make(name,p,cache):
 rt=frozen.make(name,p,cache)
 jax.config.update('jax_enable_x64',False)
 assert not jax.config.x64_enabled
 return rt
def execute(c,p,cache,phase):
 import time
 rt=frozen.Observed(make(c['base_controller_name'],p,cache));s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=rt.make_env(s)
 jax.config.update('jax_enable_x64',False)
 pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];active=[];swept=[];term='timeout';error=None;steps=0;t0=time.monotonic()
 for step in range(rt.config.max_steps):
  try:
   assert not jax.config.x64_enabled
   key=key_for(s,c['seed'],step);chain=c['chain']
   if chain=='orthoflow3':
    eta=c['eta_a'] if step<c['switch_step'] else c['eta_b'];action,_,_=rt.action(env,key,np.asarray(eta),'old');intervened=rt.last.active
   else:
    _,nominal,_=rt.nominal(env,key,False)
    if chain=='no_safety':action=nominal;intervened=False
    else:
     assert chain=='hard_safety';action,_=rt.safety(env).project(nominal);intervened=rt.last.active
   steps+=1
   _,_,done,info=env.step(action);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(action.copy());active.append(intervened);cfg=rt.config
   swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
   if done:break
  except Exception as e:error=dict(step=step,type=type(e).__name__,message=str(e)[:1000]);term='numerical_failure';break
 summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','outer_collision','obstacle_collision','agent_collision'])
 trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),projection_active=np.asarray(active),swept_clearance=np.asarray(swept),goals=env.goals)
 # Descriptive stagnation only: final 100 physical steps have <0.05 total
 # goal-distance reduction and episode did not succeed. No deadlock assertion.
 err=np.linalg.norm(np.asarray(pos)-env.goals,axis=-1).sum(-1)
 stagnant=bool(len(err)>100 and err[-101]-err[-1]<.05 and term!='success')
 return dict(experiment_uid=EXP,case=c,success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),episode_length=steps,physical_steps=len(pos)-1,outcome=term,error=error,first_action=actions[0].tolist() if actions else None,summary=summary,trace_path=str(trace),trace_sha256=sha(trace),protocol_sha256=sha(HERE/'protocol.json'),controller_code_sha256=sha(OLD/'controllers.py'),execution_code_sha256=sha(__file__),jax_enable_x64=bool(jax.config.x64_enabled),stagnation_diagnostic=stagnant,minimum_swept_clearance=min(swept) if swept else None,elapsed_seconds=time.monotonic()-t0,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']})

from common import *
import time
# Imports do not execute old runners, modify globals, or install audit hooks.
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
from diagnostics.four_controller_axis_v2.adapter import FourFlowAdapter
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
import jax
jax.config.update('jax_enable_x64',False)

def make(key,p,cache):
 if key not in cache:
  sc,name=key.split('::',1);v=p['profiles'][key]['payload'];base=FourFlowAdapter(v['flow_checkpoint_path']) if sc=='four_way_intersection' else native_runtime(sc);jax.config.update('jax_enable_x64',False);assert sha(base.checkpoint)==v['flow_checkpoint_sha256'];assert clean(base.config.to_dict())==p['environments'][sc]['config']['config'];cache[key]=FlowEndpointMatchedField(base,'isotropic' if sc=='four_way_intersection' else name.removeprefix('flow_'))
 jax.config.update('jax_enable_x64',False);return cache[key]

def execute(c,p,cache,phase):
 rt=make(c['base_controller_name'],p,cache);s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];clear=[];events=[];first=None;error=None;term='timeout';steps=0;calls=0;t0=time.monotonic()
 for t in range(rt.config.max_steps):
  try:
   assert not jax.config.x64_enabled
   if c['arm']=='nominal_only':_,a,_=rt.nominal(env,key_for(s,c['seed'],t),traced=False);n=0
   else:a,_,n=rt.action(env,key_for(s,c['seed'],t),np.asarray(c['eta_a'] if t<c['switch_step'] else c['eta_b']),'field')
   calls+=n;steps+=1
   if first is None:first=np.asarray(a).tolist()
   _,_,done,info=env.step(a);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(np.asarray(a).copy());cfg=rt.config;wall=float(info.get('min_swept_wall_distance',np.inf))-getattr(cfg,'wall_collision_margin',getattr(cfg,'collision_margin',0.));pair=float(info.get('min_swept_agent_distance',np.min(info.get('swept_agent_distances',[np.inf]))))-cfg.agent_collision_margin
   if c['scene']=='ring_exchange':wall=min(float(np.min(info['swept_obstacle_distance'])),float(np.min(info['swept_outer_distance'])))-cfg.collision_margin
   clear.append(min(wall,pair));events.append(dict(step=t+1,termination=term,candidate_deadlock=bool(info.get('candidate_deadlock',False)),deadlock=bool(info.get('deadlock',False))))
   if done:break
  except Exception as ex:error=dict(step=t,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure';break
 summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','obstacle_collision','outer_collision','agent_collision']);success=int(summary['collision_free_success'] and term=='success' and error is None);physical=len(pos)-1;trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),swept_clearance=np.asarray(clear),goals=env.goals,walls=np.asarray(getattr(env,'walls',np.empty((0,2,2)))))
 v=p['profiles'][c['base_controller_name']]['payload'];return clean(dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='toy_giveway' if c['scene']=='toy_give_way' else c['scene'],scenario_uid=s['scenario_uid'],case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=c['eta_a'],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],original_uid=s['uid'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],flow_checkpoint_sha256=v['flow_checkpoint_sha256'],success=success,deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),outcome=term,episode_length=steps,physical_steps=physical,first_action=first,error=error,summary=summary,events=events,trace_path=str(trace),trace_sha256=sha(trace),execution_code_sha256=sha(HERE/'runtime.py'),jax_enable_x64=bool(jax.config.x64_enabled),elapsed_seconds=time.monotonic()-t0))

"""New arms around unchanged nominal, safety projection, dynamics and events."""
from common import *
from controllers import make,Observed
from right_hand import RightHand
from progress import ProgressSwitch
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
import jax,time
jax.config.update('jax_enable_x64',False)
def execute(c,p,cache,phase):
 base=Observed(make(c['base_controller_name'],p,cache));rt=RightHand(base,c['rhr_rule']) if c.get('rhr_rule') else base
 s=next(x for x in p['states'] if x['state_uid']==c['state_uid']);env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];swept=[];active=[];error=None;term='timeout';steps=0;before=time.monotonic();rule=ProgressSwitch(pos[0],env.goals,700,0.9681620884830543) if c['arm']=='Sprog' else None
 for step in range(rt.config.max_steps):
  try:
   assert not jax.config.x64_enabled;key=key_for(s,c['seed'],step)
   if c['arm']=='nominal_only':_,action,_=rt.nominal(env,key);intervened=False
   else:
    eta=p['eta'][rule.eta_index(step,env.positions,env.goals)] if rule else c['eta_a'] if step<c['switch_step'] else c['eta_b'];action,_,_=rt.action(env,key,np.asarray(eta,np.float64),'field' if c['placement']=='FF' else 'old');intervened=base.last.active
   steps+=1;_,_,done,info=env.step(action);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(np.asarray(action).copy());active.append(intervened);cfg=rt.config;swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
   if done:break
  except Exception as exc:error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
 summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','obstacle_collision','outer_collision','agent_collision']);trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),goals=env.goals,projection_active=active,swept_clearance=swept);v=p['profiles'][c['base_controller_name']]['payload']
 return clean(dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='ring_exchange',scenario_uid=s['scenario_uid'],case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=c['eta_a'],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],original_uid=s['uid'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),outcome=term,episode_length=steps,physical_steps=len(pos)-1,first_action=actions[0] if actions else None,error=error,summary=summary,progress_switch_step=rule.switched_at if rule else None,trace_path=str(trace),trace_sha256=sha(trace),execution_code_sha256=sha(__file__),jax_enable_x64=bool(jax.config.x64_enabled),elapsed_seconds=time.monotonic()-before))

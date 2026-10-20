from common import *
from factor_runtime import make,config_for,RingExchangeEnv
from progress import ProgressSwitch
from prefix import choose_prefix
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
import jax,time
jax.config.update('jax_enable_x64',False)

def restore(env,pos,vel,goals,t,summary=None,done=False):
 state=env.augmented_state();state.update(positions=np.asarray(pos),last_applied_velocity=np.asarray(vel),goals=np.asarray(goals),step=int(t),done=bool(done))
 for k in ['first_success_step','first_obstacle_collision_step','first_outer_collision_step','first_agent_collision_step']:
  value=(summary or {}).get(k);state[k]=value if value is not None and value<=t else None
 env.restore_augmented_state(state)

def execute(c,p,cache,phase):
 v=p['profiles'][c['profile_key']]['payload'];assert all(sha(HERE/k)==h for k,h in v['execution_source_sha256'].items())
 s=next(x for x in p['states'] if x['state_uid']==c['state_uid']);cfg=config_for(c['base_controller_name'],c['H'],c['r']) if c.get('kind')!='expert' else __import__('ring_exchange.environment',fromlist=['LocalFrameConfig']).LocalFrameConfig(**c['evaluation_config']);env=RingExchangeEnv(cfg);physical=s['physical'];env.reset(np.asarray(physical['positions']),velocities=np.asarray(physical['velocities']),goals=np.asarray(physical['goals']));pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];swept=[];error=None;term='timeout';newsteps=0;prefixsteps=0;source=None;sourcekind=None;planning=None;candidate_count=None;before=time.monotonic();rule=ProgressSwitch(pos[0],env.goals,c['H'],c['progress_threshold']) if c['arm']=='Sprog' else None
 if c.get('kind')=='expert':
  from ring_exchange.expert import CentralizedExpert
  try:
   plan=CentralizedExpert().plan(env,require_success=False);planning=time.monotonic()-before;candidate_count=plan.candidate_count
   for action in plan.actions:
    _,_,done,info=env.step(action);pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(np.asarray(action).copy());newsteps+=1;term=info['termination'];swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
    if done:break
   assert np.array_equal(np.asarray(pos),plan.positions),'Planner/replay exact trajectory mismatch'
  except Exception as ex:error=dict(step=env.step_count,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure'
 else:
  if c.get('prefix_source'):
   ref=c['prefix_source'];old=read(ref['raw_path']);assert sha(ref['raw_path'])==ref['raw_sha256'];assert sha(old['trace_path'])==old['trace_sha256'];z=np.load(old['trace_path']);pp=z['positions'];vv=z['velocities'];gg=z['goals'];assert np.array_equal(pp[0],env.positions) and np.array_equal(vv[0],env.velocities) and np.array_equal(gg,env.goals);assert old['case']['seed']==c['seed'];assert c['r']==.90,'No geometry prefix reuse';choice=choose_prefix(c,old['case'],pp,gg,old['outcome']);prefixsteps=choice['steps'];pos=list(pp[:prefixsteps+1].copy());vel=list(vv[:prefixsteps+1].copy());actions=list(vv[1:prefixsteps+1].copy());swept=list(z['swept_clearance'][:prefixsteps].copy());source=dict(raw_path=ref['raw_path'],raw_sha256=ref['raw_sha256'],trace_path=old['trace_path'],trace_sha256=old['trace_sha256'],source_rollout_uid=old['case']['rollout_uid'],source_experiment_uid=old['experiment_uid'],source_outcome=old['outcome'],source_evidence_grade=old.get('evidence_grade','DEVELOPMENT historical R10'),source_prefix_steps=prefixsteps,compatibility=choice);sourcekind=choice['kind'];restore(env,pos[-1],vel[-1],gg,prefixsteps,old['summary'],done=sourcekind in ['TERMINAL_REUSE','BUDGET_PREFIX'])
   if rule:
    for t in range(prefixsteps):rule.eta_index(t,pos[t],gg)
   if sourcekind=='UNKNOWN_REUSE':error=old['error'];term='numerical_failure'
   elif choice['termination'] is not None:term=choice['termination']
  if sourcekind not in ['TERMINAL_REUSE','BUDGET_PREFIX','UNKNOWN_REUSE']:
   rt=make(c['base_controller_name'],c['H'],c['r'],cache)
   for step in range(prefixsteps,c['H']):
    try:
     assert not jax.config.x64_enabled
     if c['arm']=='nominal_only':_,action,_=rt.nominal(env,key_for(s,c['seed'],step),traced=False)
     else:
      if rule:eta=p['eta'][rule.eta_index(step,env.positions,env.goals)]
      else:eta=c['eta_a'] if step<c['switch_step'] else c['eta_b']
      action,_,_=rt.action(env,key_for(s,c['seed'],step),np.asarray(eta,np.float64),'field')
     _,_,done,info=env.step(action);newsteps+=1;term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(np.asarray(action).copy());swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
     if done:break
    except Exception as ex:error=dict(step=step,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure';break
 summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','outer_collision','obstacle_collision','agent_collision']);trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),goals=env.goals,swept_clearance=np.asarray(swept),metadata_json=np.asarray(json.dumps(clean(dict(evaluation_config=cfg.to_dict(),source_prefix=source,new_physical_steps=newsteps,case=c)))));v=p['profiles'][c['profile_key']]['payload'];grade='DEVELOPMENT exact reanalysis ofoldR10' if newsteps==0 and source and (source['source_experiment_uid']!=EXP or source['source_evidence_grade'].startswith('DEVELOPMENT')) else 'CONFIRMATORY current-round physical evidence or exact derivation thereof'
 return clean(dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='ring_exchange',scenario_uid=s['scenario_uid'],case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=c['eta_a'],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],original_uid=s['uid'],initial_positions=physical['positions'],initial_velocities=physical['velocities'],goals=physical['goals'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),deadlock=0,timeout=int(term=='timeout'),numerical_failure=int(error is not None),outcome=term,episode_length=len(actions),physical_steps=len(pos)-1,new_physical_steps=newsteps,prefix_reused_steps=prefixsteps,prefix_provenance=source,execution_kind='NEW_OR_SUFFIX' if newsteps else sourcekind,evidence_grade=grade,first_action=actions[0] if actions else None,summary=summary,error=error,actual_config=cfg.to_dict(),progress_switch_step=rule.switched_at if rule else None,trace_path=str(trace),trace_sha256=sha(trace),execution_code_sha256=sha(HERE/'runtime.py'),jax_enable_x64=bool(jax.config.x64_enabled),planning_seconds=planning,planner_candidate_count=candidate_count,elapsed_seconds=time.monotonic()-before))

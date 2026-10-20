from common import *
from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.scenario import Config
from bottleneck_family.expert import SequentialExpert
from bottleneck_family.evaluate_safety_audit import flow_action
from new_benchmark_common.macflow import load_checkpoint
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from shared_control.basis_families import get_basis_family
from gap_controllers import potential,nominal_reference,corrected_action
import jax,time,copy
jax.config.update('jax_enable_x64',False)

def model(n,cache):
 if n not in cache:
  a=read(HERE/'ASSET_AUDIT.json')['assets'][str(n)];assert sha(a['checkpoint'])==a['checkpoint_sha256'];assert sha(a['source_manifest'])==a['source_manifest_sha256'];cache[n]=load_checkpoint(a['checkpoint'],expected_environment_fingerprint=read(a['source_manifest'])['environment_fingerprint'])[0];assert cache[n].config['num_agents']==n
 jax.config.update('jax_enable_x64',False);return cache[n]
def execute(c,p,cache,phase):
 start=time.monotonic();s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);v=p['profiles'][c['profile_key']]['payload'];assert all(sha(HERE/f)==h for f,h in v['execution_source_sha256'].items());cfg=Config(**c['evaluation_config']);env=BottleneckEnv(cfg);env.reset(np.asarray(s['physical']['positions']),np.asarray(s['physical']['goals']));assert np.array_equal(env.velocities,np.asarray(s['physical']['velocities']));pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];clear=[];timings=[];goal_fraction=[];term='timeout';error=None;provenance=None;planning=None;planner_clearance=None;newsteps=0
 if c.get('deterministic_source'):
  ref=c['deterministic_source'];old=read(ref['raw_path']);assert sha(ref['raw_path'])==ref['raw_sha256'];assert sha(old['trace_path'])==old['trace_sha256'];assert old['case']['kind']=='P2' and c['kind']=='P2' and old['case']['arm']==c['arm'] and old['case']['state_uid']==c['state_uid'];z=np.load(old['trace_path']);pos=list(z['positions']);vel=list(z['velocities']);actions=list(z['actions']);clear=list(z['swept_clearance']);timings=list(z['step_seconds']);goal_fraction=list(z['individual_goal_fraction']);term=old['outcome'];error=old['error'];provenance=dict(raw_path=ref['raw_path'],raw_sha256=ref['raw_sha256'],trace_path=old['trace_path'],trace_sha256=old['trace_sha256'],rollout_uid=old['rollout_uid'],reason='P2 deterministic action function and environment; future key unused');env.positions=np.asarray(pos[-1]).copy();env.velocities=np.asarray(vel[-1]).copy();env.step_count=len(actions);env.collided=bool(old['collision']);env.termination=term;env.done=term!='numerical_failure'
 elif c['kind']=='expert':
  try:
   plan=SequentialExpert(grid_step=.25,clearance_buffer=.015).rollout(env);planning=time.monotonic()-start;planner_clearance=plan.get('min_clearance');pos=list(plan['states']);actions=list(plan['actions']);vel=[np.zeros(env.action_shape)]+[np.asarray(x).copy() for x in actions];newsteps=len(actions);term='success' if plan['success'] else plan['reason'];clear=[];goal_fraction=[float(np.mean(np.linalg.norm(env.goals-x,axis=1)<=cfg.goal_tolerance)) for x in pos[1:]]
   if term not in ['success','collision','timeout']:term='reference_unresolved'
  except Exception as ex:error=dict(step=env.step_count,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure';pos=[x.copy() for x in env.history];vel=[np.zeros(env.action_shape)]+[(pos[t+1]-pos[t])/cfg.dt for t in range(len(pos)-1)];actions=vel[1:];newsteps=len(actions)
 else:
  actor=model(c['N'],cache) if c['kind']=='Flow' else None;projector=CertifiedHardSafetyFilter(HardProjectionConfig());basis=get_basis_family('orthoflow3');history=c.get('historical');root=jax.random.fold_in(jax.random.PRNGKey(history['root_seed']),history['history_index']) if history else jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026093011),int(hashlib.sha256(s['state_uid'].encode()).hexdigest()[:8],16)),c['seed'])
  for step in range(cfg.max_steps):
   before=time.monotonic()
   try:
    assert not jax.config.x64_enabled
    reference=flow_action(actor,env,root,step,1,observation_version='competence_v2',latent_mode='per_step') if actor is not None else potential(env)
    if not history or history['goal_stop']:reference=nominal_reference(reference,env)
    if c['arm']=='nominal_only':action=reference
    elif c.get('validation_kind')=='flow_primitive':action=np.asarray(projector(env.snapshot(),reference).velocity,np.float64)
    else:action,_,_=corrected_action(env,reference,c['eta_a'] if step<c['switch_step'] else c['eta_b'],projector,basis)
    _,done,info=env.step(action,diagnose_stalls=False);newsteps+=1;pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(np.asarray(action).copy());clear.append(info['swept_clearance']);goal_fraction.append(float(np.mean(np.linalg.norm(env.goals-env.positions,axis=1)<=cfg.goal_tolerance)));timings.append(time.monotonic()-before);term=info['termination']
    if done:break
   except Exception as ex:error=dict(step=step,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure';break
 trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions).reshape((-1,c['N'],2)),goals=env.goals,walls=env.walls,swept_clearance=np.asarray(clear),individual_goal_fraction=np.asarray(goal_fraction),step_seconds=np.asarray(timings),metadata_json=np.asarray(json.dumps(clean(dict(case=c,config=cfg.to_dict(),deterministic_source=provenance)))));success=int(term=='success' and not env.collided and error is None);source_root=c['historical']['root_seed'] if c.get('historical') else 2026093011
 return clean(dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario=f"Gap1_N{c['N']}",scenario_uid=s['scenario_uid'],case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=c['eta_a'],future_index=c['seed'],future_root=source_root,original_uid=s['uid'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),success=success,collision=int(env.collided),deadlock=0,timeout=int(term=='timeout'),numerical_failure=int(error is not None),outcome=term,error=error,episode_length=len(actions),physical_steps=len(pos)-1,new_physical_steps=newsteps,deterministic_reused_steps=len(actions) if provenance else 0,deterministic_source=provenance,first_action=actions[0] if actions else None,final_goal_fraction=float(np.mean(np.linalg.norm(env.goals-env.positions,axis=1)<=cfg.goal_tolerance)),actual_config=cfg.to_dict(),trace_path=str(trace),trace_sha256=sha(trace),execution_code_sha256=sha(HERE/'gap_runtime.py'),jax_enable_x64=bool(jax.config.x64_enabled),elapsed_seconds=time.monotonic()-start,planning_seconds=planning,minimum_swept_clearance=min(clear) if clear else planner_clearance,evidence_grade='CONFIRMATORY current preregistered evidence',layer5_deadlock_resolution='not_established'))

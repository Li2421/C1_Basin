"""Gap native execution with only the nominal provider replaced by distributed shooting."""
from common import *
from gap_runtime import BottleneckEnv,Config,CertifiedHardSafetyFilter,HardProjectionConfig,get_basis_family
from gap_controllers import nominal_reference,corrected_action,bound
from mpc_general import distributed_shooting
import jax,time
jax.config.update('jax_enable_x64',False)
def rootkey(s,seed):return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026093011),int(hashlib.sha256(s['state_uid'].encode()).hexdigest()[:8],16)),seed)
def restore(env,pos,vel):
 env.positions=np.asarray(pos[-1]).copy();env.velocities=np.asarray(vel[-1]).copy();env.step_count=len(pos)-1;env.done=False;env.termination='running';env.collided=False;env.history=[np.asarray(x).copy() for x in pos]
def action_for(env,root,step,c,projector,basis):
 reference=nominal_reference(bound(distributed_shooting(env,jax.random.fold_in(root,step),c['parameters']),env.config.max_speed),env)
 if c['arm']=='nominal_only':return reference
 return corrected_action(env,reference,c['eta_a'] if step<c['switch_step'] else c['eta_b'],projector,basis)[0]
def episode(c,p,phase,timing=False):
 started=time.monotonic();s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);v=p['profiles'][c['profile_key']]['payload'];assert all(sha(HERE/f)==h for f,h in v['execution_source_sha256'].items());env=BottleneckEnv(Config(**c['evaluation_config']));env.reset(np.asarray(s['physical']['positions']),np.asarray(s['physical']['goals']));assert np.array_equal(env.velocities,s['physical']['velocities']);pos=[env.positions.copy()];vel=[env.velocities.copy()];acts=[];clear=[];goal=[];wall=[];cpu=[];term='timeout';error=None;reuse=None;newsteps=0;root=rootkey(s,c['seed']);projector=CertifiedHardSafetyFilter(HardProjectionConfig());basis=get_basis_family('orthoflow3')
 if c.get('prefix_source'):
  ref=c['prefix_source'];old=read(ref['raw_path']);assert sha(ref['raw_path'])==ref['raw_sha256'];assert sha(old['trace_path'])==old['trace_sha256'];oc=old['case'];assert all(oc[k]==c[k] for k in ['state_uid','seed','parameters','evaluation_config','N']);assert oc['arm']==('F13' if c['arm'].startswith('S13') else 'F15');assert oc['eta_a']==c['eta_a'];cut=c['switch_step'];terminal_before=old['outcome']!='numerical_failure' and old['physical_steps']<=cut;unknown_before=old['outcome']=='numerical_failure' and old['physical_steps']<cut;n=old['physical_steps'] if terminal_before or unknown_before else cut;assert old['physical_steps']>=n
  with np.load(old['trace_path']) as z:pos=list(z['positions'][:n+1]);vel=list(z['velocities'][:n+1]);acts=list(z['actions'][:n]);clear=list(z['swept_clearance'][:n]);goal=list(z['individual_goal_fraction'][:n]);wall=list(z['step_seconds'][:n]);cpu=list(z['step_cpu_seconds'][:n])
  restore(env,pos,vel);reuse=dict(source_rollout_uid=old['rollout_uid'],raw_sha256=ref['raw_sha256'],trace_sha256=old['trace_sha256'],reused_steps=n,full_terminal_reuse=terminal_before,preserved_unknown=unknown_before)
  if terminal_before or unknown_before:term=old['outcome'];error=old['error'];env.termination=term;env.collided=bool(old['collision']);env.done=True
 stop=min(c['H'],200) if timing else c['H']
 for step in range(len(acts),stop):
  if env.done:break
  before=time.monotonic();before_cpu=time.process_time()
  try:
   assert not jax.config.x64_enabled;action=action_for(env,root,step,c,projector,basis);_,done,info=env.step(action,diagnose_stalls=False);newsteps+=1;pos.append(env.positions.copy());vel.append(env.velocities.copy());acts.append(np.asarray(action).copy());clear.append(info['swept_clearance']);goal.append(float(np.mean(np.linalg.norm(env.goals-env.positions,axis=1)<=env.config.goal_tolerance)));wall.append(time.monotonic()-before);cpu.append(time.process_time()-before_cpu);term=info['termination']
   if done:break
  except Exception as ex:error=dict(step=step,type=type(ex).__name__,message=str(ex)[:600]);term='numerical_failure';break
 trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(acts).reshape(-1,c['N'],2),goals=env.goals,walls=env.walls,swept_clearance=np.asarray(clear),individual_goal_fraction=np.asarray(goal),step_seconds=np.asarray(wall),step_cpu_seconds=np.asarray(cpu));common=dict(experiment_uid=EXP,case=c,trace_path=str(trace),trace_sha256=sha(trace),jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started)
 if timing:return clean(dict(**common,record_type='timing_only_summary_no_outcome_labels',physical_steps=len(acts),warmup_steps=20,warmed_step_CPU_seconds=cpu[20:],cold_CPU_seconds=sum(cpu[:20]),measurement_error=error,result_labels_excluded=True))
 return clean(dict(**common,cache_identity_schema='registered_exact_uid_v1',scenario=f"Gap1_N{c['N']}",scenario_uid=s['scenario_uid'],**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=c['eta_a'],future_index=c['seed'],future_root=2026093011,original_uid=s['uid'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],flow_checkpoint_sha256=None,success=int(term=='success' and not env.collided and error is None),collision=int(env.collided),deadlock=0,timeout=int(term=='timeout'),numerical_failure=int(error is not None),outcome=term,error=error,episode_length=len(acts),physical_steps=len(acts),new_physical_steps=newsteps,exact_prefix=reuse,first_action=acts[0] if acts else None,final_goal_fraction=float(np.mean(np.linalg.norm(env.goals-env.positions,axis=1)<=env.config.goal_tolerance)),actual_config=env.config.to_dict(),minimum_swept_clearance=min(clear) if clear else None,evidence_grade='CONFIRMATORY'))
def execute(c,p,cache,phase):return episode(c,p,phase)

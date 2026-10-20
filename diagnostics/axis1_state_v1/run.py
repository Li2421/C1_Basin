from common import *
from historical_runner import make_runtime
from arms import action
import time
from shared_rollout_db.src.cache_writer import append_journal

def execute(phase,index,workers):
 prereg();p=read(HERE/'protocol.json');assert read(HERE/f'{phase}_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
 if phase=='main':assert read(HERE/'CONSISTENCY_GATE.json')['passed'];assert (HERE/'CHOICES_FROZEN.receipt.json').exists()
 from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
 import jax
 jax.config.update('jax_enable_x64',False)
 states={s['state_uid']:s for s in p['states']};cache={};rows=read(HERE/f'{phase}_MISSING_CASES.json')[index::workers];buffer=[];start=time.monotonic()
 for number,c in enumerate(rows):
  out=HERE/phase/'raw'/f"{c['rollout_uid']}.json";out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():continue
  rt=make_runtime(c['base_controller_name'],p,cache);jax.config.update('jax_enable_x64',False);s=states[c['state_uid']];env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];first=None;error=None;calls=0;steps=0;term='timeout';before=time.monotonic();clearance=[]
  for step in range(rt.config.max_steps):
   try:
    a,n=action(rt,env,key_for(s,c['seed'],step),c,step);calls+=n;steps+=1
    if first is None:first=np.asarray(a).tolist()
    _,_,done,info=env.step(a);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());cfg=rt.config
    clearance.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
    if done:break
   except Exception as exc:error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
  summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
  trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),goals=env.goals,steps=np.arange(len(pos)),swept_clearance=clearance)
  r=dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='ring_exchange',scenario_uid=s['scenario_uid'],**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},case=c,eta=c['eta_a'],future_index=c['seed'],future_root=2026100403,rng_namespace=s['rng_namespace'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],flow_checkpoint_sha256=p['base_profiles'][c['base_controller_name']]['payload']['flow_checkpoint_sha256'],success=int(summary['collision_free_success'] and term=='success' and error is None),deadlock=0,timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),episode_length=steps,physical_steps=len(pos)-1,outcome=term,first_action=first,error=error,summary=summary,trace_path=str(trace),trace_sha256=sha(trace),elapsed_seconds=time.monotonic()-before,jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=sha(__file__))
  assert not r['jax_enable_x64'];write(out,r);buffer.append(clean(r))
  if len(buffer)>=8:append_journal(buffer,EXP,f'{phase}_{index}');buffer=[]
  print(json.dumps(dict(phase=phase,index=index,completed=number+1,assigned=len(rows),elapsed_seconds=time.monotonic()-start)),flush=True)
 if buffer:append_journal(buffer,EXP,f'{phase}_{index}')
 write(HERE/phase/f'worker_{index}.json',dict(complete=True,assigned=len(rows),utc=utc()))
if __name__=='__main__':execute(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

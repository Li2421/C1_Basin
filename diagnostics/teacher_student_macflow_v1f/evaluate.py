from core import *
from guard import guard
from student import Student
from pipeline import decentralized_action
from fair_rhr import FairRuntime
from shared_rollout_db.src.cache_writer import append_journal
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from audit_utils import source_inventory
import time
G=guard()
def worker(phase,shard,shards):
 for p,h in read(HERE/f'{phase}_PREFLIGHT.receipt.json')['files'].items():assert sha(p)==h,p
 p=read(HERE/f'{phase}_protocol.json');cache={}
 for c in read(HERE/f'{phase}_MISSING_CASES.json')[shard::shards]:
  dest=HERE/phase/'raw'/f"{c['rollout_uid']}.json";trace=dest.with_suffix('.npz');assert not dest.exists() and not trace.exists(),'Never retry unknown or overwrite evidence'
  name=c['base_controller_name']
  if name not in cache:cache[name]=Student(c['student_index'],c['variant'],c['checkpoint'])
  rt=cache[name];assert sha(rt.checkpoint)==c['checkpoint_sha256'];fair=FairRuntime(rt,dict(mode='progress',steps=40,speed=.25));s=p['states'][c['state_index']];env=rt.make_env(s);initial=clean(env.augmented_state());ih=hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest();pos=[env.positions.copy()];vel=[env.velocities.copy()];acts=[];counts=np.zeros(4,int);error=None;term='timeout';calls=0;t0=time.time()
  for t in range(rt.config.max_steps):
   try:
    key=key_for(s,c['seed'],t);eta=c['eta_a'] if t<c['switch_step'] else c['eta_b']
    if c['arm']=='nominal only':a=rt.nominal(env,key)[1]
    elif c['arm']=='RHR-fair':a,cert,n=fair.action(env,key,[0.,0.,0.]);calls+=n
    else:a,d=decentralized_action(rt,env,key,eta,'local',counts);calls+=d['projection_calls']
    _,_,done,info=env.step(a);term=info['termination'];acts.append(a.copy());pos.append(env.positions.copy());vel.append(env.velocities.copy())
    if done:break
   except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e)[:800]);term='numerical_failure';break
  summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','outer_collision','obstacle_collision','agent_collision']);trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=pos,velocities=vel,actions=acts,goals=env.goals,initial_augmented_state_json=np.array(json.dumps(initial)),rhr_active=fair.active_log)
  r=dict(schema='round30f_seed_v1',experiment_uid=p['experiment_uid'],cache_identity_schema='registered_exact_uid_v1',case=c,**{k:c[k] for k in ['rollout_uid','state_uid','eta_uid','controller_uid','seed_key']},scenario_uid=p['base_profiles'][name]['payload']['scenario_uid'],flow_checkpoint_sha256=c['checkpoint_sha256'],eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=int(term=='deadlock'),numerical_failure=int(error is not None),outcome=term,error=error,episode_length=len(acts),physical_steps=len(acts),first_action=acts[0].tolist() if acts else None,summary=summary,trace_path=str(trace),trace_sha256=sha(trace),initial_state_sha256=ih,phase_protocol_sha256=sha(HERE/f'{phase}_protocol.json'),jax_enable_x64=False,projection_calls=calls,elapsed_seconds=time.time()-t0)
  guard();write(dest,r);journal=append_journal([r],p['experiment_uid'],f'{phase}_{shard}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');print(phase,shard,name,c['state_index'],c['arm'],term,flush=True)
 write(HERE/phase/f'worker{shard}.json',dict(utc=utc(),complete=True,guard=guard()));source_inventory(f'imports_evaluate_{phase}_{shard}.json')
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

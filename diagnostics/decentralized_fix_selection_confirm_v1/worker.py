from runtime_new import *
from pipeline import decentralized_action,LocalSafety,local_basis
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from audit_utils import source_inventory
from dataclasses import replace
import time,fcntl,subprocess
G=guard()
def queue(phase):
 rows=read(HERE/f'{phase}_MISSING_CASES.json');q=HERE/phase/'queue';q.mkdir(parents=True,exist_ok=True)
 while True:
  with (q/'cursor').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
  if i>=len(rows):return
  yield rows[i]
def execute(c,p,cache,phase):
 rt=make(c['base_controller_name'],p,cache);s=p['states'][c['state_index']];env=rt.make_env(s);probe=phase=='probe'
 if probe:env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False)
 pos=[env.positions.copy()];vel=[env.velocities.copy()];acts=[];active=[];term='timeout';error=None;t0=time.monotonic();initial=clean(env.augmented_state());fw=fair(rt,c['rhr_rule']) if c['rhr_rule'] else None
 for t in range(200 if probe else rt.config.max_steps):
  try:
   key=key_for(s,c['seed'],t);eta=c['eta_a'] if t<c['switch_step'] else c['eta_b'];guard()
   if c['arm']=='nominal only':a=rt.nominal(env,key)[1]
   elif fw:a,_,_=fw.action(env,key,[0.,0.,0.])
   else:a,_=decentralized_action(rt,env,key,eta,'local',np.zeros(4,int))
   if probe:
    nom=rt.nominal(env,key)[1];fs=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nom[i:i+1])[0][0] for i,f in enumerate(fs)]);target=safe.copy()
    for i in range(4):target[i]+=np.asarray(eta)@local_basis(env.positions,env.goals,safe,i,1.5,env.config.max_speed)
    active.append(np.linalg.norm(safe-nom)>1e-6 or np.linalg.norm(a-target)>1e-6)
   _,_,done,info=env.step(a);acts.append(np.asarray(a).copy());pos.append(env.positions.copy());vel.append(env.velocities.copy())
   if not probe:
    term=info['termination']
    if done:break
  except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e)[:800]);term='numerical_failure';break
 dest=HERE/phase/'raw'/f"{c['rollout_uid']}.json";trace=dest.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=pos,velocities=vel,actions=acts,goals=env.goals,projection_active=active,initial_augmented_state_json=np.array(json.dumps(initial)))
 r=dict(schema='round30e_unlabeled_probe_v1' if probe else 'round30e_seed_v1',experiment_uid=p['experiment_uid'],case=c,**{k:c[k] for k in ['rollout_uid','state_uid','eta_uid','controller_uid','seed_key']},complete=len(acts)==200 if probe else error is None,error=error,physical_steps=len(acts),trace_path=str(trace),trace_sha256=sha(trace),jax_enable_x64=False,elapsed_seconds=time.monotonic()-t0)
 if probe:r.update(terminal_labels_read=False)
 else:
  summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','outer_collision','obstacle_collision','agent_collision']);v=p['profiles'][c['base_controller_name']]['payload'];r.update(cache_identity_schema='registered_exact_uid_v1',scenario_uid=s['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=int(term=='deadlock'),numerical_failure=int(error is not None),outcome=term,episode_length=len(acts),first_action=acts[0].tolist() if acts else None,summary=summary,phase_protocol_sha256=sha(HERE/f'{phase}_protocol.json'))
 guard();write(dest,r);journal=cache_writer.append_journal([r],p['experiment_uid'],f'{phase}_{os.environ.get("SLURM_ARRAY_TASK_ID","0")}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');return r

def main():
 phase=sys.argv[1];index=int(sys.argv[2]);prereg()
 for pth,h in read(HERE/f'{phase}_PREFLIGHT.receipt.json')['files'].items():assert sha(pth)==h,pth
 sq=subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-t','RUNNING','-o','%C|%b'],text=True);cpu=sum(int(x.split('|')[0]) for x in sq.splitlines() if 'shard' not in x.split('|')[1]);assert cpu<=18,sq
 p=read(HERE/f'{phase}_protocol.json');cache={};count=0
 for c in queue(phase):
  dest=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
  if dest.exists():continue
  assert not dest.with_suffix('.npz').exists(),'Partial evidence preserved; do not retry'
  r=execute(c,p,cache,phase);count+=1;print(phase,index,count,c['base_controller_name'],c['state_index'],c['arm'],r.get('outcome','probe'),flush=True)
 write(HERE/phase/f'worker{index}.json',dict(utc=utc(),complete=True,new=count,guard=guard()));source_inventory(f'imports_worker_{phase}_{index}.json')
if __name__=='__main__':main()

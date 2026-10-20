from common import *
from geometry import GeneratedGeometry,GeneratedEnv
from bottleneck_family.scenario import Config
from sequential_expert import SequentialExpert
import fcntl,time

def worker(stage,index):
 assert read(HERE/'REPRODUCTION_GATE.json')['passed']
 cases=read(HERE/f'{stage}_MISSING_CASES.json');folder=HERE/stage;folder.mkdir(exist_ok=True)
 while True:
  with (folder/'cursor').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
  if i>=len(cases):break
  c=cases[i];s=c['state'];path=folder/'raw'/f"{c['rollout_uid']}.json"
  if path.exists():continue
  geo=GeneratedGeometry(Config(**c['config']),s['rectangles']);env=GeneratedEnv(geo,s['positions'],s['goals']);start=time.monotonic();error=None
  try:
   plan=SequentialExpert(.25,.015).rollout(env);term=plan['reason'];positions=np.asarray(plan['states']);actions=np.asarray(plan['actions']).reshape(-1,4,2)
   if term not in ['success','collision','timeout']:error=dict(type='ReferenceUnresolved',message=term)
  except Exception as e:
   term='numerical_failure';error=dict(type=type(e).__name__,message=str(e));positions=np.asarray(env.history);actions=np.diff(positions,axis=0)/env.config.dt
  trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,goals=env.goals,walls=env.walls)
  result=dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='Generated '+s['family'],**{k:c[k] for k in ['scenario_uid','state_uid','controller_uid','eta_uid','rollout_uid','seed_key']},eta=c['eta'],future_index=c['seed'],future_root=2026101740,initial_positions=s['positions'],initial_velocities=s['velocities'],goals=s['goals'],flow_checkpoint_sha256=None,success=int(term=='success' and error is None),collision=int(env.collided),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),outcome=term,error=error,episode_length=len(actions),first_action=actions[0] if len(actions) else None,trace_path=str(trace),trace_sha256=sha(trace),case=c,elapsed_seconds=time.monotonic()-start)
  write(path,result);j=cache_writer.append_journal([clean(result)],EXP,f'joint_{index}');assert j.is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
  print(i,s['family'],term,flush=True)
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))

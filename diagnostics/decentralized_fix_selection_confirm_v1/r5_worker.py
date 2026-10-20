from support import *
from worker import queue
import subprocess
G=guard()
def main():
 prereg();index=int(sys.argv[1]);p=read(HERE/'probe_protocol.json');cache={};count=0
 for path,h in read(HERE/'probe_PREFLIGHT.receipt.json')['files'].items():assert sha(path)==h
 for c in queue('probe'):
  dest=HERE/'probe/raw'/f"{c['rollout_uid']}.json"
  if dest.exists():continue
  assert not dest.with_suffix('.npz').exists(),'Preserve partial probes; no retry'
  rt=runtime(c['base_controller_name'],p,cache);s=p['states'][c['state_index']];env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False);pos=[env.positions.copy()];vel=[env.velocities.copy()];acts=[];active=[];error=None;eta=p['eta'][c['eta_index']]
  for t in range(200):
   try:
    guard();key=key_for(s,0,t);a=decentralized_action(rt,env,key,eta,'local',np.zeros(4,int))[0];nom=rt.nominal(env,key)[1];fs=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nom[i:i+1])[0][0] for i,f in enumerate(fs)]);target=safe.copy()
    for i in range(4):target[i]+=np.asarray(eta)@local_basis(env.positions,env.goals,safe,i,1.5,env.config.max_speed)
    active.append(np.linalg.norm(safe-nom)>1e-6 or np.linalg.norm(a-target)>1e-6);env.step(a);acts.append(a.copy());pos.append(env.positions.copy());vel.append(env.velocities.copy())
   except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e));break
  trace=dest.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=pos,velocities=vel,actions=acts,goals=env.goals,projection_active=active);write(dest,dict(case=c,complete=len(acts)==200,error=error,terminal_labels_read=False,task_journal_written=False,trace_path=str(trace),trace_sha256=sha(trace),guard=guard()));count+=1;print(index,count,c['base_controller_name'],c['state_index'],c['eta_index'],flush=True)
 write(HERE/'probe'/f'worker{index}.json',dict(complete=True,new=count,utc=utc(),guard=guard()));source_inventory(f'imports_r5_{index}.json')
if __name__=='__main__':main()

from support import *
from worker import execute,queue
import subprocess,time
G=guard()
FEATURES=['freeze_proportion','moving_no_progress_robot_seconds','progress100','progress200','progress300','intervention','mixed_proportion']
def diagnostic(c,p,cache):
 rt=runtime(c['base_controller_name'],p,cache);s=p['states'][c['state_index']];env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False);pos=[env.positions.copy()];vel=[env.velocities.copy()];acts=[];nom=[];error=None
 for t in range(300):
  try:
   guard();key=key_for(s,0,t);a=decentralized_action(rt,env,key,[0.,0.,0.],'local',np.zeros(4,int))[0];n=rt.nominal(env,key)[1];env.step(a);pos.append(env.positions.copy());vel.append(env.velocities.copy());acts.append(a.copy());nom.append(n.copy())
  except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e));break
 dest=HERE/'diagnostic/raw'/f"{c['rollout_uid']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,positions=pos,velocities=vel,actions=acts,nominal=nom,goals=env.goals)
 r=dict(case=c,complete=len(acts)==300,error=error,terminal_labels_read=False,task_journal_written=False,trace_path=str(dest),trace_sha256=sha(dest),guard=guard())
 if r['complete']:
  pos=np.asarray(pos);vel=np.asarray(vel);dist=np.linalg.norm(env.goals-pos,axis=-1);speed=np.linalg.norm(vel,axis=-1);unfinished=dist[1:]>env.config.goal_tolerance;freeze=(speed[1:]<.01)&unfinished;move=(speed[40:]>=.01)&(dist[:-40]-dist[40:]<.05);total=dist.sum(-1);angles=np.unwrap(np.arctan2(pos[...,1],pos[...,0]),axis=0);delta=angles-angles[0];mixed=(delta[1:]>.15).any(-1)&(delta[1:]<-.15).any(-1)
  values=[freeze.mean(),move.sum()*.05,*[total[0]-total[t] for t in [100,200,300]],np.linalg.norm(np.asarray(acts)-np.asarray(nom),axis=-1).mean(),mixed.mean()];r.update(features=dict(zip(FEATURES,values)),unfinished_conditional_freeze=freeze.sum()/max(unfinished.sum(),1),unfinished_moving_no_progress_seconds=float((move&(dist[40:]>env.config.goal_tolerance)).sum()*.05))
 write(dest.with_suffix('.json'),r);return r
def main():
 phase=sys.argv[1];index=int(sys.argv[2]);prereg();assert read(HERE/'REPLAY_GATE.json')['passed']
 for path,h in read(HERE/f'{phase}_PREFLIGHT.receipt.json')['files'].items():assert sha(path)==h,path
 q=subprocess.check_output(['squeue','-h','-u','zhihan','-t','RUNNING','-o','%C|%b'],text=True);assert sum(int(x.split('|')[0]) for x in q.splitlines() if 'shard' not in x.split('|')[1])<=18
 p=read(HERE/f'{phase}_protocol.json');cache={};count=0
 for c in queue(phase):
  dest=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
  if dest.exists():continue
  assert not dest.with_suffix('.npz').exists(),'Partial evidence preserved, no retry'
  runtime(c['base_controller_name'],p,cache);r=diagnostic(c,p,cache) if phase=='diagnostic' else execute(c,p,cache,phase);count+=1;print(phase,index,count,c['base_controller_name'],c['state_index'],c.get('arm'),r.get('outcome','diagnostic'),flush=True)
 write(HERE/phase/f'worker{index}.json',dict(utc=utc(),complete=True,new=count,guard=guard()));source_inventory(f'imports_worker_{phase}_{index}.json')
if __name__=='__main__':main()

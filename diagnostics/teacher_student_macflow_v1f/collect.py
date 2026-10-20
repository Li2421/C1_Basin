from core import *
from guard import guard
from student import Student
from pipeline import decentralized_action
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from audit_utils import source_inventory
import time
G=guard()
def checkpoint(index,variant,rd):
 if variant=='V1' and rd==1:return Path(read(HERE/'STUDENTS.json')[str(index)]['checkpoint'])
 return HERE/'students'/variant/str(index)/('offline.pkl' if rd==1 else f'round{rd-1}.pkl')
def collect(index,variant,rd):
 assert read(HERE/'ROUND28_REPLAY_GATE.json')['passed']
 receipt=read(HERE/'IMPLEMENTATION_FROZEN.receipt.json')
 for p,h in receipt['files'].items():assert sha(p)==h,p
 ck=checkpoint(index,variant,rd);rt=Student(index,variant,ck);states=read(HERE/'TRAIN_STATES.json')[rd-1];out=HERE/'dagger'/variant/str(index)/f'round{rd}';out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 stats=[]
 for si,s in enumerate(states):
  dest=out/f'episode{si:02}.json';trace=dest.with_suffix('.npz')
  if dest.exists():
   row=read(dest);assert sha(trace)==row['trace_sha256'];stats.append(row);continue
  assert not trace.exists(),'partial artifact: preserve and record gap'
  env=rt.env_cls(rt.config);p=s['physical'];env.reset(np.asarray(p['positions']),velocities=np.asarray(p['velocities']),goals=np.asarray(p['goals']));xs=[];os=[];keys=[];positions=[];velocities=[];steps=[];actions=[];counts=np.zeros(4,int);error=None;term='timeout';t0=time.time()
  for t in range(rt.config.max_steps):
   try:
    key=key_for(s,0,t)
    if t%5==0:
     from ring_exchange.local_frame import local_observation
     xs.append(local_observation(env.positions,env.velocities,env.goals,env.config));os.append(rt._base.observation(env));keys.append(np.asarray(key));positions.append(env.positions.copy());velocities.append(env.velocities.copy());steps.append(t)
    a=rt.nominal(env,key)[1] if s['collection_arm']=='nominal only' else decentralized_action(rt,env,key,[0.,0.,0.],'local',counts)[0]
    actions.append(a);_,_,done,info=env.step(a);term=info['termination']
    if done:break
   except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e)[:800]);term='unknown';break
  np.savez_compressed(trace,local=np.asarray(xs,np.float32),teacher=np.asarray(os,np.float32),keys=np.asarray(keys,np.uint32),positions=positions,velocities=velocities,goals=env.goals,steps=steps,actions=actions)
  row=dict(state=s,collection_index=si,student_index=index,variant=variant,round=rd,checkpoint=str(ck),checkpoint_sha256=sha(ck),samples=len(xs),physical_steps=len(actions),termination=term,error=error,trace_sha256=sha(trace),trace_path=str(trace),elapsed=time.time()-t0,evaluation_label=False);write(dest,row);stats.append(row);print(variant,index,rd,si,term,len(xs),flush=True)
 write(out/'COMPLETE.json',dict(utc=utc(),student_index=index,variant=variant,round=rd,episodes=len(stats),samples=sum(r['samples'] for r in stats),unknown=sum(r['error'] is not None for r in stats),sources=stats,guard=guard()));source_inventory(f'imports_collect_{variant}_{index}_{rd}.json')
if __name__=='__main__':collect(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))

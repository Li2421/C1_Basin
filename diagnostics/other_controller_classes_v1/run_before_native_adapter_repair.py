from core import *
import time
from controllers import make
import jax
from poc.field_controller import Runtime
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for,rollout_tt
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE
jax.config.update('jax_enable_x64',False)
EXP='exp_other_controller_classes_v1'
class Tracker:
 def __init__(self,base):self.base=base;self.active=False
 def __getattr__(self,k):return getattr(self.base,k)
 def project(self,target):
  a,d=self.base.project(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
class Observed:
 def __init__(self,base):self.base=base;self.last=None
 def __getattr__(self,k):return getattr(self.base,k)
 def safety(self,env):self.last=Tracker(self.base.safety(env));return self.last
 def prepare(self,*a,**kw):return Runtime.prepare(self,*a,**kw)
 def action(self,env,key,eta,chain='old'):return Runtime.action(self,env,key,eta,chain)
def execute(c,p,cache):
 rt=Observed(make(c['base_controller_name'],p,cache));s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];active=[];swept=[];term='timeout';error=None;calls=0;steps=0;t0=time.monotonic()
 for step in range(rt.config.max_steps):
  try:
   assert not jax.config.x64_enabled;eta=c['eta_a'] if step<c['switch_step'] else c['eta_b'];action,_,n=rt.action(env,key_for(s,c['seed'],step),np.asarray(eta),'old');calls+=n;steps+=1
   _,_,done,info=env.step(action);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(action.copy());active.append(rt.last.active);cfg=rt.config;swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
   if done:break
  except Exception as e:error=dict(step=step,type=type(e).__name__,message=str(e)[:500]);term='numerical_failure';break
 summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','outer_collision','obstacle_collision','agent_collision']);phase=p['phase'];trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),projection_active=np.asarray(active),swept_clearance=np.asarray(swept),goals=env.goals)
 return dict(schema='round27_native_TT_v1',experiment_uid=EXP,case=c,success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=int(term=='deadlock'),numerical_failure=int(error is not None),episode_length=steps,physical_steps=len(pos)-1,outcome=term,error=error,first_action=actions[0].tolist() if actions else None,summary=summary,trace_path=str(trace),trace_sha256=sha(trace),protocol_sha256=sha(HERE/f'{phase}_protocol.json'),controller_code_sha256=sha(HERE/'controllers.py'),execution_code_sha256=sha(__file__),jax_enable_x64=bool(jax.config.x64_enabled),projection_calls=calls,elapsed_seconds=time.monotonic()-t0,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']})
def worker(phase,index,count):
 prereg();p=read(HERE/f'{phase}_protocol.json');assert sha(__file__)==p['runner_sha256'];assert sha(HERE/'controllers.py')==p['controller_code_sha256'];assert read(HERE/f'{phase}_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
 if phase=='main':
  for filename in ['R5_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json']:
   for path,digest in read(HERE/filename)['files'].items():assert sha(path)==digest
 else:assert (HERE/'CONTROLLER_IMPLEMENTATION_FROZEN.receipt.json').exists()
 cache={};rows=read(HERE/f'{phase}_MISSING_CASES.json')[index::count];new=0
 for c in rows:
  out=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
  if out.exists():assert read(out)['case']==c;continue
  r=execute(c,p,cache)
  if phase=='gate':
   s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);rt=make(c['base_controller_name'],p,{});capture=[];oldmake=rt.make_env
   def captured(state):
    env=oldmake(state);capture.append(env.positions.copy());original=env.step
    def observed(action):
     result=original(action);capture.append(env.positions.copy());return result
    env.step=observed;return env
   rt.make_env=captured
   reference=rollout_tt(rt,dict(id=c['rollout_uid'],state=s,eta=c['eta_a'],eta_index=c['eta_index'],seed=c['seed']))
   own=np.load(r['trace_path'])['positions'];first=(r['first_action']==reference['first_action']) if r['first_action'] is None else np.array_equal(np.asarray(r['first_action']),np.asarray(reference['first_action']))
   assert first and own.shape==np.asarray(capture).shape and own.tobytes()==np.asarray(capture).tobytes() and r['success']==reference['success'] and r['outcome']==reference['termination'] and r['episode_length']==reference['steps'],'TT CONSISTENCY GATE FAILED'
   write(HERE/'gate/reference'/f"{c['rollout_uid']}.json",dict(reference=reference,full_trajectory_bitwise_equal=True,new_trace_sha256=r['trace_sha256']))
  write(out,r);cache_writer.append_journal([r],EXP,f'{phase}_{index}');new+=1;print(json.dumps(dict(phase=phase,worker=index,new=new,total=len(rows),controller=c['base_controller_name'],seconds=r['elapsed_seconds'])),flush=True)
 write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),assigned=len(rows),new=new,complete=True))
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

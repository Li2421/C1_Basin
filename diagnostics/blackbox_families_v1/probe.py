from common import *
from registry import register,normalized_state
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity,connect
import copy

def design(stage):
 p=read(HERE/f'{stage}_protocol.json');cap=read(HERE/f'{stage}_CAPABILITY.json');p['controller_order']=[n for n in cap['eligible'] if p['types'][n]!='centralized_sampling'];p['states']=[normalized_state(s) for s in read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')['states']];rows=[];profiles={}
 for name in p['controller_order']:
  v=copy.deepcopy(p['profiles'][name]['payload']);v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=200,terminate_on_success=False,terminate_on_collision=False,probe_code_sha256=sha(__file__));cp=dict(controller_uid=uid('ctl',v),payload=v);profiles[cp['controller_uid']]=cp
  for j,s in enumerate(p['states']):
   for a in [10,13]:
    eid=eta_identity(p['eta'][a])[0];sk=canonical({'future_index':0});rid=uid('probe',dict(state=s['state_uid'],eta=eid,controller=cp['controller_uid'],seed=sk));rows.append(dict(rollout_uid=rid,state_uid=s['state_uid'],controller_uid=cp['controller_uid'],eta_uid=eid,seed_key=sk,base_controller_name=name,state_index=j,eta=p['eta'][a],eta_index=a,seed=0))
 register(('probe_only_' if stage=='pilot_centralized_sampling' else 'probe_')+stage,p,rows,profiles)

def execute(c,p,cache,phase):
 from controllers import make,Observed
 from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
 from dataclasses import replace
 import jax
 jax.config.update('jax_enable_x64',False);rt=Observed(make(c['base_controller_name'],p,cache));s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=rt.env_cls(replace(rt.config,terminate_on_success=False,terminate_on_collision=False));env.reset(**{k:np.asarray(s['physical'][k]) for k in ['positions','velocities','goals']});positions=[env.positions.copy()];actions=[];active=[];error=None
 for step in range(200):
  try:
   a,_,_=rt.action(env,key_for(s,0,step),np.asarray(c['eta']),'old');env.step(a);positions.append(env.positions.copy());actions.append(np.asarray(a).copy());active.append(rt.last.active)
  except Exception as exc:error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);break
 positions=np.asarray(positions);actions=np.asarray(actions);stats=None
 if len(actions)==200:
  initial=np.linalg.norm(env.goals-positions[0],axis=-1).sum();final=np.linalg.norm(env.goals-positions[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(positions[:,:,None]-positions[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(actions,axis=-1);stats=[(initial-final)/initial,np.mean(active),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
 trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,projection_active=active,goals=env.goals)
 return clean(dict(schema='round38_unlabeled_TT_probe_v1',experiment_uid=EXP,case=c,complete=len(actions)==200,statistics=stats,error=error,physical_steps=len(actions),trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled)))
if __name__=='__main__':design(sys.argv[1])

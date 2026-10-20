from common import *
from registry import register
from design import case
from shared_rollout_db.src.rollout_db import uid
import copy

def design():
 from states_probe_supplement import main as supplement
 supplement();assert (HERE/'PROBE_DURATION_AMENDMENT.receipt.json').exists();cap=read(HERE/'CAPABILITY_GATE.json')['eligible'];p=read(HERE/'gate_v2_protocol.json');p['states']=[];states=read(HERE/'STATES.json');rows=[];profiles={}
 for sc,names in cap.items():
  if not names:continue
  p['states']+=states[sc]['probe']
  for name in names:
   key=sc+'::'+name;cp=copy.deepcopy(p['profiles'][key]);cp['payload'].update(unlabeled_probe=True,probe_horizon=round(3*int(cp['payload']['horizon'])/7),termination_flags_off=True);cp['controller_uid']=uid('ctl',cp['payload']);profiles[cp['controller_uid']]=cp
   for s in states[sc]['probe']:
    for eta in [13,15]:
     c=case(s,key,cp,'F'+str(eta),0);c['probe_steps']=round(3*int(cp['payload']['horizon'])/7);c['eta_index']=eta;rows.append(c)
 p['eligible']=cap;register('probe',p,rows,profiles);seal('PROBE_INPUTS_FROZEN.receipt.json',[HERE/'probe_protocol.json',HERE/'probe_CASES.json',HERE/'PROBE_DURATION_AMENDMENT.receipt.json',HERE/'probes.py',HERE/'runtime.py'],unlabeled=True,no_new_evaluation_labels=True)

class Tracker:
 def __init__(self,base):self.base=base;self.active=False
 def __getattr__(self,k):return getattr(self.base,k)
 def project(self,target):
  a,d=self.base.project(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
 def project_field_target(self,target):
  a,d=self.base.project_field_target(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
class Observed:
 def __init__(self,rt):self.rt=rt;self.last=None
 def __getattr__(self,k):return getattr(self.rt,k)
 def safety(self,env):self.last=Tracker(self.rt.safety(env));return self.last
 def prepare(self,*args,**kw):
  from poc.field_controller import Runtime
  return Runtime.prepare(self,*args,**kw)
 def action(self,*args,**kw):
  from poc.field_controller import Runtime
  return Runtime.action(self,*args,**kw)

def stats(positions,actions,active,goals,radius,speed):
 n=len(goals);initial=np.linalg.norm(goals-positions[0],axis=1);progress=float(np.mean(initial-np.linalg.norm(goals-positions[-1],axis=1)>0));vel=np.linalg.norm(actions,axis=-1);streak=np.zeros(n,int);ever=np.zeros(n,bool)
 for row in vel:streak=np.where(row<.01,streak+1,0);ever|=streak>=40
 ii,jj=np.triu_indices(n,1);dist=np.linalg.norm(positions[1:,:,None]-positions[1:,None,:],axis=-1)[:,ii,jj]
 return [progress,float(ever.mean()),float(dist.min()/radius),float(vel.mean()/speed),float(np.mean(active))]

def worker(index):
 from runtime import make,key_for
 from run import cases
 from dataclasses import replace
 import jax
 jax.config.update('jax_enable_x64',False);p=read(HERE/'probe_protocol.json');states={s['state_uid']:s for s in p['states']};cache={};count=0
 for c in cases('probe'):
  out=HERE/'probe/raw'/f"{c['rollout_uid']}.json"
  if out.exists():continue
  rt=Observed(make(c['base_controller_name'],p,cache));flags=dict(terminate_on_success=False,terminate_on_collision=False)
  if hasattr(rt.config,'terminate_on_deadlock'):flags['terminate_on_deadlock']=False
  env=rt.env_cls(replace(rt.config,**flags));s=states[c['state_uid']];physical=s['physical'];pos=np.asarray(physical['positions']);vel=np.asarray(physical['velocities']);goals=np.asarray(physical['goals']);sc=c['scene']
  if sc=='double_bottleneck':env.reset(pos,regime=s['metadata']['regime']);env.velocities=vel.copy()
  elif sc=='ring_exchange':env.reset(pos,velocities=vel,goals=goals)
  elif sc=='toy_give_way':env.reset(pos)
  else:env.reset(pos,velocities=vel)
  positions=[env.positions.copy()];actions=[];active=[];error=None
  for t in range(c['probe_steps']):
   try:
    assert not jax.config.x64_enabled;a,_,_=rt.action(env,key_for(s,0,t),np.asarray(c['eta_a']),'field');env.step(a);positions.append(env.positions.copy());actions.append(np.asarray(a).copy());active.append(rt.last.active)
   except Exception as ex:error=dict(step=t,type=type(ex).__name__,message=str(ex)[:500]);break
  positions=np.asarray(positions);actions=np.asarray(actions);complete=len(actions)==c['probe_steps'];values=stats(positions,actions,active,env.goals,rt.config.agent_radius,rt.config.max_speed) if complete else None;trace=out.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,projection_active=active,goals=env.goals);r=clean(dict(schema='axis3_R20_duration_unlabeled_probe_v1',experiment_uid=EXP,case=c,physical_steps=len(actions),complete=complete,statistics=values,error=error,trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,jax_enable_x64=False));write(out,r);journal=cache_writer.append_journal([r],EXP,f'probe_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP);count+=1;print(json.dumps(dict(phase='probe',index=index,completed=count)),flush=True)
 write(HERE/'probe'/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count))

def freeze():
 p=read(HERE/'probe_protocol.json');cap=read(HERE/'CAPABILITY_GATE.json')['eligible'];keys=[sc+'::'+n for sc in SCENES for n in cap[sc]];features=np.full((len(keys),16,2,5),np.nan);source=[]
 for c in read(HERE/'probe_CASES.json'):
  path=HERE/'probe/raw'/f"{c['rollout_uid']}.json";r=read(path);assert not r['terminal_labels_read'] and not any(k in r for k in ['success','outcome','termination']);assert sha(r['trace_path'])==r['trace_sha256']
  if r['complete']:features[keys.index(c['base_controller_name']),c['state_index'],[13,15].index(c['eta_index'])]=r['statistics']
  source.append(dict(path=str(path),sha256=sha(path),trace_sha256=r['trace_sha256'],complete=r['complete']))
 # Old source Toy remains FP64 and partial; no new Toy rollout after failed gate.
 rows=[c for c in read(R20/'probe_CASES.json') if c['variant']=='toy_give_way_original'];toy=[];radius=read(HERE/'HISTORICAL_INVENTORY.json')['environments']['toy_give_way']['config']['config']['agent_radius']
 for c in rows:
  path=R20/'probe/raw'/f"{c['rollout_uid']}.json";r=read(path);assert not r['contains_success_label'] and not r['contains_termination_label'];assert c['probe_steps']==364
  toy.append([r['progress_fraction'],r['stall_fraction'],r['minimum_pair_distance']/radius,np.nan,np.nan] if not r['error'] else [np.nan]*5);source.append(dict(path=str(path),sha256=sha(path),historical_Toy_FP64=True,partial=True))
 assert len(toy)==32;np.savez_compressed(HERE/'R5_FEATURES.npz',controller_keys=keys,raw=features,features=np.nanmean(features,axis=(1,2)),Toy_source_isotropic=np.nanmean(np.asarray(toy),axis=0));write(HERE/'R5_RAW_INDEX.json',source);seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json',HERE/'PROBE_INPUTS_FROZEN.receipt.json'],before_new_evaluation_labels=True,Toy_partial2missing=True);print('Five-feature R5 frozen',len(keys),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='probe':worker(int(sys.argv[2]))
 else:globals()[sys.argv[1]]()

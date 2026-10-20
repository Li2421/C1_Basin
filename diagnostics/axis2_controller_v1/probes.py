from common import *
from registry import normalized_state,register
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity,connect
import copy

def design():
 cap=read(HERE/'CAPABILITY_GATE.json');p=read(HERE/'NEW_BASE_PROTOCOL.json');p.update(controller_order=cap['eligible'],states=[normalized_state(s) for s in read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')['states']],phase='probe');assert len(p['states'])==16;profiles={};rows=[]
 for name in p['controller_order']:
  v=copy.deepcopy(p['profiles'][name]['payload']);v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=200,terminate_on_success=False,terminate_on_collision=False,probe_code_sha256=sha(__file__));cp=dict(controller_uid=uid('ctl',v),payload=v);profiles[cp['controller_uid']]=cp
  for j,s in enumerate(p['states']):
   for a in [10,13]:
    eid=eta_identity(p['eta'][a])[0];sk=canonical({'future_index':0});rid=uid('probe',dict(state=s['state_uid'],eta=eid,controller=cp['controller_uid'],seed=sk));rows.append(dict(rollout_uid=rid,state_uid=s['state_uid'],controller_uid=cp['controller_uid'],eta_uid=eid,seed_key=sk,base_controller_name=name,state_index=j,eta=p['eta'][a],eta_index=a,seed=0))
 with connect(True) as db:source_count=db.execute("SELECT COUNT(*) FROM source_file WHERE classification IN ('SUMMARY_ONLY','UNLABELED_SHORT_PROBE')").fetchone()[0]
 write(HERE/'PROBE_SOURCE_CACHE_AUDIT.json',dict(utc=utc(),source_files_scanned_count=source_count,new_controller_fingerprints=True,compatible_prior_probe_outputs=0,why='New Gaussian checkpoint hashes and DM/VO/PT implementations/parameters have no historical exact probe identity; existing M/P R5 reused only for source training.',no_task_labels_expected=True))
 register('probe',p,rows,profiles);seal('PROBE_PREFLIGHT_SUPPLEMENT.receipt.json',[HERE/'probe_PREFLIGHT.receipt.json',HERE/'PROBE_SOURCE_CACHE_AUDIT.json'],unlabeled=True,planned=len(rows),task_rollout_rows_expected=0)

def worker(index):
 from controllers import make,Observed
 from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
 from dataclasses import replace
 import jax,time
 jax.config.update('jax_enable_x64',False);assert (HERE/'PROBE_PREFLIGHT_SUPPLEMENT.receipt.json').exists();p=read(HERE/'probe_protocol.json');states={s['state_uid']:s for s in p['states']};cache={};count=0
 from run import cases
 for c in cases('probe'):
  out=HERE/'probe/raw'/f"{c['rollout_uid']}.json"
  if out.exists():continue
  rt=Observed(make(c['base_controller_name'],p,cache));s=states[c['state_uid']];env=rt.env_cls(replace(rt.config,terminate_on_success=False,terminate_on_collision=False));physical=s['physical'];env.reset(np.asarray(physical['positions']),velocities=np.asarray(physical['velocities']),goals=np.asarray(physical['goals']));positions=[env.positions.copy()];actions=[];active=[];error=None
  for step in range(200):
   try:
    a,_,_=rt.action(env,key_for(s,0,step),np.asarray(c['eta']),'old');env.step(a);positions.append(env.positions.copy());actions.append(np.asarray(a).copy());active.append(rt.last.active)
   except Exception as exc:error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);break
  positions=np.asarray(positions);actions=np.asarray(actions);stats=None
  if len(actions)==200:
   initial=np.linalg.norm(env.goals-positions[0],axis=-1).sum();final=np.linalg.norm(env.goals-positions[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(positions[:,:,None]-positions[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(actions,axis=-1);stats=[(initial-final)/initial,np.mean(active),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
  trace=out.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,projection_active=active,goals=env.goals);r=clean(dict(schema='axis2_unlabeled_TT_probe_v1',experiment_uid=EXP,case=c,complete=len(actions)==200,statistics=stats,error=error,physical_steps=len(actions),trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled)));assert not any(k in r for k in ['success','outcome','termination']);write(out,r);cache_writer.append_journal([r],EXP,f'probe_{index}');count+=1;print(json.dumps(dict(phase='probe',index=index,completed=count)),flush=True)
 write(HERE/'probe'/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count))

def freeze():
 p=read(HERE/'probe_protocol.json');names=p['controller_order'];f=np.full((len(names),16,2,5),np.nan,np.float32);index=[]
 for c in read(HERE/'probe_CASES.json'):
  path=HERE/'probe/raw'/f"{c['rollout_uid']}.json";r=read(path);assert not r['terminal_labels_read'] and not r['jax_enable_x64'];assert sha(r['trace_path'])==r['trace_sha256'];assert not any(k in r for k in ['success','outcome','termination'])
  if r['complete']:f[names.index(c['base_controller_name']),c['state_index'],[10,13].index(c['eta_index'])]=r['statistics']
  index.append(dict(probe_uid=c['rollout_uid'],raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256'],complete=r['complete']))
 np.savez_compressed(HERE/'R5_FEATURES.npz',features=f.reshape(len(names),160),controller_names=names);write(HERE/'R5_RAW_INDEX.json',index);seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json',HERE/'probe_protocol.json',HERE/'probes.py'],before_new_evaluation_labels=True,task_labels_read=False);print('Frozen R5',len(names),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(int(sys.argv[2]))
 else:globals()[sys.argv[1]]()

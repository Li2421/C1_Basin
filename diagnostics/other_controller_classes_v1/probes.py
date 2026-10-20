from core import *
import copy,time
from dataclasses import replace
from design import register,REFS,dedup
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE
PROBE_EXP='exp_other_controller_classes_v1_r5'
def create():
 prereg();p=read(HERE/'base_protocol.json');names=read(HERE/'CAPABILITY_GATE.json')['eligible']+REFS;states=read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')['states'];p.update(controller_order=names,states=states,controllers={},phase='probe');rows=[]
 for n in names:
  v=copy.deepcopy(p['base_profiles'][n]['payload']);v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=200,no_early_termination=True,probe_code_sha256=sha(__file__));ctl=uid('ctl',v);p['controllers'][n]=dict(controller_uid=ctl,payload=v)
  for i,s in enumerate(states):
   for a in [10,13]:
    seed=canonical(dict(future_index=0,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(p['eta'][a])[0];rows.append(dict(probe_uid=uid('probe',dict(state=s['state_uid'],eta=e,controller=ctl,seed=seed)),state_uid=s['state_uid'],state_index=i,controller_name=n,controller_uid=ctl,eta_uid=e,eta_index=a,eta=p['eta'][a],seed=0,seed_key=seed))
 write(HERE/'probe_protocol.json',p);write(HERE/'probe_CASES.json',rows);register(p,states,p['controllers'])
 with connect(True) as db:sources=[dict(r) for r in db.execute("SELECT experiment_uid,path,sha256 FROM source_file WHERE classification='UNLABELED_SHORT_PROBE'")]
 assert not any(r['experiment_uid']==PROBE_EXP for r in sources)
 write(HERE/'probe_PREFLIGHT.json',dict(utc=utc(),status='PASS',planned=len(rows),genuinely_missing=len(rows),ambiguous=0,source_file_rows_scanned=len(sources),semantic_fingerprint_novel=True,no_task_labels=True));seal('probe_PREFLIGHT.receipt.json',[HERE/'probe_PREFLIGHT.json',HERE/'probe_protocol.json',HERE/'probe_CASES.json'],before_probes=True)
 from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
 native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');fresh=generate('round27_eval',32,202610152720,native);dedup(fresh,'main');assert not ({s['state_uid'] for s in fresh}&{s['state_uid'] for s in read(HERE/'PILOT_STATES.json')});write(HERE/'EVAL_STATES.json',fresh)
 print('probe preflight',len(rows),flush=True)
def worker(index,workers):
 prereg();from run import make,Observed
 from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
 import jax;jax.config.update('jax_enable_x64',False)
 p=read(HERE/'probe_protocol.json');assert read(HERE/'probe_PREFLIGHT.json')['status']=='PASS';states={s['state_uid']:s for s in p['states']};cache={};count=0
 for c in read(HERE/'probe_CASES.json')[index::workers]:
  out=HERE/'probe/raw'/f"{c['probe_uid']}.json"
  if out.exists():continue
  rt=Observed(make(c['controller_name'],p,cache));s=states[c['state_uid']];env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False);positions=[env.positions.copy()];actions=[];active=[];error=None;t0=time.time()
  for step in range(200):
   try:
    assert not jax.config.x64_enabled;a,_,_=rt.action(env,key_for(s,0,step),np.asarray(c['eta']),'old');env.step(a);positions.append(env.positions.copy());actions.append(a.copy());active.append(rt.last.active)
   except Exception as e:error=dict(step=step,type=type(e).__name__,message=str(e)[:500]);break
  positions=np.asarray(positions);actions=np.asarray(actions);complete=len(actions)==200;stats=None
  if complete:
   initial=np.linalg.norm(env.goals-positions[0],axis=-1).sum();final=np.linalg.norm(env.goals-positions[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(positions[:,:,None]-positions[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(actions,axis=-1);stats=[(initial-final)/initial,np.mean(active),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
  trace=out.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,projection_active=np.asarray(active),goals=env.goals)
  r=dict(schema='round27_unlabeled_TT_probe_v1',experiment_uid=PROBE_EXP,case=c,complete=complete,statistics=stats,error=error,physical_steps=len(actions),trace_path=str(trace),trace_sha256=sha(trace),protocol_sha256=sha(HERE/'probe_protocol.json'),probe_code_sha256=sha(__file__),controller_code_sha256=sha(HERE/'controllers.py'),terminal_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled),elapsed_seconds=time.time()-t0);write(out,r);cache_writer.append_journal([clean(r)],PROBE_EXP,f'probe_{index}');count+=1
  if count%8==0:print(json.dumps(dict(worker=index,new=count,controller=c['controller_name'])),flush=True)
 write(HERE/'probe'/f'worker_{index}.json',dict(complete=True,new=count))
def freeze():
 prereg();p=read(HERE/'probe_protocol.json');names=p['controller_order'];f=np.full((len(names),16,2,5),np.nan,np.float32);index=[];unknown=[]
 for c in read(HERE/'probe_CASES.json'):
  path=HERE/'probe/raw'/f"{c['probe_uid']}.json";r=read(path);assert not r['terminal_labels_read'] and not r['jax_enable_x64'];assert not any(k in r for k in ['success','outcome','timeout','collision']);assert r['protocol_sha256']==sha(HERE/'probe_protocol.json') and sha(r['trace_path'])==r['trace_sha256'];i=names.index(c['controller_name']);si=c['state_index'];ei=[10,13].index(c['eta_index'])
  if r['complete']:
   z=np.load(r['trace_path']);pos=z['positions'];acts=z['actions'];ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)[:,ii,jj];initial=np.linalg.norm(z['goals']-pos[0],axis=-1).sum();final=np.linalg.norm(z['goals']-pos[-1],axis=-1).sum();speed=np.linalg.norm(acts,axis=-1);stats=[(initial-final)/initial,z['projection_active'].mean(),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()];assert np.allclose(stats,r['statistics'],rtol=0,atol=1e-12);f[i,si,ei]=stats
  else:unknown.append(dict(case=c,error=r['error']))
  index.append(dict(probe_uid=c['probe_uid'],raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256']))
 np.savez_compressed(HERE/'R5_FEATURES.npz',features=f.reshape(len(names),160),controller_names=names);write(HERE/'R5_RAW_INDEX.json',index);write(HERE/'PROBE_POSTFLIGHT.json',dict(utc=utc(),status='PASS',rows=len(index),unknown=unknown,task_label_rows=0,all_statistics_reproduced=True))
 with connect() as db,transaction(db):
  db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(PROBE_EXP,'Round27 unlabeled TT R5',str(HERE/'probe'),sha(HERE/'probe_protocol.json'),sha(__file__),canonical(dict(no_terminal_labels=True))))
  for path in sorted((HERE/'journals'/PROBE_EXP).glob('*.jsonl')):
   rows=[json.loads(l) for l in path.read_text().splitlines()];assert all(not r['terminal_labels_read'] for r in rows);suid=uid('source',dict(path=str(path),sha256=sha(path)));db.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen,rows_imported) VALUES(?,?,?,?,?,?,?,?)',(suid,PROBE_EXP,str(path),sha(path),'.jsonl','UNLABELED_SHORT_PROBE',len(rows),0))
 seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json',HERE/'PROBE_POSTFLIGHT.json',HERE/'probe_protocol.json'],before_evaluation_labels=True)
 print('R5 frozen',len(index),'unknown',len(unknown),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
 else:globals()[sys.argv[1]]()

from core import *
from guard import guard
from student import Student
from pipeline import decentralized_action,LocalSafety,local_basis
from registration import register,normalized_state,state_row,canonical,uid,eta_identity
from shared_rollout_db.src.planner import preflight
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from audit_utils import source_inventory
from dataclasses import replace
G=guard()
def design():
 names=read(HERE/'CAPABILITY_GATE.json')['eligible'];states=[normalized_state(s) for s in read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')['states']];eta=read(D/'decentralized_pipeline_v1/main_protocol.json')['eta'];gate=read(HERE/'DISTILLATION_GATE.json');regs=[state_row(s) for s in states];cases=[]
 for name in names:
  rec=next(r for r in gate['controllers'] if r['name']==name);payload=dict(family='round30f_unlabeled_R5',checkpoint_sha256=rec['checkpoint_sha256'],variant=rec['variant'],pipeline_sha256=sha(HERE/'pipeline.py'),probe_sha256=sha(__file__),student_sha256=sha(HERE/'student.py'),steps=200,terminate_on_success=False,terminate_on_collision=False);ctl=uid('ctl',payload);regs.append(('controller_config',dict(controller_uid=ctl,scenario_uid=states[0]['scenario_uid'],flow_checkpoint_sha256=rec['checkpoint_sha256'],config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')))
  for si,s in enumerate(states):
   for j in [10,13]:
    eid,ev,hx=eta_identity(eta[j]);regs.append(('eta',dict(eta_uid=eid,eta1=ev[0],eta2=ev[1],eta3=ev[2],canonical_hex=hx)));sk=canonical(dict(future_index=0));cases.append(dict(probe_uid=uid('probe',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk)),name=name,index=rec['index'],variant=rec['variant'],checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],state_index=si,state_uid=s['state_uid'],eta_index=j,eta=ev,eta_uid=eid,controller_uid=ctl,seed_key=sk))
 register('probes',regs);write(HERE/'probe_protocol.json',dict(states=states,controller_order=names));write(HERE/'probe_CASES.json',cases);manifest=HERE/'probe_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0;write(HERE/'probe_CACHE_PREFLIGHT.json',pf);seal('probe_PREFLIGHT.receipt.json',[manifest,HERE/'probe_CACHE_PREFLIGHT.json',HERE/'probe_CASES.json'],before_submission=True,task_labels=False)
def worker(shard,shards):
 assert (HERE/'probe_PREFLIGHT.receipt.json').exists();p=read(HERE/'probe_protocol.json');cache={}
 for c in read(HERE/'probe_CASES.json')[shard::shards]:
  dest=HERE/'probe/raw'/f"{c['probe_uid']}.json";trace=dest.with_suffix('.npz');assert not dest.exists() and not trace.exists()
  if c['name'] not in cache:cache[c['name']]=Student(c['index'],c['variant'],c['checkpoint'])
  rt=cache[c['name']];s=p['states'][c['state_index']];env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False);positions=[env.positions.copy()];actions=[];active=[];error=None
  for t in range(200):
   try:
    key=key_for(s,0,t);a,_=decentralized_action(rt,env,key,c['eta'],'local',np.zeros(4,int));nom=rt.nominal(env,key)[1];fs=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nom[i:i+1])[0][0] for i,f in enumerate(fs)]);target=safe.copy()
    for i in range(4):target[i]+=np.asarray(c['eta'])@local_basis(env.positions,env.goals,safe,i,1.5,env.config.max_speed)
    active.append(np.linalg.norm(safe-nom)>1e-6 or np.linalg.norm(a-target)>1e-6);env.step(a);positions.append(env.positions.copy());actions.append(a)
   except Exception as e:error=dict(step=t,type=type(e).__name__,message=str(e)[:800]);break
  trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,projection_active=active,goals=env.goals);write(dest,dict(case=c,complete=len(actions)==200,error=error,trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,guard=guard()));print('probe',shard,c['name'],c['state_index'],c['eta_index'],flush=True)
 write(HERE/f'probe_worker{shard}.json',dict(utc=utc(),complete=True,guard=guard()))
 source_inventory(f'imports_probe_{shard}.json')
def freeze():
 p=read(HERE/'probe_protocol.json');names=p['controller_order'];f=np.full((len(names),16,2,5),np.nan,np.float32);index=[];unknown=[]
 for c in read(HERE/'probe_CASES.json'):
  dest=HERE/'probe/raw'/f"{c['probe_uid']}.json";r=read(dest);assert not r['terminal_labels_read'];assert sha(r['trace_path'])==r['trace_sha256']
  if r['complete']:
   z=np.load(r['trace_path']);pos=z['positions'];acts=z['actions'];goals=z['goals'];initial=np.linalg.norm(goals-pos[0],axis=-1).sum();final=np.linalg.norm(goals-pos[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(acts,axis=-1);f[names.index(c['name']),c['state_index'],[10,13].index(c['eta_index'])]=[(initial-final)/initial,z['projection_active'].mean(),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
  else:unknown.append(r)
  index.append(dict(case=c,raw_sha256=sha(dest),trace_sha256=r['trace_sha256']))
 np.savez_compressed(HERE/'R5_FEATURES.npz',features=f.reshape(len(names),160),controller_names=names);write(HERE/'R5_RAW_INDEX.json',index);write(HERE/'probe_CACHE_POSTFLIGHT.json',dict(utc=utc(),complete_records=len(index),unknown=unknown,task_labels=0));seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json'],before_main_labels=True);source_inventory('imports_probe_freeze.json')
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
 else:globals()[sys.argv[1]]()

"""Official design/cache/journal stage for initial demonstrations and relabel rounds."""
from common import *
from dataclasses import asdict
from geometry import GeneratedGeometry,task
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from dagger39 import collect_episode,round_specs,aggregate
from policies39 import load_policy
from design_generated import identities as base_identities
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical
import math,fcntl,jax

POLICIES=[(k,s) for k in ['deterministic','mixture'] for s in [17,23]]
def checkpoint(family,kind,seed,round_index):return HERE/'policies39'/f'{family}_{kind}_{seed}'/f'round_{round_index}'/'best.msgpack'

def policy_assets(path):
 if path is None:return dict(checkpoint_sha256=None,normalization_sha256=None,policy_config_sha256=None)
 path=Path(path)
 return dict(checkpoint_sha256=sha(path),normalization_sha256=sha(path.parent/'NORMALIZATION.json'),policy_config_sha256=sha(path.parent/'CONFIG.json'))

def prepare(family,round_index,kind=None,seed=None):
 from freeze_guard39 import verify
 verify()
 assert read(HERE/'TRAINING39_UNIT_TESTS.json')['passed'];assert (HERE/'TRAINING_PIPELINE39.receipt.json').exists()
 assert read(HERE/'EXPERT_GATE_RESULTS.json')['families'][family]['passed']
 assert round_index in range(4)
 catalog=read(HERE/'PILOT_GEOMETRY_FROZEN.json')[family];train=[s for s in catalog if s['role'] in ['train','train_new']];assert len(train)==64
 spec=round_specs(family);policy_index=POLICIES.index((kind,seed)) if round_index else None
 previous=checkpoint(family,kind,seed,round_index-1) if round_index else None
 if previous:assert (previous.parent/'COMPLETE.receipt.json').exists()
 stage=f'data39_{family}_round_{round_index}'+(f'_{kind}_{seed}' if round_index else '')
 cases=[];registered=[];rejections=[]
 for m,source in enumerate(train):
  geo=GeneratedGeometry(Config(**source['config']),source['rectangles'])
  for role,count in ([('round0_train',32),('round0_dev',8)] if round_index==0 else [('relabel_train',16)]):
   for j in range(count):
    if round_index==0:base=spec['initial_task_base']+m*spec['initial_map_stride']+(spec['initial_dev_offset'] if role=='round0_dev' else 0)+j*100
    else:base=spec['relabel_task_base']+round_index*spec['relabel_round_stride']+policy_index*spec['relabel_policy_stride']+m*spec['relabel_map_stride']+j*100
    for retry in range(100):
     try:p,g,paths,_=task(geo,family,base+retry);break
     except ValueError as e:rejections.append(dict(seed=base+retry,reason=str(e)))
    else:raise RuntimeError('task seed block exhausted')
    robot=j%4;H=int(math.ceil(2.5*np.linalg.norm(np.diff(paths[robot],axis=0),axis=1).sum()/.5/.05/100)*100)
    state=dict(source,role=role,round=round_index,map_index=m,index=j,task_seed=base+retry,alias=f'round40_{stage}_{m}_{role}_{j}',positions=p[robot:robot+1].tolist(),goals=g[robot:robot+1].tolist(),velocities=np.zeros((1,2)).tolist(),paths=[paths[robot].tolist()],H=H,config=dict(source['config'],num_agents=1,max_steps=H))
    c,rows=base_identities(state,'navigation_field','navigation',checkpoint=previous)
    row=next(v for t,v in rows if t=='controller_config');payload=json.loads(row['config_json']);payload.update(kind='learner39_relabel' if previous else 'expert39_demonstration',round=round_index,observation_dimensions=39,policy_kind=kind,policy_seed=seed,sources39={n:sha(HERE/n) for n in ['policies.py','policies39.py','dagger39.py','collect39.py','observation39.py','navigation_field.py','batch_fields39.py','batch_collect39.py']},rng_root=2026101740)
    payload.update(policy_assets(previous))
    cid=uid('ctl',payload);row.update(controller_uid=cid,config_json=canonical(payload),conditioning_version=payload['kind'],rng_semantics_version='root2026101740-fold_stateprefix32-fold_physical_step')
    c.update(controller_uid=cid,kind=payload['kind'],round=round_index,policy_kind=kind,policy_seed=seed,collection_stage=stage)
    c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key']))
    cases.append(c);registered+=rows
 write(HERE/f'{stage}_REJECTIONS.json',rejections);missing=register(stage,cases,registered)
 groups={}
 for i,c in enumerate(missing):groups.setdefault((c['state']['geometry_seed'],c['state']['role']),[]).append(i)
 work=[indices[j:j+32] for indices in groups.values() for j in range(0,len(indices),32)]
 write(HERE/f'{stage}_MISSING_GROUPS.json',work)
 seal(f'{stage}_BATCH_PLAN.receipt.json',[HERE/f'{stage}_MISSING_GROUPS.json',HERE/f'{stage}_MISSING_CASES.json'],independent_single_agent_environments=True,maximum_group_size=32)
 return stage

def worker(stage,index):
 from freeze_guard39 import verify
 from batch_collect39 import collect_batch
 verify()
 cases=read(HERE/f'{stage}_MISSING_CASES.json');groups=read(HERE/f'{stage}_MISSING_GROUPS.json');folder=HERE/stage;folder.mkdir(exist_ok=True)
 while True:
  with (folder/'cursor').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);group=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(group+1));f.flush()
  if group>=len(groups):break
  selected=[cases[i] for i in groups[group] if not (folder/'raw'/f"{cases[i]['rollout_uid']}.json").exists()]
  if not selected:continue
  cfg=dict(selected[0]['config']);cfg['num_agents']=4;geo=GeneratedGeometry(Config(**cfg),selected[0]['state']['rectangles']);envs=[];policies=[];keys=[]
  for c in selected:
   s=c['state'];cfg=dict(c['config']);cfg['num_agents']=4;envs.append(SingleAgentGeneratedEnv(geo,s['positions'],s['goals'],Config(**cfg)))
   policies.append(load_policy(c['checkpoint']) if c['checkpoint'] else None)
   keys.append(jax.random.fold_in(jax.random.PRNGKey(2026101740),int(hashlib.sha256(c['state_uid'].encode()).hexdigest()[:8],16)))
  try:records=collect_batch(envs,policies,keys)
  except Exception as e:
   records=[dict(observations=np.zeros((0,39),np.float32),labels=np.zeros((0,2),np.float32),actions=np.zeros((0,1,2)),positions=np.asarray(c['state']['positions'])[None],wall_clearance=np.zeros(0),outcome='numerical_failure',error=dict(type=type(e).__name__,message=str(e))) for c in selected]
  for c,record in zip(selected,records):
   s=c['state'];error=record['error'];trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(exist_ok=True)
   np.savez_compressed(trace,**{k:v for k,v in record.items() if k not in ['outcome','error']})
   term=record['outcome'];actions=record['actions']
   result=dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='Generated '+s['family'],**{k:c[k] for k in ['scenario_uid','state_uid','controller_uid','eta_uid','rollout_uid','seed_key']},eta=c['eta'],future_index=c['seed'],future_root=2026101740,initial_positions=s['positions'],initial_velocities=s['velocities'],goals=s['goals'],flow_checkpoint_sha256=c['flow_checkpoint_sha256'],success=int(term=='success' and error is None),collision=int(term=='collision'),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),outcome=term,error=error,episode_length=len(actions),first_action=actions[0] if len(actions) else None,trace_path=str(trace),trace_sha256=sha(trace),case=c)
   write(folder/'raw'/f"{c['rollout_uid']}.json",result);journal=cache_writer.append_journal([clean(result)],EXP,f'{stage}_{index}');assert journal.is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
  print(group,len(records),{term:sum(r['outcome']==term for r in records) for term in set(r['outcome'] for r in records)},flush=True)

def freeze_dataset(family,kind,seed,round_index):
 training={s['geometry_seed'] for s in read(HERE/'PILOT_GEOMETRY_FROZEN.json')[family] if s['role'] in ['train','train_new']};records=[]
 stages=[f'data39_{family}_round_0']+[f'data39_{family}_round_{r}_{kind}_{seed}' for r in range(1,round_index+1)]
 for stage in stages:
  for c in read(HERE/f'{stage}_CASES.json'):
   raw=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");s=c['state']
   # Unknown remains in provenance, contributes zero verified labels when collection failed.
   records.append(dict(role=s['role'],round=s['round'],geometry_seed=s['geometry_seed'],trace_path=raw['trace_path'],trace_sha256=raw['trace_sha256'],rollout_uid=raw['rollout_uid'],outcome=raw['outcome']))
 arrays,provenance=aggregate([r for r in records if r['role']=='round0_train'],[r for r in records if r['role']=='round0_dev'],[r for r in records if r['role']=='relabel_train'],round_index,training)
 p=HERE/'datasets39'/f'{family}_{kind}_{seed}_round_{round_index}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**arrays)
 write(p.with_suffix('.json'),dict(family=family,kind=kind,seed=seed,round=round_index,dataset_sha256=sha(p),heldout_geometry_used=False,provenance=provenance,transitions={k:len(v) for k,v in arrays.items()}))
 seal(str(p.relative_to(HERE))+'.receipt.json',[p,p.with_suffix('.json')],before_training=True)
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))

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

from full_geometry40 import require_pilot,verify_full
POLICIES=[(k,s) for k in ['deterministic','mixture'] for s in [17,23]]
def checkpoint(family,kind,seed,round_index):return HERE/'policies39'/f'{family}_{kind}_{seed}'/f'round_{round_index}'/'best.msgpack'

def policy_assets(path):
 if path is None:return dict(checkpoint_sha256=None,normalization_sha256=None,policy_config_sha256=None)
 path=Path(path)
 return dict(checkpoint_sha256=sha(path),normalization_sha256=sha(path.parent/'NORMALIZATION.json'),policy_config_sha256=sha(path.parent/'CONFIG.json'))

def _prepare(family,round_index,kind=None,seed=None):
 from freeze_guard39 import verify
 verify()
 assert read(HERE/'TRAINING39_UNIT_TESTS.json')['passed'];assert (HERE/'TRAINING_PIPELINE39.receipt.json').exists()
 assert read(HERE/f'FULL_{family}_EXPERT_GATE_RESULTS.json')['families'][family]['passed']
 assert round_index in range(4)
 catalog=read(HERE/f'FULL_{family}_GEOMETRY_FROZEN.json')[family];train=[s for s in catalog if s['role'] in ['train','train_new']];assert len(train)==64
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

def _freeze_dataset(family,kind,seed,round_index):
 training={s['geometry_seed'] for s in read(HERE/f'FULL_{family}_GEOMETRY_FROZEN.json')[family] if s['role'] in ['train','train_new']};records=[]
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


def prepare(family,round_index,kind=None,seed=None):
 require_pilot();verify_full();assert family in 'ACDEG' and len(family)==1
 return _prepare(family,round_index,kind,seed)

def freeze_dataset(family,kind,seed,round_index):
 require_pilot();verify_full();assert family in 'ACDEG' and len(family)==1
 return _freeze_dataset(family,kind,seed,round_index)

"""Expert relabeling at every pre-step visited state; training geometry only."""
from common import *
from navigation_field import NavigationGrid,expert_action
from observation39 import observe
import jax

def collect_episode(env,policy=None,key=None):
 assert env.num_agents==1
 fields=[NavigationGrid(env.instance.geometry).field(g) for g in env.goals]
 positions=[env.positions.copy()];observed=[];labels=[];executed=[];clearances=[]
 while not env.done:
  # Only states before the native terminal step are input/label pairs.
  x=observe(env,fields);teacher=expert_action(env,fields)
  u=teacher if policy is None else policy(env,fields,jax.random.fold_in(key,env.step_count))
  assert x.shape==(1,39) and teacher.shape==(1,2) and np.isfinite(x).all() and np.isfinite(teacher).all()
  _,_,info=env.step(u,diagnose_stalls=False)
  observed.append(x[0]);labels.append(teacher[0]);executed.append(u);positions.append(env.positions.copy());clearances.append(info['min_swept_wall_clearance'])
 return dict(observations=np.asarray(observed,dtype=np.float32).reshape(-1,39),labels=np.asarray(labels,dtype=np.float32).reshape(-1,2),actions=np.asarray(executed).reshape(-1,1,2),positions=np.asarray(positions),wall_clearance=np.asarray(clearances),outcome=env.termination)

def validate_sources(records,training_seeds):
 for record in records:
  assert record['geometry_seed'] in training_seeds,'held-out geometry forbidden'
  assert record['role'] in ['round0_train','round0_dev','relabel_train']
  assert record['round'] in range(4)
  if record['role']=='round0_dev':assert record['round']==0

def aggregate(initial_train,initial_dev,relabels,round_index,training_seeds):
 assert round_index in range(4)
 records=list(initial_train)+list(initial_dev)+list(relabels);validate_sources(records,training_seeds)
 assert all(r['role']=='round0_train' and r['round']==0 for r in initial_train)
 assert all(r['role']=='round0_dev' and r['round']==0 for r in initial_dev)
 assert all(r['role']=='relabel_train' and 1<=r['round']<=round_index for r in relabels)
 arrays={k:[] for k in ['train_X','train_Y','dev_X','dev_Y']};provenance=[]
 for record in records:
  path=Path(record['trace_path']);assert sha(path)==record['trace_sha256'];z=np.load(path)
  x=z['observations'];y=z['labels'];assert x.shape==(len(y),39) and y.shape[1:]==(2,)
  assert len(x)==len(z['actions']) and len(z['positions'])==len(x)+1
  assert np.isfinite(x).all() and np.isfinite(y).all()
  split='dev' if record['role']=='round0_dev' else 'train';arrays[split+'_X'].append(x);arrays[split+'_Y'].append(y);provenance.append(record)
 assert all(arrays.values())
 return {k:np.concatenate(v).astype(np.float32) for k,v in arrays.items()},provenance

def round_specs(family):
 ordinal='ABCDEFG'.index(family)
 return dict(initial_train_count=32,initial_dev_count=8,relabel_count_per_policy_round=16,initial_task_base=202680000000+ordinal*1000000000,initial_map_stride=1000000,initial_dev_offset=500000,task_stride=100,relabel_task_base=202690000000+ordinal*1000000000,relabel_round_stride=100000000,relabel_policy_stride=10000000,relabel_map_stride=100000,relabel_rounds=[1,2,3],policy_kinds=['deterministic','mixture'],policy_seeds=[17,23])

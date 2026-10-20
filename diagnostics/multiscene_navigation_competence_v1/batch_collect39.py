"""Independent single-agent environments, shared batched static-field arithmetic."""
import numpy as np
import jax
from navigation_field import NavigationGrid
from batch_fields39 import IndependentFields

def collect_batch(envs,policies,keys):
 assert len(envs)==len(policies)==len(keys) and all(e.num_agents==1 for e in envs)
 grid=NavigationGrid(envs[0].instance.geometry);fields=[grid.field(e.goals[0]) for e in envs];batch=IndependentFields(fields)
 buffers=[dict(observations=[],labels=[],actions=[],positions=[e.positions.copy()],wall_clearance=[]) for e in envs];errors=[None for _ in envs];active=list(range(len(envs)))
 while active:
  current=[envs[i] for i in active]
  try:observed=batch.observed(current,active);teacher=batch.expert(current,active)
  except Exception as e:
   for i in active:errors[i]=dict(type=type(e).__name__,message=str(e))
   break
  continuing=[]
  for row,i in enumerate(active):
   env=envs[i];record=buffers[i]
   try:
    u=teacher[row:row+1] if policies[i] is None else policies[i].from_observation(env,observed[row:row+1],jax.random.fold_in(keys[i],env.step_count))
    _,done,info=env.step(u,diagnose_stalls=False)
    record['observations'].append(observed[row]);record['labels'].append(teacher[row]);record['actions'].append(u);record['positions'].append(env.positions.copy());record['wall_clearance'].append(info['min_swept_wall_clearance'])
    if not done:continuing.append(i)
   except Exception as e:errors[i]=dict(type=type(e).__name__,message=str(e))
  active=continuing
 results=[]
 for env,b,error in zip(envs,buffers,errors):
  results.append(dict(observations=np.asarray(b['observations'],np.float32).reshape(-1,39),labels=np.asarray(b['labels'],np.float32).reshape(-1,2),actions=np.asarray(b['actions']).reshape(-1,1,2),positions=np.asarray(b['positions']),wall_clearance=np.asarray(b['wall_clearance']),outcome=env.termination if error is None else 'numerical_failure',error=error))
 return results

from common import *
from geometry import make_geometry,task
from single_agent import SingleAgentGeneratedEnv
from navigation_field import NavigationGrid,expert_action
from observation39 import observe
from batch_fields39 import IndependentFields

def main():
 checks=[]
 for family in ['B','F','D']:
  geo,_=make_geometry(family,202640900000+ord(family));grid=NavigationGrid(geo);envs=[];fields=[]
  for j in range(16):
   for attempt in range(100):
    try:p,g,_,_=task(geo,family,202710000000+ord(family)*10000+j*100+attempt);break
    except ValueError:continue
   else:raise RuntimeError('unit fixture draw exhausted')
   i=j%4;env=SingleAgentGeneratedEnv(geo,p[i:i+1],g[i:i+1]);envs.append(env);fields.append(grid.field(g[i]))
  batch=IndependentFields(fields);rows=np.arange(len(envs));positions=np.concatenate([e.positions for e in envs])
  for s in [.3,.5,1.5]:
   a=batch.waypoint(positions,s);b=np.concatenate([f.waypoint(p[None],s) for f,p in zip(fields,positions)]);assert a.tobytes()==b.tobytes(),(family,s,np.max(abs(a-b)))
  a=batch.observed(envs,rows);b=np.concatenate([observe(e,[f]) for e,f in zip(envs,fields)]);assert a.tobytes()==b.tobytes(),family
  a=batch.expert(envs,rows);b=np.concatenate([expert_action(e,[f]) for e,f in zip(envs,fields)]);assert a.tobytes()==b.tobytes(),family
  # Subset mapping and target points also agree, including inactive rows.
  subset=np.array([2,5,9]);a=batch.waypoint(positions[subset],.5,subset);b=np.concatenate([fields[i].waypoint(positions[i:i+1],.5) for i in subset]);assert a.tobytes()==b.tobytes()
  assert batch.waypoint(batch.goals,1.5).tobytes()==batch.goals.tobytes()
  checks.append(dict(family=family,independent_episodes=16,waypoints_bitwise=True,observations_bitwise=True,expert_actions_bitwise=True,subset_bitwise=True))
 write(HERE/'BATCH_FIELDS39_TEST.json',dict(passed=True,checks=checks,static_only=True));print(checks)
if __name__=='__main__':main()

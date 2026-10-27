"""Five evidence levels; geometry diagnostics read stored simulation/task data only."""
from common import *
from design import *
import numpy as np
from collections import deque

def literature_paths(scene):
 from legacy_states import LiteratureConfig
 from swap_geometry import Geometry
 from swap_geometry_user_fix_20261023 import GeometryRevision
 from flow_scene import LiteratureGeometry
 cfg=LiteratureConfig(**scene['config']);n=scene['name'];g=(GeometryRevision(cfg)if n=='lit_swap8_obstacles'else Geometry(cfg)if n=='lit_swap8_free'else LiteratureGeometry(cfg))
 assert np.array_equal(np.asarray(scene['walls']).reshape(-1,2,2),g.walls.reshape(-1,2,2))
 states=[];all_points=np.vstack([np.asarray(s[k])for s in scene['states']for k in ['positions','goals']]);lo=all_points.min(0)-1;hi=all_points.max(0)+1;step=.025
 x=np.arange(lo[0],hi[0]+step/2,step);y=np.arange(lo[1],hi[1]+step/2,step);grid=None;H,W=len(y),len(x)
 def neighbors(point):
  i=int(np.argmin(abs(y-point[1])));j=int(np.argmin(abs(x-point[0])));out=[]
  for a in range(max(0,i-2),min(H,i+3)):
   for b in range(max(0,j-2),min(W,j+3)):
    if grid[a,b]and g.visible(point,np.array([x[b],y[a]])):out.append((a,b))
  return out
 def path(a,b):
  nonlocal grid
  if g.visible(a,b):return np.stack([a,b])
  if grid is None:
   xx,yy=np.meshgrid(x,y);grid=g.valid_points(np.stack([xx,yy],-1))
  starts=neighbors(a);ends=set(neighbors(b));parents={p:None for p in starts};q=deque(starts);found=None
  while q:
   v=q.popleft()
   if v in ends:found=v;break
   for w in [(v[0]-1,v[1]),(v[0]+1,v[1]),(v[0],v[1]-1),(v[0],v[1]+1)]:
    if w not in parents and 0<=w[0]<H and 0<=w[1]<W and grid[w]:parents[w]=v;q.append(w)
  if found is None:return None
  cells=[];v=found
  while v is not None:cells.append(np.array([x[v[1]],y[v[0]]]));v=parents[v]
  p=np.vstack([a,list(reversed(cells)),b]);assert all(g.visible(a,b)for a,b in zip(p[:-1],p[1:]));return p
 for s in scene['states']:
  p=np.asarray(s['positions']);goals=np.asarray(s['goals']);i,j=np.triu_indices(len(p),1)
  valid=bool(g.valid_points(p).all()and g.valid_points(goals).all()and np.all(np.linalg.norm(p[i]-p[j],axis=-1)>2*cfg.agent_radius)and np.all(np.linalg.norm(goals[i]-goals[j],axis=-1)>=2*cfg.agent_radius-1e-12))
  paths=[path(a,b)for a,b in zip(p,goals)];states.append(dict(index=s['index'],geometric_valid=valid,single_robot_paths=all(p is not None for p in paths),path_witnesses=[None if p is None else p.tolist()for p in paths],grid_step=step,joint_solvable='未认证',deadlock_resolution='未认证',eligible=valid and all(p is not None for p in paths)))
 return dict(scene=n,config=scene['config'],states=states,source='Read-only exact frozen geometry; direct visibility or four-neighbor .025 grid witnesses; every witness segment rechecked.')

def main():
 scenes=read(HERE/'SCENES.json');tables=read(HERE/'LABEL_TABLES.json');reports=read(HERE/'GEOMETRY_BASE_REPORTS.json');cap=read(HERE/'CAPABILITY_TABLE.json')
 for scene in scenes:
  n=scene['name']
  if n.startswith('lit_'):reports[n]=literature_paths(scene)
  r=reports[n];r['joint_solvable']='未认证';r['deadlock_resolution']='未认证';r['capability_gate']=cap['scene_status'][n]
  for s in r['states']:
   if n not in tables:s['controller_success']='能力门不适用，无主标签';continue
   t=tables[n];i=t['state_indices'].index(s['index']);y=np.asarray(t['y'])[:,i];u=np.asarray(t['unknown'])[:,i]
   s['controller_success']=dict(successful_controller_candidate_cells=int(y.sum()),total_cells=int(y.size),unknown_cells=int(u.sum()),per_controller={ctl:int(y[j].sum())for j,ctl in enumerate(CONTROL_IDS)})
   s['joint_solvable']='未认证';s['deadlock_resolution']='未认证'
 write(HERE/'GEOMETRY_REPORTS.json',reports)
if __name__=='__main__':main()

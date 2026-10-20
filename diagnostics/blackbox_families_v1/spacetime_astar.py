"""Centralized priority planning with continuous swept disc reservations."""
from common import *
from legacy_controllers import Nominal,bound
import heapq,math

def segment_distance(a,b):
 d=b-a;t=np.clip(-np.dot(a,d)/max(np.dot(d,d),1e-30),0,1);return float(np.linalg.norm(a+t*d))
class SpaceTimeGrid:
 def __init__(self,resolution,time_step,inner=0.,outer=float('inf'),separation=0.,max_time=700):
  self.res=resolution;self.dt=time_step;self.inner=inner;self.outer=outer;self.sep=separation;self.max_time=max_time;self.reserved=[]
 def point(self,g):return np.asarray(g,dtype=float)*self.res
 def valid_edge(self,a,b,t):
  if segment_distance(a,b)<self.inner-1e-12 or max(np.linalg.norm(a),np.linalg.norm(b))>self.outer+1e-12:return False
  for path in self.reserved:
   c=path[min(t,len(path)-1)];d=path[min(t+1,len(path)-1)]
   if segment_distance(a-c,b-d)<self.sep-1e-12:return False
  return True
 def search(self,start,goal):
  start=np.asarray(start,float);goal=np.asarray(goal,float);sg=tuple(np.rint(start/self.res).astype(int));gg=tuple(np.rint(goal/self.res).astype(int))
  def heuristic(g):return float(np.linalg.norm(np.asarray(g)-gg)/np.sqrt(2))
  initial=(sg[0],sg[1],0);queue=[(heuristic(sg),0,initial)];distance={initial:0.};parent={};serial=0
  moves=[(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
  while queue:
   _,_,u=heapq.heappop(queue);g=u[:2];t=u[2];a=start if t==0 else self.point(g)
   if g==gg and self.valid_edge(a,goal,t):
    # The goal is reserved for all remaining time; reject future occupation.
    if all(self.valid_edge(goal,goal,j) for j in range(t+1,self.max_time)):
     chain=[u]
     while chain[-1] in parent:chain.append(parent[chain[-1]])
     chain=chain[::-1];path=np.asarray([start]+[self.point(v[:2]) for v in chain[1:]]+[goal]);self.reserved.append(path);return path
   if t>=self.max_time-1:continue
   for dx,dy in moves:
    ng=(g[0]+dx,g[1]+dy);b=self.point(ng)
    # Diagonal moves take one quantum, so quantum accounts for sqrt(2).
    if not self.valid_edge(a,b,t):continue
    v=(*ng,t+1);cost=distance[u]+1
    if cost>=distance.get(v,float('inf')):continue
    distance[v]=cost;parent[v]=u;serial+=1;heapq.heappush(queue,(cost+heuristic(ng),serial,v))
  return None

class PrioritizedAStar(Nominal):
 def __init__(self,parameters,manifest):
  super().__init__('ASTAR',parameters,manifest);self.last_replan=None;self.paths={};self.replan_steps=[]
 def plan(self,env):
  res,K,priority=self.parameters;cfg=env.config;quantum=max(cfg.dt,res/.52)
  grid=SpaceTimeGrid(res,quantum,cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin,cfg.outer_radius-cfg.agent_radius-cfg.collision_margin,2*cfg.agent_radius+cfg.agent_collision_margin,700)
  order=list(range(len(env.positions)))
  if priority=='far':order=sorted(order,key=lambda i:(-np.linalg.norm(env.goals[i]-env.positions[i]),i))
  self.paths={}
  for i in order:
   path=grid.search(env.positions[i],env.goals[i]);self.paths[i]=path
   if path is None:grid.reserved.append(np.asarray([env.positions[i]]))
  self.last_replan=env.step_count;self.replan_steps.append(env.step_count);self.grid=grid
 def nominal(self,env,key,traced=False):
  assert not traced
  if env.step_count==0 and self.last_replan!=0:self.last_replan=None;self.replan_steps=[]
  if self.last_replan is None or env.step_count-self.last_replan>=self.parameters[1]:self.plan(env)
  result=[];elapsed=(env.step_count-self.last_replan)*env.config.dt
  for i,pos in enumerate(env.positions):
   path=self.paths[i]
   if path is None:result.append(np.zeros(2));continue
   # Respect planned wait slots; then follow by arc length with 0.3m lookahead.
   slot=min(int(elapsed/self.grid.dt),len(path)-1)
   if slot<len(path)-1 and np.linalg.norm(path[slot+1]-path[slot])<1e-12:result.append(np.zeros(2));continue
   nearest=slot+int(np.argmin(np.linalg.norm(path[slot:]-pos,axis=-1)));target=path[-1];length=0.;previous=pos
   for point in path[nearest:]:
    segment=np.linalg.norm(point-previous)
    if length+segment>=.3:
     target=previous+(.3-length)/max(segment,1e-30)*(point-previous);break
    length+=segment;previous=point
   delta=target-pos;speed=.52*min(1.,np.linalg.norm(env.goals[i]-pos)/.3);result.append(delta/max(np.linalg.norm(delta),1e-30)*speed)
  return None,bound(np.asarray(result),env.config.max_speed),None

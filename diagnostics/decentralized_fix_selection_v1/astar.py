"""Deterministic private A*: analytic inflated-disc edge checks, eight neighbors."""
from common import *
from legacy_controllers import Nominal,bound
import heapq,math
class GridPlanner:
 def __init__(self,cfg,resolution):
  self.cfg=cfg;self.res=float(resolution);self.extent=math.ceil(cfg.outer_radius/self.res);self.axis=np.arange(-self.extent,self.extent+1)*self.res
  self.inner=cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin;self.outer=cfg.outer_radius-cfg.agent_radius-cfg.collision_margin
  self.xy=np.stack(np.meshgrid(self.axis,self.axis,indexing='ij'),-1);r=np.linalg.norm(self.xy,axis=-1);self.static=(r>self.inner)&(r<self.outer)
  self.moves=[(a,b) for a in [-1,0,1] for b in [-1,0,1] if a or b]
 def clear(self,a,b,obstacles):
  delta=b-a;alpha=np.clip(-a@delta/max(delta@delta,1e-30),0,1)
  if np.linalg.norm(a+alpha*delta)<=self.inner or max(np.linalg.norm(a),np.linalg.norm(b))>=self.outer:return False
  for p in obstacles:
   q=a-p;t=np.clip(-q@delta/max(delta@delta,1e-30),0,1)
   if np.linalg.norm(q+t*delta)<=2*self.cfg.agent_radius:return False
  return True
 def plan(self,start,goal,obstacles=()):
  start=np.asarray(start,float);goal=np.asarray(goal,float);obstacles=np.asarray(obstacles,float).reshape(-1,2)
  valid=self.static.copy()
  for p in obstacles:valid&=np.linalg.norm(self.xy-p,axis=-1)>2*self.cfg.agent_radius
  nodes=np.argwhere(valid)
  if not len(nodes):return np.empty((0,2))
  def anchor(point):
   dist=np.square(self.xy[valid]-point).sum(-1)
   for q in np.argsort(dist,kind='stable'):
    ij=tuple(nodes[q]);v=self.xy[ij]
    if self.clear(point,v,obstacles):return ij
   return None
  s=anchor(start);g=anchor(goal)
  if s is None or g is None:return np.empty((0,2))
  if self.clear(start,goal,obstacles):return np.stack([start,goal])
  def h(u):return self.res*math.hypot(u[0]-g[0],u[1]-g[1])
  costs={s:0.};parent={};heap=[(h(s),0.,s)];closed=set();N=len(self.axis)
  while heap:
   _,cost,u=heapq.heappop(heap)
   if u in closed:continue
   if u==g:
    path=[g]
    while path[-1]!=s:path.append(parent[path[-1]])
    path=[start]+[self.xy[q] for q in path[::-1]]+[goal]
    return np.asarray(path)
   closed.add(u)
   for di,dj in self.moves:
    v=(u[0]+di,u[1]+dj)
    if not (0<=v[0]<N and 0<=v[1]<N and valid[v]) or v in closed:continue
    if di and dj and not (valid[u[0]+di,u[1]] and valid[u[0],u[1]+dj]):continue
    if not self.clear(self.xy[u],self.xy[v],obstacles):continue
    new=cost+self.res*math.hypot(di,dj)
    if new<costs.get(v,float('inf')):costs[v]=new;parent[v]=u;heapq.heappush(heap,(new+h(v),new,v))
  return np.empty((0,2))
def track(position,goal,path,speed):
 if len(path)==0:return np.zeros(2)
 # Project onto nearest path segment, then walk 0.3 m along the polyline.
 delta=np.diff(path,axis=0);den=np.sum(delta*delta,axis=1);t=np.clip(np.sum((position-path[:-1])*delta,axis=1)/np.maximum(den,1e-30),0,1);proj=path[:-1]+t[:,None]*delta;i=int(np.square(proj-position).sum(1).argmin());q=proj[i];remaining=.3
 for end in path[i+1:]:
  length=np.linalg.norm(end-q)
  if length>=remaining:q=q+(end-q)*remaining/max(length,1e-30);break
  remaining-=length;q=end
 direction=q-position;norm=np.linalg.norm(direction);v=speed*min(1.,np.linalg.norm(goal-position)/.3)
 return direction/max(norm,1e-30)*v
class AStarNominal(Nominal):
 def __init__(self,parameters,manifest):
  super().__init__('AS',parameters,manifest);self.planner=GridPlanner(self.config,parameters[0]);self.paths={};self.last=-1;self.cached=None
 def make_env(self,s):
  self.paths={};self.last=-1;self.cached=None;return super().make_env(s)
 def private_action(self,position,goal,neighbors,step,i):
  if i not in self.paths or step%self.parameters[1]==0:self.paths[i]=self.planner.plan(position,goal,neighbors if self.parameters[2] else ())
  return track(position,goal,self.paths[i],self.config.max_speed)
 def nominal(self,env,key,traced=False):
  assert not traced
  if env.step_count==self.last:return None,self.cached.copy(),None
  answer=[]
  for i in range(4):
   positions=np.asarray([env.positions[j] for j in range(4) if j!=i and np.linalg.norm(env.positions[j]-env.positions[i])<=1.5]).reshape(-1,2)
   answer.append(self.private_action(env.positions[i],env.goals[i],positions,env.step_count,i))
  self.cached=np.asarray(answer);self.last=env.step_count;return None,self.cached.copy(),None

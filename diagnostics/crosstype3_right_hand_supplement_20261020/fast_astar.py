"""Same priority A* algorithm/ties with a compiled scalar search kernel."""
from common import *
from spacetime_astar import SpaceTimeGrid,PrioritizedAStar
import ctypes
lib=ctypes.CDLL(str(HERE/'astar_kernel.so'));ptr=ctypes.POINTER(ctypes.c_double);iptr=ctypes.POINTER(ctypes.c_int)
lib.astar.argtypes=[ctypes.c_double]*4+[ctypes.c_int,ptr,ptr,ctypes.c_int,iptr,ptr,ptr];lib.astar.restype=ctypes.c_int
class FastGrid(SpaceTimeGrid):
 def search(self,start,goal):
  start=np.ascontiguousarray(start,dtype=np.float64);goal=np.ascontiguousarray(goal,dtype=np.float64);lens=np.asarray([len(p) for p in self.reserved],np.int32);reserved=np.ascontiguousarray(np.concatenate(self.reserved) if self.reserved else np.zeros((0,2)),dtype=np.float64);output=np.zeros((self.max_time+2,2),np.float64)
  length=lib.astar(self.res,self.inner,self.outer,self.sep,self.max_time,start.ctypes.data_as(ptr),goal.ctypes.data_as(ptr),len(lens),lens.ctypes.data_as(iptr),reserved.ctypes.data_as(ptr),output.ctypes.data_as(ptr))
  if not length:return None
  path=output[:length].copy();self.reserved.append(path);return path
class FastPrioritizedAStar(PrioritizedAStar):
 def plan(self,env):
  res,K,priority=self.parameters;cfg=env.config;quantum=max(cfg.dt,res/.52)
  grid=FastGrid(res,quantum,cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin,cfg.outer_radius-cfg.agent_radius-cfg.collision_margin,2*cfg.agent_radius+cfg.agent_collision_margin,700)
  order=list(range(len(env.positions)))
  if priority=='far':order=sorted(order,key=lambda i:(-np.linalg.norm(env.goals[i]-env.positions[i]),i))
  self.paths={}
  for i in order:
   path=grid.search(env.positions[i],env.goals[i]);self.paths[i]=path
   if path is None:grid.reserved.append(np.asarray([env.positions[i]]))
  self.last_replan=env.step_count;self.replan_steps.append(env.step_count);self.grid=grid

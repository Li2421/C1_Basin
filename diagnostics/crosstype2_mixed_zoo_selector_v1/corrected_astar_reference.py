"""Engineering-only correction; never used for this round's frozen rollouts.

The original frozen search is retained. Its candidate path is smoothed to the
straight segment only when every segment remains geometrically and temporally
valid against the already planned robot discs. Slot count and endpoint holding
are unchanged. This changes nominal execution semantics and needs a fresh study.
"""
from common import *
from fast_astar import FastGrid,FastPrioritizedAStar
class StraightnessCheckedGrid(FastGrid):
 def search(self,start,goal):
  original=super().search(start,goal)
  if original is None:return None
  self.reserved.pop()
  candidate=np.linspace(np.asarray(start,float),np.asarray(goal,float),len(original))
  valid=all(self.valid_edge(candidate[t],candidate[t+1],t) for t in range(len(candidate)-1))
  path=candidate if valid else original
  self.reserved.append(path);return path
class CorrectedPrioritizedAStar(FastPrioritizedAStar):
 def plan(self,env):
  res,K,priority=self.parameters;cfg=env.config
  grid=StraightnessCheckedGrid(res,max(cfg.dt,res/.52),cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin,cfg.outer_radius-cfg.agent_radius-cfg.collision_margin,2*cfg.agent_radius+cfg.agent_collision_margin,700)
  order=list(range(len(env.positions)))
  if priority=='far':order=sorted(order,key=lambda i:(-np.linalg.norm(env.goals[i]-env.positions[i]),i))
  self.paths={}
  for i in order:
   path=grid.search(env.positions[i],env.goals[i]);self.paths[i]=path
   if path is None:grid.reserved.append(np.asarray([env.positions[i]]))
  self.last_replan=env.step_count;self.replan_steps.append(env.step_count);self.grid=grid
 def nominal(self,env,key,traced=False):
  if getattr(self,'episode_env',None) is not env:self.episode_env=env;self.last_replan=None;self.replan_steps=[]
  return super().nominal(env,key,traced)

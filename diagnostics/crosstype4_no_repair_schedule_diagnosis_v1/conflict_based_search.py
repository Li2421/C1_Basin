"""Sum-of-costs conflict tree over the original continuous-disc space-time grid."""
from common import *
from spacetime_astar import SpaceTimeGrid,segment_distance
from fast_astar import FastPrioritizedAStar
import ctypes,heapq,itertools
ptr=ctypes.POINTER(ctypes.c_double);iptr=ctypes.POINTER(ctypes.c_int)
lib=ctypes.CDLL(str(HERE/'cbs_kernel.so'))
lib.astar_constrained.argtypes=[ctypes.c_double]*4+[ctypes.c_int,ptr,ptr,ctypes.c_int,iptr,ptr,ctypes.c_int,ptr,ptr,ptr];lib.astar_constrained.restype=ctypes.c_int
class ConstraintGrid(SpaceTimeGrid):
    def __init__(self,*args,constraints=(),bounds=(-np.inf,np.inf,-np.inf,np.inf),**kwargs):super().__init__(*args,**kwargs);self.constraints=tuple(constraints);self.bounds=bounds
    def search(self,start,goal):
        start=np.ascontiguousarray(start,np.float64);goal=np.ascontiguousarray(goal,np.float64);lens=np.asarray([len(p) for p in self.reserved],np.int32);reserved=np.ascontiguousarray(np.concatenate(self.reserved) if self.reserved else np.zeros((0,2)),np.float64);negative=np.ascontiguousarray(self.constraints,np.float64).reshape(-1,5);bounds=np.ascontiguousarray(self.bounds,np.float64);out=np.zeros((self.max_time+2,2),np.float64)
        length=lib.astar_constrained(self.res,self.inner,self.outer,self.sep,self.max_time,start.ctypes.data_as(ptr),goal.ctypes.data_as(ptr),len(lens),lens.ctypes.data_as(iptr),reserved.ctypes.data_as(ptr),len(negative),negative.ctypes.data_as(ptr),bounds.ctypes.data_as(ptr),out.ctypes.data_as(ptr))
        if not length:return None
        return out[:length].copy()
def edge(path,t):return path[min(t,len(path)-1)],path[min(t+1,len(path)-1)]
def first_conflict(paths,separation):
    horizon=max(len(p) for p in paths)
    for t in range(horizon):
        for i in range(len(paths)):
            a,b=edge(paths[i],t)
            for j in range(i+1,len(paths)):
                c,d=edge(paths[j],t)
                if segment_distance(a-c,b-d)<separation-1e-12:return i,j,t,a,b,c,d
    return None

def search(starts,goals,resolution,quantum,inner,outer,separation,max_nodes,max_time=700,bounds=(-np.inf,np.inf,-np.inf,np.inf)):
    cache={}
    def single(i,constraints):
        key=(i,constraints)
        if key not in cache:cache[key]=ConstraintGrid(resolution,quantum,inner,outer,separation,max_time,constraints=constraints,bounds=bounds).search(starts[i],goals[i])
        return cache[key]
    constraints=tuple(() for _ in starts);paths=[single(i,()) for i in range(len(starts))]
    if any(p is None for p in paths):return paths,dict(fallback=True,reason='Individual grid path absent',expanded_nodes=0,unresolved_conflict=True,lowlevel_searches=len(cache))
    serial=itertools.count();queue=[(sum(len(p)-1 for p in paths),next(serial),constraints,paths)];seen={constraints};expanded=0
    while queue:
        cost,_,constraints,paths=heapq.heappop(queue);expanded+=1;conflict=first_conflict(paths,separation)
        if conflict is None:return paths,dict(fallback=False,reason=None,expanded_nodes=expanded,unresolved_conflict=False,sum_of_costs=cost,lowlevel_searches=len(cache))
        if expanded>=max_nodes:return paths,dict(fallback=True,reason='Conflict-tree node budget reached',expanded_nodes=expanded,unresolved_conflict=True,sum_of_costs=cost,lowlevel_searches=len(cache))
        i,j,t,a,b,c,d=conflict
        for robot,u,v in ((i,a,b),(j,c,d)):
            negative=(int(t),float(u[0]),float(u[1]),float(v[0]),float(v[1]));current=list(constraints);current[robot]=tuple(sorted((*current[robot],negative)));current=tuple(current)
            if current in seen:continue
            seen.add(current);path=single(robot,current[robot])
            if path is None:continue
            updated=list(paths);updated[robot]=path;heapq.heappush(queue,(sum(len(p)-1 for p in updated),next(serial),current,updated))
    return paths,dict(fallback=True,reason='Conflict tree exhausted within finite grid horizon',expanded_nodes=expanded,unresolved_conflict=True,lowlevel_searches=len(cache))
class ConflictBasedSearch(FastPrioritizedAStar):
    def __init__(self,parameters,manifest):
        super().__init__(parameters,manifest);self.episode_env=None;self.fallback_count=0;self.planning_records=[]
    def plan(self,env):
        res,interval,budget=self.parameters;cfg=env.config;quantum=max(cfg.dt,res/.52);paths,diagnostic=search(env.positions,env.goals,res,quantum,cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin,cfg.outer_radius-cfg.agent_radius-cfg.collision_margin,2*cfg.agent_radius+cfg.agent_collision_margin,int(budget))
        self.paths=dict(enumerate(paths));self.grid=SpaceTimeGrid(res,quantum);self.last_replan=env.step_count;self.replan_steps.append(env.step_count);self.fallback_count+=int(diagnostic['fallback']);self.planning_records.append(dict(step=env.step_count,**diagnostic))
    def nominal(self,env,key,traced=False):
        if env is not self.episode_env:self.episode_env=env;self.last_replan=None;self.replan_steps=[];self.fallback_count=0;self.planning_records=[]
        return super().nominal(env,key,traced)

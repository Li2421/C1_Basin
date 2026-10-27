"""Original priority/CBS search and tracker, arbitrary finite wall segments."""
from experiment import *
from controllers import bound
import ctypes
import heapq
import itertools
import numpy as np

PTR=ctypes.POINTER(ctypes.c_double);IPTR=ctypes.POINTER(ctypes.c_int)
ASTAR=ctypes.CDLL(str(HERE/'astar_kernel_certified.so'));CBS=ctypes.CDLL(str(HERE/'cbs_kernel_certified.so'))
ASTAR.astar.argtypes=[ctypes.c_double]*4+[ctypes.c_int,PTR,PTR,ctypes.c_int,IPTR,PTR,ctypes.c_int,PTR,ctypes.c_double,PTR]
CBS.astar_constrained.argtypes=[ctypes.c_double]*4+[ctypes.c_int,PTR,PTR,ctypes.c_int,IPTR,PTR,ctypes.c_int,PTR,PTR,ctypes.c_int,PTR,ctypes.c_double,PTR]
ASTAR.astar.restype=CBS.astar_constrained.restype=ctypes.c_int

def segment_distance(a,b):
    d=b-a;t=np.clip(-np.dot(a,d)/max(np.dot(d,d),1e-30),0,1);return float(np.linalg.norm(a+t*d))
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

class Grid:
    def __init__(self,env,res,quantum):
        self.env=env;self.res=res;self.dt=quantum;self.max_time=700;self.reserved=[];self.walls=np.ascontiguousarray(env.walls,np.float64);self.margin=env.instance.geometry.margin
    def search(self,start,goal,constraints=None):
        start=np.ascontiguousarray(start,np.float64);goal=np.ascontiguousarray(goal,np.float64);lens=np.asarray([len(p) for p in self.reserved],np.int32)
        reserved=np.ascontiguousarray(np.concatenate(self.reserved) if self.reserved else np.zeros((0,2)),np.float64);out=np.zeros((self.max_time+2,2),np.float64)
        args=[self.res,0.,float('inf'),2*self.env.config.agent_radius+self.env.config.agent_collision_margin,self.max_time,start.ctypes.data_as(PTR),goal.ctypes.data_as(PTR),len(lens),lens.ctypes.data_as(IPTR),reserved.ctypes.data_as(PTR)]
        if constraints is not None:
            negative=np.ascontiguousarray(constraints,np.float64).reshape(-1,5);bounds=np.asarray([-np.inf,np.inf,-np.inf,np.inf],np.float64)
            args.extend([len(negative),negative.ctypes.data_as(PTR),bounds.ctypes.data_as(PTR)]);fn=CBS.astar_constrained
        else:fn=ASTAR.astar
        args.extend([len(self.walls),self.walls.ctypes.data_as(PTR),self.margin,out.ctypes.data_as(PTR)])
        length=fn(*args)
        if not length:return None
        path=out[:length].copy()
        if constraints is None:self.reserved.append(path)
        return path

def cbs_search(env,res,quantum,max_nodes):
    cache={};grid=Grid(env,res,quantum);starts=env.positions;goals=env.goals
    def single(i,constraints):
        key=(i,constraints)
        if key not in cache:cache[key]=grid.search(starts[i],goals[i],constraints)
        return cache[key]
    constraints=tuple(() for _ in starts);paths=[single(i,()) for i in range(len(starts))]
    if any(p is None for p in paths):return paths,dict(fallback=True,reason='Individual grid path absent',expanded_nodes=0)
    serial=itertools.count();queue=[(sum(len(p)-1 for p in paths),next(serial),constraints,paths)];seen={constraints};expanded=0
    while queue:
        cost,_,constraints,paths=heapq.heappop(queue);expanded+=1;conflict=first_conflict(paths,2*env.config.agent_radius+env.config.agent_collision_margin)
        if conflict is None:return paths,dict(fallback=False,reason=None,expanded_nodes=expanded)
        if expanded>=max_nodes:return paths,dict(fallback=True,reason='Conflict-tree node budget reached',expanded_nodes=expanded)
        i,j,t,a,b,c,d=conflict
        for robot,u,v in ((i,a,b),(j,c,d)):
            negative=(int(t),float(u[0]),float(u[1]),float(v[0]),float(v[1]));current=list(constraints);current[robot]=tuple(sorted((*current[robot],negative)));current=tuple(current)
            if current in seen:continue
            seen.add(current);path=single(robot,current[robot])
            if path is None:continue
            updated=list(paths);updated[robot]=path;heapq.heappush(queue,(sum(len(p)-1 for p in updated),next(serial),current,updated))
    return paths,dict(fallback=True,reason='Conflict tree exhausted within finite grid horizon',expanded_nodes=expanded)

class Planner:
    def __init__(self,row):self.parameters=json.loads(row['parameters']);self.kind=row['type_code'];self.last_replan=None;self.paths={};self.planning_records=[]
    def plan(self,env):
        res,interval,policy=self.parameters;quantum=max(env.config.dt,res/env.config.max_speed);self.grid=Grid(env,res,quantum)
        if self.kind=='conflict_based_search':
            paths,record=cbs_search(env,res,quantum,int(policy));self.paths=dict(enumerate(paths));self.planning_records.append(record)
        else:
            order=list(range(len(env.positions)))
            if policy=='far':order=sorted(order,key=lambda i:(-np.linalg.norm(env.goals[i]-env.positions[i]),i))
            self.paths={}
            for i in order:
                path=self.grid.search(env.positions[i],env.goals[i]);self.paths[i]=path
                if path is None:self.grid.reserved.append(np.asarray([env.positions[i]]))
        self.last_replan=env.step_count
    def action(self,env):
        if self.last_replan is None or env.step_count-self.last_replan>=self.parameters[1]:self.plan(env)
        result=[];elapsed=(env.step_count-self.last_replan)*env.config.dt
        for i,pos in enumerate(env.positions):
            path=self.paths[i]
            if path is None:result.append(np.zeros(2));continue
            slot=min(int(elapsed/self.grid.dt),len(path)-1)
            if slot<len(path)-1 and np.linalg.norm(path[slot+1]-path[slot])<1e-12:result.append(np.zeros(2));continue
            nearest=slot+int(np.argmin(np.linalg.norm(path[slot:]-pos,axis=-1)));target=path[-1];length=0.;previous=pos
            for point in path[nearest:]:
                segment=np.linalg.norm(point-previous)
                if length+segment>=.3:target=previous+(.3-length)/max(segment,1e-30)*(point-previous);break
                length+=segment;previous=point
            delta=target-pos;speed=env.config.max_speed*min(1.,np.linalg.norm(env.goals[i]-pos)/.3);result.append(delta/max(np.linalg.norm(delta),1e-30)*speed)
        return bound(np.asarray(result),env.config.max_speed)

from controllers import Nominal as CanonicalNominal
class MemoNominal(CanonicalNominal):
    def action(self,env,key):
        if self.kind in ('prioritized_astar','conflict_based_search'):
            if not hasattr(self,'planner'):self.planner=Planner(self.row)
            return self.planner.action(env)
        return super().action(env,key)

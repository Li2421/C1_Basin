"""Exact no-incoming-goal-vertex certificate; canonical search otherwise."""
from experiment import *
import numpy as np
import heapq
import itertools
from tiled_planning_pa_memo_query_disk_candidate_20261022 import Grid as CanonicalGrid,Planner as CanonicalPlanner,first_conflict
from nominal_adapter import Nominal as CanonicalNominal,EmptyWallControllerView
from controllers import bound

def no_goal_vertex(start,goal,reserved,resolution,max_time,separation):
    start=np.asarray(start,np.float64);goal=np.asarray(goal,np.float64)
    start_grid=np.rint(start/resolution).astype(np.int64);goal_grid=np.rint(goal/resolution).astype(np.int64)
    if np.array_equal(start_grid,goal_grid) or not reserved:return None
    earliest=int(np.max(np.abs(start_grid-goal_grid)))
    if earliest>=max_time:
        return dict(reason='Goal grid vertex requires at least max_time moves',earliest=earliest,max_time=max_time)
    # A native incoming edge checks the reserved position at its endpoint.
    # Use a stricter numerical margin than the native sep-1e-12 predicate.
    threshold=separation-1e-10
    times=np.arange(earliest,max_time,dtype=int);vertex=goal_grid*resolution
    witnesses=np.full(len(times),-1,dtype=int)
    for j,path in enumerate(reserved):
        path=np.asarray(path,np.float64)
        distance=np.linalg.norm(vertex-path[np.minimum(times,len(path)-1)],axis=-1)
        witnesses[(witnesses<0)&(distance<threshold)]=j
    if np.any(witnesses<0):return None
    runs=[];begin=0
    for k in range(1,len(times)+1):
        if k==len(times) or witnesses[k]!=witnesses[begin]:
            runs.append([int(times[begin]),int(times[k-1]),int(witnesses[begin])]);begin=k
    return dict(reason='Every reachable arrival time has a blocked incoming goal vertex',earliest=earliest,max_time=max_time,
        goal_grid_vertex=vertex.tolist(),blocked_time_witness_runs=runs,numerical_certificate_margin=1e-10,
        native_collision_margin=1e-12,reserved_sha256=hashlib.sha256(b''.join(np.asarray(p,np.float64).tobytes() for p in reserved)).hexdigest())

class CertifiedGrid(CanonicalGrid):
    def __init__(self,env,res,quantum,certificates):super().__init__(env,res,quantum);self.certificates=certificates
    def search(self,start,goal,constraints=None):
        proof=no_goal_vertex(start,goal,self.reserved,self.res,self.max_time,2*self.env.config.agent_radius+self.env.config.agent_collision_margin)
        if proof is not None:
            self.certificates.append(dict(step=self.env.step_count,start=np.asarray(start).tolist(),goal=np.asarray(goal).tolist(),**proof));return None
        return super().search(start,goal,constraints)

def cbs_search(env,res,quantum,max_nodes,certificates):
    # Same conflict-tree algorithm, with only the Grid factory extended.
    cache={};grid=CertifiedGrid(env,res,quantum,certificates);starts=env.positions;goals=env.goals
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

class CertifiedPlanner(CanonicalPlanner):
    def __init__(self,row):super().__init__(row);self.certificates=[]
    def plan(self,env):
        res,interval,policy=self.parameters;quantum=max(env.config.dt,res/env.config.max_speed);self.grid=CertifiedGrid(env,res,quantum,self.certificates)
        if self.kind=='conflict_based_search':
            paths,record=cbs_search(env,res,quantum,int(policy),self.certificates);self.paths=dict(enumerate(paths));self.planning_records.append(record)
        else:
            order=list(range(len(env.positions)))
            if policy=='far':order=sorted(order,key=lambda i:(-np.linalg.norm(env.goals[i]-env.positions[i]),i))
            self.paths={}
            for i in order:
                path=self.grid.search(env.positions[i],env.goals[i]);self.paths[i]=path
                if path is None:self.grid.reserved.append(np.asarray([env.positions[i]]))
        self.last_replan=env.step_count

class ReservationNominal(CanonicalNominal):
    def action(self,env,key):
        if self.kind not in ('prioritized_astar','conflict_based_search'):return super().action(env,key)
        view=EmptyWallControllerView(env) if env.config.scene=='lit_crowded_random' and not len(env.walls) else env
        if not hasattr(self,'planner'):self.planner=CertifiedPlanner(self.row)
        return self.planner.action(view)

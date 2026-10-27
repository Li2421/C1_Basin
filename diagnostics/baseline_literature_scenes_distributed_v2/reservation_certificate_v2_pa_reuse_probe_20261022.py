"""Exact rejection when every canonical terminal-goal test must fail."""
from experiment import *
import numpy as np
from reservation_certificate_pa_reuse_probe_20261022 import CertifiedPlanner as FirstPlanner,CertifiedGrid as FirstGrid
from reservation_certificate_pa_reuse_probe_20261022 import cbs_search,CanonicalNominal,EmptyWallControllerView

def swept_distance(a,b):
    delta=b-a;denominator=np.maximum(np.sum(delta*delta,axis=-1),1e-30)
    fraction=np.clip(-np.sum(a*delta,axis=-1)/denominator,0,1)
    return np.linalg.norm(a+fraction[...,None]*delta,axis=-1)

def no_terminal_goal(start,goal,reserved,resolution,max_time,separation):
    if not reserved:return None
    start=np.asarray(start,np.float64);goal=np.asarray(goal,np.float64)
    initial=np.rint(start/resolution).astype(np.int64);grid=np.rint(goal/resolution).astype(np.int64);vertex=grid*resolution
    earliest=int(np.max(abs(initial-grid)));threshold=separation-1e-10
    if earliest>=max_time:return dict(reason='Goal grid requires more moves than any expanded state',earliest=earliest,max_time=max_time)
    times=np.arange(max_time,dtype=int);paths=[np.asarray(p,np.float64) for p in reserved]
    before=np.stack([p[np.minimum(times,len(p)-1)] for p in paths],axis=1)
    after=np.stack([p[np.minimum(times+1,len(p)-1)] for p in paths],axis=1)
    hold=swept_distance(goal-before,goal-after)<threshold
    forbidden_hold=np.any(hold,axis=1);future_hold=np.r_[np.maximum.accumulate(forbidden_hold[::-1])[::-1],False]
    origin=np.broadcast_to(vertex,(max_time,2)).copy();origin[0]=start
    terminal=swept_distance(origin[:,None]-before,goal-after)<threshold
    incoming=np.linalg.norm(vertex-before,axis=-1)<threshold
    incoming[0]=False
    rejected=[]
    moves=np.asarray([[1,0],[0,1],[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1],[0,0]])
    for t in range(earliest,max_time):
        if np.any(terminal[t]):rejected.append([t,'terminal_transition',int(np.flatnonzero(terminal[t])[0])]);continue
        if future_hold[t+1]:rejected.append([t,'future_stationary_goal_conflict']);continue
        if t and np.any(incoming[t]):rejected.append([t,'blocked_goal_vertex',int(np.flatnonzero(incoming[t])[0])]);continue
        if t:
            predecessors=grid[None]-moves
            possible=np.max(abs(predecessors-initial),axis=-1)<=t-1
            predecessors=predecessors[possible]
            positions=predecessors*resolution
            if t==1:positions=np.broadcast_to(start,positions.shape)
            if not len(positions):rejected.append([t,'no_reachable_predecessor_grid']);continue
            r0=np.stack([p[min(t-1,len(p)-1)] for p in paths]);r1=np.stack([p[min(t,len(p)-1)] for p in paths])
            blocked=swept_distance(positions[:,None]-r0,vertex-r1)<threshold
            if np.all(np.any(blocked,axis=1)):
                rejected.append([t,'all_reachable_predecessor_edges_blocked']);continue
        return None
    # Every rejection is a necessary canonical goal/incoming-edge condition.
    ranges=[]
    for item in rejected:
        t,*witness=item
        if ranges and ranges[-1][1]+1==t and ranges[-1][2:]==witness:ranges[-1][1]=t
        else:ranges.append([t,t,*witness])
    return dict(reason='All expanded-time canonical terminal-goal tests are certified impossible',earliest=earliest,max_time=max_time,
        rejected_arrival_time_ranges=ranges,grid_vertex=vertex.tolist(),separation=separation,strict_numerical_guard=1e-10,
        reserved_sha256=hashlib.sha256(b''.join(p.tobytes() for p in paths)).hexdigest())

class CertifiedGrid(FirstGrid):
    def search(self,start,goal,constraints=None):
        proof=no_terminal_goal(start,goal,self.reserved,self.res,self.max_time,2*self.env.config.agent_radius+self.env.config.agent_collision_margin)
        if proof is not None:
            self.certificates.append(dict(step=self.env.step_count,start=np.asarray(start).tolist(),goal=np.asarray(goal).tolist(),**proof));return None
        return super().search(start,goal,constraints)

class CertifiedPlanner(FirstPlanner):
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

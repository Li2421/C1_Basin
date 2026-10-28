"""Constructive witnesses only: original nominal/safety, temporary task goals."""
from core import *
import runtime
import signal,time

class SequentialRelease:
    def __init__(self,c):
        self.cfg=RN.LiteratureConfig(**c['scenario_payload']['config'])
        self.recipe=c['controller_payload']['witness'];self.groups=self.recipe['groups'];self.group=0
        self.parked={};self.parking=None;self.last=[];self.calls=0;self.events=[];self.finished=False
    def targets(self,geom,p,goals):
        cfg=self.cfg;t=self.calls;self.calls+=1
        arrived=np.linalg.norm(goals-p,axis=1)<=cfg.goal_tolerance
        while self.group<len(self.groups) and all(arrived[i]for i in self.groups[self.group]):
            self.events.append(dict(step=t,event='group_arrived',group=self.groups[self.group]));self.group+=1;self.last=[]
        target=p.copy()
        if self.group==len(self.groups):self.finished=True;return goals.copy()
        active=[i for i in self.groups[self.group]if not arrived[i]]
        if self.parking is not None:
            j,point=self.parking
            if np.linalg.norm(p[j]-point)<=cfg.goal_tolerance:
                self.events.append(dict(step=t,event='parked',agent=j,point=point.tolist()));self.parked[j]=point;self.parking=None;self.last=[]
            else:
                target[j]=point;return target
        target[active]=goals[active]
        remaining=[]
        for i in active:
            path=geom.path(p[i],goals[i])
            remaining.append(float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())if path is not None else float('inf'))
        self.last.append((t,sum(remaining)));window=int(round(2/cfg.dt))
        self.last=[x for x in self.last if x[0]>=t-window]
        if len(self.last)<window+1 or self.last[0][1]-self.last[-1][1]>=.05:return target
        waiting=[i for group in self.groups[self.group+1:]for i in group if not arrived[i]]
        if not waiting:self.last=[];return target
        paths=[geom.path(p[i],goals[i])for i in range(len(p))if not arrived[i]]
        def path_distance(q,path):
            if path is None:return float('inf')
            if len(path)==1:return float(np.linalg.norm(q-path[0]))
            a,b=path[:-1],path[1:];v=b-a
            w=np.clip(np.sum((q-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-30),0,1)
            return float(np.min(np.linalg.norm(q-(a+w[:,None]*v),axis=1)))
        active_paths=[geom.path(p[i],goals[i])for i in active]
        waiting.sort(key=lambda j:(min(path_distance(p[j],path)for path in active_paths),j))
        bounds=self.recipe['bounds'];xs=np.arange(bounds[0]+cfg.agent_radius,bounds[2]-cfg.agent_radius+1e-12,.05);ys=np.arange(bounds[1]+cfg.agent_radius,bounds[3]-cfg.agent_radius+1e-12,.05)
        candidates=np.array([[x,y]for x in xs for y in ys]);candidates=candidates[geom.valid_points(candidates)]
        for path in paths:
            candidates=np.asarray([q for q in candidates if path_distance(q,path)>=2.5*(2*cfg.agent_radius)])
            if not len(candidates):break
        for j in waiting:
            valid=[q for q in candidates if np.min(np.linalg.norm(np.delete(p,j,axis=0)-q,axis=1),initial=np.inf)>=2*cfg.agent_radius+.05]
            valid.sort(key=lambda q:(float(np.linalg.norm(q-p[j])),float(q[0]),float(q[1])))
            for point in valid:
                if geom.path(p[j],point)is None:continue
                self.parking=(j,point.copy());self.events.append(dict(step=t,event='clear_path',agent=j,point=point.tolist(),active=active));self.last=[]
                target=p.copy();target[j]=point;return target
        self.events.append(dict(step=t,event='no_valid_docking_point',active=active));self.last=[]
        return target

def run(c,original_run):
    recipe=c['controller_payload']['witness'];provider=None;previous=runtime.guidance
    if recipe['kind']=='sequential_release':
        provider=SequentialRelease(c)
        def guidance(geom,p,true_goals,enabled):
            targets=provider.targets(geom,p,true_goals)
            guides,_,detail=previous(geom,p,targets,enabled)
            _,remaining,_=previous(geom,p,true_goals,enabled)
            return guides,remaining,detail
        runtime.guidance=guidance
    def timeout(signum,frame):raise TimeoutError('Witness tool exceeded1800s pilot budget')
    oldhandler=signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:result,arrays=original_run(c)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,oldhandler);runtime.guidance=previous
    result['witness']=dict(kind=recipe['kind'],task_horizon_seconds=recipe['task_horizon_steps']*c['scenario_payload']['config']['dt'],witness_horizon_seconds=result['horizon_time'],within_task_horizon=bool(result['success']and result['completion_time']<=recipe['task_horizon_steps']*c['scenario_payload']['config']['dt']),events=provider.events if provider else [],tool_timeout=bool(result.get('error')and result['error']['type']=='TimeoutError'))
    return result,arrays

def make(scene,ctl,index,kind,order='shortest'):
    from runner import case
    copied=json.loads(json.dumps(scene));task_steps=scene['config']['max_steps'];copied['config']['max_steps']=3*task_steps
    c=case(copied,ctl,index,purpose='joint_feasibility_witness_v2')
    p=np.asarray(c['state']['positions']);paths=scene['initial_paths'][index]
    lengths=[float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())for path in paths]
    if order in ('left_first','right_first'):
        groups=[list(range(len(p)//2)),list(range(len(p)//2,len(p)))]
        if order=='right_first':groups.reverse()
    else:groups=[[i]for i in sorted(range(len(p)),key=lambda i:(lengths[i],i))]
    c['controller_payload']['witness']=dict(kind=kind,order=order,groups=groups,bounds=scene['bounds'],parking_grid=.05,stall_window_seconds=2.,stall_progress=.05,path_clearance_diameters=2.5,task_horizon_steps=task_steps,source_sha256=sha(__file__),worker_sha256=sha(HERE/'worker_witness.py'))
    c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));return c

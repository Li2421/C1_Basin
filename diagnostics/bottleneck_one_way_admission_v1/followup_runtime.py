"""Frozen follow-up coordination schemes, all before the original second filter."""
import numpy as np
from admission import AdmissionState,apply as admission
from passages import entry
from core import RN

DEFAULT=dict(R_e=.8,R_c=.5,rho='fcfs',K=None,c1=1.,c2=.5,c3=1.)

def relations(p,g,remaining,paths,cfg,gate,lookahead):
    u=np.asarray(gate['axis']);n=np.array([-u[1],u[0]]);q=p-gate['centre']
    geometric=(np.abs(q@u)<=gate['half_length']+cfg.agent_radius)&(np.abs(q@n)<=gate['width']/2)
    inside=geometric&(remaining>cfg.goal_tolerance);request=np.zeros(len(p),bool);direction=np.zeros(len(p),int);distance=np.full(len(p),np.inf)
    direction[inside]=np.sign((g-p)@u)[inside].astype(int)
    for i in range(len(p)):
        if not inside[i]and remaining[i]>cfg.goal_tolerance:
            ent=entry(paths[i],gate,cfg.agent_radius,lookahead)
            if ent is not None and ent[1]:distance[i],direction[i]=ent;request[i]=True
    return geometric,inside,request,direction,distance

def fcfs(p,gate,inside,request,direction,distance):
    ids=np.flatnonzero(inside&(direction!=0));u=np.asarray(gate['axis'])
    if len(ids):return int(direction[min(ids,key=lambda i:(np.linalg.norm(p[i]-(np.asarray(gate['centre'])+direction[i]*gate['half_length']*u)),i))])
    ids=np.flatnonzero(request&(direction!=0))
    return int(direction[min(ids,key=lambda i:(distance[i],i))])if len(ids)else 0

def meta(gate,s,geometric,inside,request,direction,waiting,**kw):
    return dict(direction=int(s),geometry_inside=geometric.tolist(),inside=inside.tolist(),request=request.tolist(),robot_direction=direction.tolist(),waiting=waiting.tolist(),**kw)

class Coordinator:
    def __init__(self,settings,count):
        self.spec=settings;self.scheme=settings['scheme'];self.admission=AdmissionState(count)
        self.clocks=[dict(phase='idle',direction=0,start=None)for _ in range(count)]
        self.reservations=[{}for _ in range(count)];self.expired=[{}for _ in range(count)]
        self.pair=None;self.yielder=None;self.passer=None;self.waiting_points={}

    def apply(self,target,p,g,remaining,paths,cfg,gates,filters,safe,step,geom):
        p=np.asarray(p);g=np.asarray(g);remaining=np.asarray(remaining)
        if paths is None:paths=[np.stack([x,y])for x,y in zip(p,g)]
        method=self.scheme
        if method=='B':
            c=self.spec['c'];return admission(target,p,g,remaining,paths,cfg,gates,self.admission,dict(DEFAULT,c1=c,c2=c,c3=c))
        if method=='C':
            _,history=admission(target,p,g,remaining,paths,cfg,gates,self.admission,dict(DEFAULT,c1=0.,c2=0.,c3=0.))
            out=target.copy()
            for record in history:
                waiting=np.asarray(record['request'])&(np.asarray(record['robot_direction'])!=record['direction'])if record['direction']else np.zeros(len(p),bool)
                out[waiting]*=1-self.spec['c'];record['waiting']=waiting.tolist()
            return out,history
        if method=='D':return self.signal(target,p,g,remaining,paths,cfg,gates,step)
        if method=='E':return self.team(target,p,g,remaining,cfg,filters,safe)
        if method=='F':return self.reserve(target,p,g,remaining,paths,cfg,gates,step)
        if method=='G':return self.waypoints(target,p,g,remaining,paths,cfg,gates,geom)
        raise ValueError('Unknown frozen followup scheme '+str(method))

    def signal(self,target,p,g,remaining,paths,cfg,gates,step):
        out=target.copy();history=[];now=step*cfg.dt
        for gate,clock in zip(gates,self.clocks):
            geometric,inside,request,direction,distance=relations(p,g,remaining,paths,cfg,gate,.8)
            if clock['phase']=='idle':
                s=fcfs(p,gate,inside,request,direction,distance)
                if s:clock.update(phase='green',direction=s,start=now)
            if clock['phase']=='green'and now-clock['start']>=self.spec['T']-1e-12:clock['phase']='clearing'
            if clock['phase']=='clearing'and not np.any(inside):clock.update(phase='green',direction=-clock['direction'],start=now)
            s=clock['direction']if clock['phase']=='green'else 0
            waiting=request&((direction!=s)if s else np.ones(len(p),bool))
            out[waiting]=0.
            history.append(meta(gate,s,geometric,inside,request,direction,waiting,scheme='D',phase=clock['phase'],phase_start=clock['start']))
        return out,history

    def team(self,target,p,g,remaining,cfg,filters,safe):
        delta=g-p;norm=np.linalg.norm(delta,axis=1);heading=delta/np.maximum(norm,1e-30)[:,None]
        if self.pair is not None and (np.linalg.norm(p[self.pair[0]]-p[self.pair[1]])>.5 or all(remaining[i]<=cfg.goal_tolerance for i in self.pair)):
            self.pair=None;self.yielder=None;self.passer=None
        if self.pair is None:
            pairs={};tau=.05*cfg.max_speed
            for i,f in enumerate(filters):
                for row,lower,label in zip(f.A,f.lower,f.kinds):
                    if label[0]!='pair':continue
                    j=int(label[2]);rn=np.linalg.norm(row)
                    if rn<1e-14 or heading[i]@heading[j]>=-.5:continue
                    slack=float((row@safe[i]-lower)/rn)
                    if slack>tau or(remaining[i]<=cfg.goal_tolerance and remaining[j]<=cfg.goal_tolerance):continue
                    pair=tuple(sorted((i,j)));pairs[pair]=min(slack,pairs.get(pair,float('inf')))
            if pairs:
                self.pair=min(pairs,key=lambda ij:(pairs[ij],ij));unfinished=[i for i in self.pair if remaining[i]>cfg.goal_tolerance]
                self.passer=min(unfinished,key=lambda i:(remaining[i],i));self.yielder=next(i for i in self.pair if i!=self.passer)
        team=[];out=target.copy();headon=False;distance=None
        if self.pair is not None:
            eligible=heading@heading[self.yielder]>=.5;eligible[self.yielder]=True
            seen={self.yielder};queue=[self.yielder];radius=RN.contact_radius(cfg)
            while queue:
                i=queue.pop(0)
                for j in range(len(p)):
                    if j not in seen and eligible[j]and np.linalg.norm(p[i]-p[j])<=radius:seen.add(j);queue.append(j)
            team=sorted(seen);c=self.spec['c'];out[team]*=1-c;out[team]-=c*cfg.max_speed*heading[team]
            i,j=self.pair;distance=float(np.linalg.norm(p[i]-p[j]));headon=bool(heading[i]@heading[j]<-.5 and distance<=.5)
        return out,[dict(scheme='E',pair=self.pair,yielder=self.yielder,passer=self.passer,team=team,headon=headon,pair_distance=distance)]

    def reserve(self,target,p,g,remaining,paths,cfg,gates,step):
        out=target.copy();history=[];now=step*cfg.dt;vp=self.spec['v_p'];speed=cfg.max_speed
        heading=(g-p)/np.maximum(np.linalg.norm(g-p,axis=1),1e-30)[:,None]
        for j,gate in enumerate(gates):
            geometric,inside,request,direction,distance=relations(p,g,remaining,paths,cfg,gate,1.2)
            table=self.reservations[j];expired=self.expired[j];u=np.asarray(gate['axis'])
            # Record actual entry before assessing the missed-entry timeout.
            for i,res in list(table.items()):
                if inside[i]:res['entered']=True
                if(not res['entered']and now>res['enter']+2.)or now>res['leave']:
                    del table[i];expired[i]=step
            duration=(2*gate['half_length']+2*cfg.agent_radius)/vp;spacing=2*cfg.agent_radius/vp
            for i in np.flatnonzero(request):
                if int(i)in table or expired.get(int(i))==step:continue
                proposed=now+distance[i]/speed;s=int(direction[i])
                while True:
                    updated=proposed
                    for res in sorted(table.values(),key=lambda r:(r['enter'],r['robot'])):
                        if res['direction']==s:
                            if abs(proposed-res['enter'])<spacing-1e-12:updated=max(updated,res['enter']+spacing)
                        elif not(proposed+duration+1.<=res['enter']or proposed>=res['leave']+1.):updated=max(updated,res['leave']+1.)
                    if updated<=proposed+1e-12:break
                    proposed=updated
                table[int(i)]=dict(robot=int(i),direction=s,enter=float(proposed),leave=float(proposed+duration),entered=False)
            waiting=np.zeros(len(p),bool);reverse=np.zeros(len(p),bool);admitted=[]
            for i,res in sorted(table.items()):
                if request[i]and now<res['enter']:
                    cap=distance[i]/(res['enter']-now);forward=float(out[i]@heading[i])
                    if forward>cap and forward>0:out[i]*=max(0.,cap/forward);waiting[i]=True
                if res['enter']<=now<=res['leave']:admitted.append(res)
            for i in np.flatnonzero(inside):
                if int(i)in table:continue
                opposite=[res for robot,res in table.items()if(inside[robot]or request[robot])and res['direction']==-direction[i]and direction[i]!=0]
                if opposite:
                    selected=min(opposite,key=lambda r:(r['enter'],r['robot']));out[i]=speed*selected['direction']*u;reverse[i]=True
            s=min(admitted,key=lambda r:(r['enter'],r['robot']))['direction']if admitted else 0
            history.append(meta(gate,s,geometric,inside,request,direction,waiting,scheme='F',reverse=reverse.tolist(),reservations=[dict(r)for _,r in sorted(table.items())]))
        return out,history

    def waypoints(self,target,p,g,remaining,paths,cfg,gates,geom):
        menu=self.spec['menu'];_,history=admission(target,p,g,remaining,paths,cfg,gates,self.admission,dict(menu,c1=0.,c2=0.,c3=0.))
        out=target.copy()
        for record in history:out[np.asarray(record['waiting'])]*=1-menu['c1']
        for j,(gate,record)in enumerate(zip(gates,history)):
            s=record['direction'];eligible=np.flatnonzero(record['clear_exit']);record['waypoint_assignments']=[]
            if not s or not len(eligible):continue
            u=np.asarray(gate['axis']);n=np.array([-u[1],u[0]]);ex=np.asarray(gate['centre'])+s*gate['half_length']*u;cache=(j,s,menu['R_c'])
            if cache not in self.waiting_points:
                radius=menu['R_c']+.5;lo=np.ceil((ex-radius)/.05).astype(int);hi=np.floor((ex+radius)/.05).astype(int)
                grid=np.array([[x*.05,y*.05]for x in range(lo[0],hi[0]+1)for y in range(lo[1],hi[1]+1)])
                dist=np.linalg.norm(grid-ex,axis=1);valid=(dist>=2.5*cfg.agent_radius)&(dist<=radius)&(np.abs((grid-gate['centre'])@n)>=2*cfg.agent_radius+cfg.agent_collision_margin)
                grid=grid[valid];grid=grid[geom.valid_points(grid)]if len(grid)else grid
                self.waiting_points[cache]=np.array(sorted(grid.tolist(),key=lambda q:(np.linalg.norm(np.asarray(q)-ex),q[0],q[1]))).reshape(-1,2)
            candidates=self.waiting_points[cache];assigned=[]
            for i in sorted(eligible,key=lambda i:(np.linalg.norm(p[i]-ex),i)):
                available=[q for q in candidates if all(np.linalg.norm(q-other)>=2*cfg.agent_radius for other in assigned)]
                if not available:record['waypoint_assignments'].append(dict(robot=int(i),point=None));continue
                q=min(available,key=lambda q:(np.linalg.norm(q-p[i]),np.linalg.norm(q-ex),q[0],q[1]));assigned.append(q)
                delta=q-p[i];norm=np.linalg.norm(delta)
                if norm>.02:out[i]+=menu['c2']*cfg.max_speed*delta/norm
                record['waypoint_assignments'].append(dict(robot=int(i),point=q.tolist()))
        for gate,record in zip(gates,history):out[np.asarray(record['reverse'])]+=menu['c3']*cfg.max_speed*record['direction']*np.asarray(gate['axis'])
        return out,history

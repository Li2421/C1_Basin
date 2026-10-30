"""Frozen passage admission state machine; applied before original safety."""
import numpy as np
from passages import entry

class AdmissionState:
    def __init__(self,count):
        self.states=[dict(direction=0,count=0,previous_inside=None,prefer=0)for _ in range(count)]

def apply(target,positions,guides,remaining,paths,cfg,passages,state,menu):
    p=np.asarray(positions);out=target.copy();metadata=[];plans=[]
    if paths is None:paths=[np.stack([x,g])for x,g in zip(p,guides)]
    unfinished=np.asarray(remaining)>cfg.goal_tolerance
    for passage,turn in zip(passages,state.states):
        u=np.asarray(passage['axis']);n=np.array([-u[1],u[0]]);local=p-passage['centre'];ax=local@u
        geometry_inside=(np.abs(ax)<=passage['half_length']+cfg.agent_radius)&(np.abs(local@n)<=passage['width']/2)
        inside=geometry_inside&unfinished
        request=np.zeros(len(p),bool);direction=np.zeros(len(p),int);distance=np.full(len(p),np.inf)
        direction[inside]=np.sign((guides-p)@u)[inside].astype(int)
        for i in range(len(p)):
            if unfinished[i] and not inside[i]:
                ent=entry(paths[i],passage,cfg.agent_radius,menu['R_e'])
                if ent is not None and ent[1]!=0:request[i]=True;distance[i],direction[i]=ent
        s=turn['direction'];limit=menu['K']
        if turn['previous_inside'] is not None and s:
            turn['count']+=int(np.sum(inside & ~np.asarray(turn['previous_inside']) & (direction==s)))
        opposite=bool(np.any(request&(direction==-s)))if s else False
        quota=bool(s and limit is not None and turn['count']>=limit and opposite)
        if s and not np.any(inside&(direction==s)) and (not np.any(request&(direction==s))or quota):
            turn.update(direction=0,prefer=-s if quota else 0,count=0);s=0
        candidates=np.flatnonzero((inside|request)&(direction!=0))
        if not s and len(candidates):
            in_ids=[i for i in candidates if inside[i]]
            if in_ids:leader=min(in_ids,key=lambda i:(np.linalg.norm(p[i]-(np.asarray(passage['centre'])+direction[i]*passage['half_length']*u)),i))
            else:leader=min(candidates,key=lambda i:(distance[i],i))
            s=int(direction[leader]);prefer=turn['prefer']
            if prefer and np.any((inside|request)&(direction==prefer)):s=prefer
            elif menu['rho']=='majority':
                plus=np.sum(direction[candidates]==1);minus=np.sum(direction[candidates]==-1)
                if plus!=minus:s=1 if plus>minus else -1
            turn.update(direction=s,count=0,prefer=0)
        quota=bool(s and limit is not None and turn['count']>=limit and np.any(request&(direction==-s)))
        waiting=(request&((direction!=s)|quota))|(inside&(direction!=s))if s else np.zeros(len(p),bool)
        exit_point=np.asarray(passage['centre'])+s*passage['half_length']*u
        eligible=(np.linalg.norm(p-exit_point,axis=1)<=menu['R_c'])&~geometry_inside&~((inside|request)&(direction==s)) if s else np.zeros(len(p),bool)
        reverse=inside&(direction!=s)if s else np.zeros(len(p),bool)
        turn['previous_inside']=inside.tolist()
        plans.append((u,n,s,exit_point,waiting,eligible,reverse))
        metadata.append(dict(direction=s,admitted=turn['count'],quota_reached=quota,inside=inside.tolist(),geometry_inside=geometry_inside.tolist(),request=request.tolist(),robot_direction=direction.tolist(),waiting=waiting.tolist(),clear_exit=eligible.tolist(),reverse=reverse.tolist()))
    # Fixed global order: all slowdowns, then all exit-clearance terms, then
    # all reverse-exit terms. Multiple passages are independent and ordered.
    for u,n,s,ex,waiting,eligible,reverse in plans:out[waiting]*=1-menu['c1']
    for u,n,s,ex,waiting,eligible,reverse in plans:
        for i in np.flatnonzero(eligible):
            d=p[i]-ex;parallel=(d@u)*u;perp=(d@n)*n
            if np.linalg.norm(perp)<=1e-12:
                tangent=np.asarray(paths[i][1])-p[i] if unfinished[i] and len(paths[i])>1 else s*u
                norm=np.linalg.norm(tangent)
                # Exact zero route tangent is recorded; no undisclosed fallback.
                perp=np.array([tangent[1],-tangent[0]])/norm if norm>1e-12 else np.zeros(2)
            away=parallel+2*perp;norm=np.linalg.norm(away)
            if norm>1e-12:out[i]+=menu['c2']*cfg.max_speed*away/norm
    for u,n,s,ex,waiting,eligible,reverse in plans:out[reverse]+=menu['c3']*cfg.max_speed*s*u
    return out,metadata

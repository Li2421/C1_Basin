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
    geometric=[]
    for passage in passages:
        u=np.asarray(passage['axis']);n=np.array([-u[1],u[0]]);local=p-passage['centre']
        geometric.append((np.abs(local@u)<=passage['half_length']+cfg.agent_radius)&(np.abs(local@n)<=passage['width']/2))
    inside_any=np.any(geometric,axis=0)if geometric else np.zeros(len(p),bool)
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
        waiting=(request&((direction!=s)|quota)&~inside_any)|(inside&(direction!=s))if s else np.zeros(len(p),bool)
        exit_point=np.asarray(passage['centre'])+s*passage['half_length']*u
        eligible=(np.linalg.norm(p-exit_point,axis=1)<=menu['R_c'])&~geometry_inside&~((inside|request)&(direction==s)) if s else np.zeros(len(p),bool)
        reverse=inside&(direction!=s)if s else np.zeros(len(p),bool)
        turn['previous_inside']=inside.tolist()
        plans.append((u,n,s,exit_point,waiting,eligible,reverse))
        metadata.append(dict(direction=s,admitted=turn['count'],quota_reached=quota,inside=inside.tolist(),geometry_inside=geometry_inside.tolist(),inside_any=inside_any.tolist(),request=request.tolist(),robot_direction=direction.tolist(),waiting=waiting.tolist(),clear_exit=eligible.tolist(),reverse=reverse.tolist(),goal_yield=[False]*len(p),goal_yield_details=[]))
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
    # New goal-yield term follows the inherited global three-term order.
    # Passages and robot IDs retain their frozen order if regions overlap.
    q=menu['c1']
    for passage,record in zip(passages,metadata):
        u=np.asarray(passage['axis']);n=np.array([-u[1],u[0]]);center=np.asarray(passage['centre'])
        ends={s:center+s*passage['half_length']*u for s in(-1,1)}
        for f in np.flatnonzero(~unfinished):
            inside_f=record['geometry_inside'][f]
            nearest=min((-1,1),key=lambda s:(np.linalg.norm(p[f]-ends[s]),s))
            if not inside_f and np.linalg.norm(p[f]-ends[nearest])>menu['R_c']:continue
            blockers=[]
            for r in np.flatnonzero(unfinished):
                hit=forward_distance(paths[r],p[f],menu['R_e'])
                if hit>2*cfg.agent_radius+cfg.agent_collision_margin:continue
                ent=entry(paths[r],passage,cfg.agent_radius,menu['R_e'])
                # A route passing beside the rectangle can still occupy the
                # blocking disk. Its entering end is the nearer end to r.
                direction=ent[1]if ent is not None and ent[1]else-min((-1,1),key=lambda s:(np.linalg.norm(p[r]-ends[s]),s))
                blockers.append(dict(robot=int(r),direction=int(direction),distance=float(hit)))
            if not blockers:continue
            if inside_f:
                sides={r['direction']for r in blockers}
                s=next(iter(sides))if len(sides)==1 else nearest
                vector=s*u
            else:
                s=nearest;d=p[f]-ends[s];parallel=(d@u)*u;perp=(d@n)*n
                if np.linalg.norm(perp)<=1e-12:perp=np.array([s*u[1],-s*u[0]])
                away=parallel+2*perp;norm=np.linalg.norm(away);vector=away/norm if norm>1e-12 else np.zeros(2)
            out[f]*=1-q;out[f]+=q*cfg.max_speed*vector
            record['goal_yield'][f]=True
            record['goal_yield_details'].append(dict(robot=int(f),inside=bool(inside_f),exit_direction=int(s),vector=vector.tolist(),blockers=blockers))
    return out,metadata

def forward_distance(path,point,lookahead):
    """Distance to the forward polyline clipped at exact arc lookahead."""
    path=np.asarray(path,float);point=np.asarray(point);arc=0.;best=float('inf')
    for a,b in zip(path[:-1],path[1:]):
        d=b-a;length=np.linalg.norm(d)
        if length<=1e-14:continue
        available=min(length,max(0.,lookahead-arc));segment=d*(available/length)
        den=segment@segment;t=np.clip((point-a)@segment/den,0.,1.)if den>0 else 0.
        best=min(best,float(np.linalg.norm(point-a-t*segment)))
        arc+=length
        if arc>=lookahead:break
    return best

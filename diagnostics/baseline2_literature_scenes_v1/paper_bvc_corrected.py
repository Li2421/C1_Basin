from experiment import *
import numpy as np
import math,time

def bvc_corrected(c):
    import cvxpy as cp
    s=c['state'];p=np.asarray(s['positions'],float);g=np.asarray(s['goals'],float);v=np.zeros_like(p);n=len(p)
    dt=.2 if s['scene']=='lit_narrow_passage' else .15;K=10 if dt==.2 else 12;H=int(50/dt)
    pos=[p.copy()];vel=[v.copy()];actions=[];targets=g.copy();detour=np.zeros(n,bool);changes=[];term='timeout'
    for step in range(H):
        nextp=[];nextv=[];inputs=[]
        for i in range(n):
            if detour[i] and np.linalg.norm(p[i]-targets[i])<.1:
                targets[i]=g[i];detour[i]=False;changes.append([step,i,'return_to_goal'])
            w=max(1,round(1/dt))
            if not detour[i] and len(pos)>w and np.max(np.linalg.norm(np.asarray(pos[-w-1:])[:,i]-pos[-w-1][i],axis=-1))<.01 and np.linalg.norm(p[i]-g[i])>.08:
                d=g[i]-p[i];targets[i]=p[i]+.5*np.array([d[1],-d[0]])/max(np.linalg.norm(d),1e-30);detour[i]=True;changes.append([step,i,'right_detour'])
            pp=cp.Variable((K+1,2));vv=cp.Variable((K+1,2));uu=cp.Variable((K,2));constraints=[pp[0]==p[i],vv[0]==v[i]]
            constraints += [pp[1:]==pp[:-1]+dt*vv[:-1],vv[1:]==vv[:-1]+dt*uu]
            constraints += [cp.norm(vv,axis=1)<=1,cp.norm(uu,axis=1)<=1.5]
            for j in range(n):
                if i==j:continue
                d=p[i]-p[j];a=d/np.linalg.norm(d);constraints += [(pp[1:]-(p[i]+p[j])/2)@a>=.15]
            problem=cp.Problem(cp.Minimize(30*cp.sum_squares(pp[-1]-targets[i])+cp.sum_squares(uu)),constraints)
            problem.solve(solver='CLARABEL')
            if problem.status not in ('optimal','optimal_inaccurate'):raise RuntimeError('BVC QP '+problem.status)
            nextp.append(pp.value[1]);nextv.append(vv.value[1]);inputs.append(uu.value[0])
        p=np.asarray(nextp);v=np.asarray(nextv);pos.append(p.copy());vel.append(v.copy());actions.append(inputs)
        if np.all(np.linalg.norm(p-g,axis=-1)<.08) and np.all(np.linalg.norm(v,axis=-1)<.1):term='success';break
    ii,jj=np.triu_indices(n,1);hmin=float(np.min(np.linalg.norm(np.asarray(pos)[:,ii]-np.asarray(pos)[:,jj],axis=-1)-.3))
    return dict(success=int(term=='success' and hmin>=0),paper_converged=int(term=='success'),collision=int(hmin<0),timeout=int(term=='timeout'),numerical_failure=0,outcome=term,episode_length=len(actions),minimum_agent_clearance=hmin,detour_events=changes,livelock_certified=False),dict(positions=pos,velocities=vel,actions=actions,goals=g)

def run(c):
    start=time.monotonic()
    try:
        result,arrays=bvc_corrected(c)
    except Exception as e:
        import traceback
        tb=e.__traceback__;local=None
        while tb:
            if tb.tb_frame.f_code.co_name=='bvc_corrected':local=tb.tb_frame.f_locals
            tb=tb.tb_next
        infeasible=bool(local is not None and local.get('problem') is not None and local['problem'].status=='infeasible')
        if local is not None:
            arrays=dict(positions=np.asarray(local['pos']),velocities=np.asarray(local['vel']),actions=np.asarray(local['actions']),goals=local['g'])
        else:arrays=dict(positions=np.asarray([c['state']['positions']]),goals=c['state']['goals'])
        result=dict(success=0,paper_converged=0,collision=0,timeout=0,numerical_failure=0 if infeasible else 1,outcome='optimization_infeasible' if infeasible else 'unknown',episode_length=len(arrays.get('actions',[])),error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()),optimization_infeasible=infeasible)
    result['elapsed_seconds']=time.monotonic()-start
    return result,arrays

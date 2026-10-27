"""Literature-version drivers; never modify or monkey-patch official modules."""
from experiment import *
import numpy as np
import math
import time
from scipy import sparse
from scipy.linalg import solve_continuous_are
from scipy.optimize import linprog
import clarabel

def qpsolve(target,A,lower,outer_A,outer_lower):
    """Hard pair rows and quadratic-slack soft outer-circle rows."""
    n=len(target);m=len(outer_lower)
    matrix=np.block([[-A,np.zeros((len(A),m))],[-outer_A,-np.eye(m)],[np.zeros((m,n)),-np.eye(m)]])
    rhs=np.r_[-lower,-outer_lower,np.zeros(m)]
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_feas=1e-9;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
    def solve(rhs):
        return clarabel.DefaultSolver(sparse.diags(np.r_[np.ones(n),np.ones(m)*1e4],format='csc'),np.r_[-target,np.zeros(m)],sparse.csc_matrix(matrix),rhs,[clarabel.NonnegativeConeT(len(rhs))],settings).solve()
    sol=solve(rhs);infeasible=False
    if str(sol.status) in ('PrimalInfeasible','AlmostPrimalInfeasible'):
        infeasible=True
        lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.c_[-A,-np.ones(len(A))],b_ub=-lower,bounds=[(None,None)]*n+[(0,None)],method='highs')
        if not lp.success:raise RuntimeError('Minimally infeasible restoration LP failed: '+lp.message)
        rhs[:len(A)]+=lp.x[-1]+1e-8
        sol=solve(rhs)
    if str(sol.status) not in ('Solved','AlmostSolved'):raise RuntimeError('QP '+str(sol.status))
    x=np.asarray(sol.x)
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite QP')
    return x[:n],infeasible,float(np.maximum(outer_lower-outer_A@x[:n],0).max(initial=0))

def circle(c):
    s=c['state'];p=np.asarray(s['positions'],float);g=np.asarray(s['goals'],float);v=np.zeros_like(p)
    Aplant=np.block([[np.zeros((2,2)),np.eye(2)],[np.zeros((2,4))]])
    Bplant=np.r_[np.zeros((2,2)),np.eye(2)]
    P=solve_continuous_are(Aplant,Bplant,.2*np.eye(4),np.eye(2));gain=Bplant.T@P
    positions=[p.copy()];velocities=[v.copy()];actions=[];nominal_speeds=[];hmin=np.inf;infeasible_steps=0;softmax=0;term='timeout'
    for step in range(2000):
        u0=-np.concatenate([p-g,v],1)@gain.T
        nominal_speeds.extend(np.linalg.norm(v,axis=-1).tolist())
        pair_A=[];pair_lower=[];ii,jj=np.triu_indices(5,1)
        for i,j in zip(ii,jj):
            d=p[i]-p[j];dv=v[i]-v[j];a=2*(dv@dv)+10*(d@dv)+6*(d@d-16)
            b=np.zeros((5,2));b[i]=2*d;b[j]=-2*d
            pair_A.append(b.ravel());pair_lower.append(-a)
        outer_A=np.zeros((5,10));outer_lower=[]
        for i in range(5):
            outer_A[i,2*i:2*i+2]=-2*p[i]
            outer_lower.append(2*(v[i]@v[i])+10*(p[i]@v[i])-6*(81-p[i]@p[i]))
        pair_A=np.asarray(pair_A);pair_lower=np.asarray(pair_lower);outer_lower=np.asarray(outer_lower)
        if c['method']=='circle_centralized':
            flat,inf,soft=qpsolve(u0.ravel(),pair_A,pair_lower,outer_A,outer_lower);u=flat.reshape(5,2)
        else:
            u=[];inf=False;soft=0.
            factor=1. if c['method']=='circle_df' else .5
            for i in range(5):
                neigh=[j for j in range(5) if j!=i];AA=[];ll=[]
                for j in neigh:
                    d=p[i]-p[j];dv=v[i]-v[j]
                    AA.append(2*d);ll.append(-factor*(2*(dv@dv)+10*(d@dv)+6*(d@d-16)))
                val,inf_i,soft_i=qpsolve(u0[i],np.asarray(AA),np.asarray(ll),outer_A[i:i+1,2*i:2*i+2],outer_lower[i:i+1])
                u.append(val);inf|=inf_i;soft=max(soft,soft_i)
            u=np.asarray(u)
        infeasible_steps+=int(inf);softmax=max(softmax,soft)
        # Exact zero-order-held acceleration, with continuous inter-sample
        # pair minimum from polynomial critical points.
        for i,j in zip(ii,jj):
            d=p[i]-p[j];dv=v[i]-v[j];du=u[i]-u[j]
            coeff=[.5*(du@du),1.5*(dv@du),dv@dv+d@du,d@dv]
            roots=np.roots(np.trim_zeros(coeff,'f')) if any(coeff) else []
            ts=[0.,.05]+[float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<.05]
            hmin=min(hmin,min(np.linalg.norm(d+t*dv+.5*t*t*du)**2-16 for t in ts))
        p=p+.05*v+.5*.05**2*u;v=v+.05*u
        positions.append(p.copy());velocities.append(v.copy());actions.append(u.copy())
        if np.all(np.linalg.norm(p-g,axis=-1)<.1) and np.all(np.linalg.norm(v,axis=-1)<.1):term='success';break
    # A separate unfiltered nominal LQR reference needs no new rollout:
    # solve its fixed linear dynamics analytically on the same sampled states.
    from scipy.linalg import expm
    M=expm((Aplant-Bplant@gain)*.05)
    x=np.concatenate([np.asarray(s['positions'])-g,np.zeros_like(g)],1)
    lqr_speeds=[]
    for _ in range(2001):lqr_speeds.extend(np.linalg.norm(x[:,2:],axis=-1).tolist());x=x@M.T
    return dict(success=int(term=='success' and hmin>=0),paper_converged=int(term=='success'),collision=int(hmin<0),timeout=int(term=='timeout'),numerical_failure=0,outcome=term,episode_length=len(actions),h_min=hmin,infeasible_steps=infeasible_steps,outer_soft_max_violation=softmax,paper_stuck_candidate=int(term=='timeout'),gridlock_certified=False),dict(positions=np.asarray(positions),velocities=np.asarray(velocities),actions=np.asarray(actions),nominal_lqr_speeds=np.asarray(lqr_speeds),goals=g)

def impc(c):
    sys.path.insert(0,str(HERE/'external/IMPC-DR/2D'))
    import SET
    from test import initialize
    from others import get_obstacle_list,check_reach_target
    from run import run_one_step
    s=c['state'];dt=.2 if s['scene']=='lit_narrow_passage' else .15;K=10 if dt==.2 else 12;H=math.ceil(50/dt)
    SET.initialize_set(s['N'],np.asarray(s['positions']),np.zeros((s['N'],2)),np.asarray(s['goals']),.3,.1,dt,K,H)
    agents=initialize();pos=[np.asarray([a.p.copy() for a in agents])];vel=[np.asarray([a.v.copy() for a in agents])];act=[];term='timeout'
    for _ in range(H):
        obstacle=get_obstacle_list(agents,s['N']);agents=run_one_step(agents,obstacle)
        pos.append(np.asarray([a.p.copy() for a in agents]));vel.append(np.asarray([a.v.copy() for a in agents]));act.append(np.asarray([a.u.copy() for a in agents]))
        if check_reach_target(agents):term='success';break
    ii,jj=np.triu_indices(s['N'],1);hmin=float(np.min(np.linalg.norm(np.asarray(pos)[:,ii]-np.asarray(pos)[:,jj],axis=-1)-.3))
    return dict(success=int(term=='success' and hmin>=0),paper_converged=int(term=='success'),collision=int(hmin<0),timeout=int(term=='timeout'),numerical_failure=0,outcome=term,episode_length=len(act),minimum_agent_clearance=hmin,official_r_min=agents[0].r_min,official_terminal_cost_coefficient=15.,official_rho0=.2,official_eta_initial=1.),dict(positions=pos,velocities=vel,actions=act,goals=s['goals'])

def bvc(c):
    import cvxpy as cp
    s=c['state'];p=np.asarray(s['positions'],float);g=np.asarray(s['goals'],float);v=np.zeros_like(p);n=len(p)
    dt=.2 if s['scene']=='lit_narrow_passage' else .15;K=10 if dt==.2 else 12;H=math.ceil(50/dt)
    pos=[p.copy()];vel=[v.copy()];actions=[];targets=g.copy();detour=np.zeros(n,bool);changes=[];term='timeout'
    for step in range(H):
        nextp=[];nextv=[];inputs=[]
        for i in range(n):
            if detour[i] and np.linalg.norm(p[i]-targets[i])<.1:
                targets[i]=g[i];detour[i]=False;changes.append([step,i,'return_to_goal'])
            w=max(1,round(1/dt))
            if not detour[i] and len(pos)>w and np.max(np.linalg.norm(np.asarray(pos[-w-1:])[:,i]-pos[-w-1][i],axis=-1))<.01 and np.linalg.norm(p[i]-g[i])>.08:
                d=g[i]-p[i];targets[i]=p[i]+.5*np.array([d[1],-d[0]])/max(np.linalg.norm(d),1e-30);detour[i]=True;changes.append([step,i,'right_detour'])
            pp=cp.Variable((K+1,2));vv=cp.Variable((K+1,2));uu=cp.Variable((K,2));constraints=[pp[0]==p[i],vv[0]==v[i],vv[-1]==0]
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

def doorway(c):
    sys.path.insert(0,str(HERE/'external/mpc-cbf'))
    import config
    from mpc_cbf import MPC
    initial=np.c_[c['state']['positions'],np.zeros(2)];goals=np.c_[c['state']['goals'],np.zeros(2)]
    xf=np.zeros((2,3));history=np.zeros((100,3));counter=0;pos=[initial[:,:2].copy()];head=[initial.copy()];length=0
    obs=[(1.,y,.1) for y in [.4,.5,.6,.7,.8,.9,1.,-.3,-.3,-.3,-.4,-.5,-.6,-.7,-.8,-.9]]
    highres=[]
    for j in range(50):
        for i in range(2):
            config.x0=initial[i,:];config.goal=goals[i,:];config.obs=list(obs);config.v_limit=.3
            for k in range(2):
                if k!=i:config.obs.append((initial[k,0],initial[k,1],.02))
            liveliness='on' if c['method']=='doorway_on' else 'off'
            controller=MPC(history,j,i,liveliness)
            history[counter,:]=xf[i,:]
            controller.run_simulation(history,j)
            xf[i,:]=controller.run_simulation_to_get_final_condition(history,j,i,liveliness).ravel()
            highres.append(np.asarray(controller.mpc.data['_x']).copy());counter+=1
        initial=xf
        pos.append(xf[:,:2].copy());head.append(xf.copy());length+=3
    p=np.asarray(pos);err=np.linalg.norm(p[-1]-goals[:,:2],axis=-1);pair=np.linalg.norm(p[:,0]-p[:,1],axis=-1)-.2
    wall=np.linalg.norm(p[:,:,None,:]-np.asarray(obs)[None,None,:,:2],axis=-1)-.2
    collision=bool(pair.min()<0 or wall.min()<0);success=bool(np.all(err<.1) and not collision)
    quiet=bool(np.max(np.linalg.norm(p[-min(17,len(p)):]-p[-min(17,len(p))],axis=-1))<.1)
    return dict(success=int(success),paper_converged=int(np.all(err<.1)),collision=int(collision),timeout=int(not success),numerical_failure=0,outcome='success' if success else 'timeout',episode_length=length,stationary_unreached_candidate=quiet and not success,paper_deadlock_certified=False,physical_time_seconds=15.,original_main_outer_loops=50),dict(positions=p,poses=head,goals=goals,optimizer_states=np.asarray(highres))

def run(c):
    t=time.monotonic()
    fn=circle if c['method'].startswith('circle_') else impc if c['method']=='impc_dr' else bvc if c['method']=='buffered_voronoi_right' else doorway
    try:
        result,arrays=fn(c)
    except Exception as e:
        import traceback
        result=dict(success=0,paper_converged=0,collision=0,timeout=0,numerical_failure=1,outcome='unknown',episode_length=0,error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        arrays=dict(positions=np.asarray([c['state']['positions']]),goals=c['state']['goals'])
    result['elapsed_seconds']=time.monotonic()-t
    return result,arrays

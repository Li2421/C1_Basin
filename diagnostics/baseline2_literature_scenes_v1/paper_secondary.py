"""Second-priority paper equations (17)--(19) and official intersection."""
from experiment import *
from paper import qpsolve
import numpy as np
from scipy.linalg import solve_continuous_are
import time

def cooptimization(p,v,u0,method,estimate):
    n=len(p);ii,jj=np.triu_indices(n,1);A=[];lower=[]
    for i,j in zip(ii,jj):
        d=p[i]-p[j];dv=v[i]-v[j];a=2*(dv@dv)+10*(d@dv)+6*(d@d-16)
        row=np.zeros((n,2));row[i]=2*d;row[j]=-2*d;A.append(row.ravel());lower.append(-a)
    A=np.asarray(A);lower=np.asarray(lower);u=[];virtual=[];infeasible=False;soft=0.
    for host in range(n):
        outer=np.zeros((1,2*n));outer[0,2*host:2*host+2]=-2*p[host]
        outerlower=np.asarray([2*(v[host]@v[host])+10*(p[host]@v[host])-6*(81-p[host]@p[host])])
        own=np.zeros_like(p);own[host]=u0[host]
        if method=='circle_ccs':
            # All variables are corrections; only the host adds its nominal.
            rhs=lower-2*(A@own.ravel());target=np.zeros(2*n)
            outerlower=outerlower-outer@own.ravel()
        else:
            # Eq18: actual host u_hh; virtual neighbor u_hj + w_hat_hj.
            rhs=lower-A@estimate[host].ravel();target=own.ravel()
        x,inf,slack=qpsolve(target,A,rhs,outer,outerlower);x=x.reshape(n,2)
        virtual.append(x);u.append(x[host]+(u0[host] if method=='circle_ccs' else 0.));infeasible|=inf;soft=max(soft,slack)
    return np.asarray(u),np.asarray(virtual),infeasible,soft

def circle(c):
    s=c['state'];p=np.asarray(s['positions'],float);g=np.asarray(s['goals'],float);v=np.zeros_like(p);n=len(p)
    Aplant=np.block([[np.zeros((2,2)),np.eye(2)],[np.zeros((2,4))]]);Bplant=np.r_[np.zeros((2,2)),np.eye(2)]
    gain=Bplant.T@solve_continuous_are(Aplant,Bplant,.2*np.eye(4),np.eye(2))
    pos=[p.copy()];vel=[v.copy()];actions=[];estimate=np.zeros((n,n,2));hmin=np.inf;infeasible_steps=0;softmax=0.;term='timeout'
    ii,jj=np.triu_indices(n,1)
    for step in range(2000):
        u0=-np.concatenate([p-g,v],1)@gain.T
        u,virtual,inf,soft=cooptimization(p,v,u0,c['method'],estimate)
        observed=u[None]-virtual;observed[np.arange(n),np.arange(n)]=0
        if c['method']=='circle_pcca_filtered':
            # Exact sampled first-order filter with constant input per period.
            alpha=1-np.exp(-.05/.2);estimate=estimate+alpha*(observed-estimate)
        elif c['method']=='circle_pcca':estimate=observed
        infeasible_steps+=int(inf);softmax=max(softmax,soft)
        for i,j in zip(ii,jj):
            d=p[i]-p[j];dv=v[i]-v[j];du=u[i]-u[j]
            coeff=[.5*(du@du),1.5*(dv@du),dv@dv+d@du,d@dv]
            roots=np.roots(np.trim_zeros(coeff,'f')) if any(coeff) else []
            ts=[0.,.05]+[float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<.05]
            hmin=min(hmin,min(np.linalg.norm(d+t*dv+.5*t*t*du)**2-16 for t in ts))
        p=p+.05*v+.5*.05**2*u;v=v+.05*u;pos.append(p.copy());vel.append(v.copy());actions.append(u.copy())
        if np.all(np.linalg.norm(p-g,axis=-1)<.1) and np.all(np.linalg.norm(v,axis=-1)<.1):term='success';break
    return dict(success=int(term=='success' and hmin>=0),paper_converged=int(term=='success'),collision=int(hmin<0),timeout=int(term=='timeout'),numerical_failure=0,outcome=term,episode_length=len(actions),h_min=hmin,infeasible_steps=infeasible_steps,outer_soft_max_violation=softmax,gridlock_certified=False),dict(positions=pos,velocities=vel,actions=actions,goals=g)

def intersection_inputs():
    # Execute only the official assignment expressions in a fresh namespace.
    # Never import main.py (which runs/plots), or patch existing modules.
    import ast
    from types import SimpleNamespace
    source=HERE/'external/mpc-cbf/main.py';tree=ast.parse(source.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');ns=dict(np=np,scenario='intersection',config=SimpleNamespace())
    branches=[x for x in fn.body if isinstance(x,ast.If)]
    for node in branches[:2]:exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    loop=next(x for x in fn.body if isinstance(x,ast.For));inner=loop.body[0]
    obstacle=next(x for x in inner.body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare) and isinstance(x.test.left,ast.Name) and x.test.left.id=='scenario')
    exec(compile(ast.Module(body=[obstacle],type_ignores=[]),str(source),'exec'),ns)
    return ns['initial'],ns['goals'],ns['config'].obs

def intersection(c):
    sys.path.insert(0,str(HERE/'external/mpc-cbf'));import config
    from mpc_cbf import MPC
    initial,goals,obs=intersection_inputs();assert np.array_equal(initial[:,:2],c['state']['positions'])
    xf=np.zeros((2,3));history=np.zeros((100,3));counter=0;pos=[initial[:,:2].copy()];head=[initial.copy()];highres=[]
    for j in range(50):
        for i in range(2):
            config.x0=initial[i,:];config.goal=goals[i,:];config.obs=list(obs);config.v_limit=.3
            config.obs.append((initial[1-i,0],initial[1-i,1],.02));liveliness='on' if c['method']=='intersection_on' else 'off'
            controller=MPC(history,j,i,liveliness);history[counter,:]=xf[i,:]
            controller.run_simulation(history,j);xf[i,:]=controller.run_simulation_to_get_final_condition(history,j,i,liveliness).ravel()
            highres.append(np.asarray(controller.mpc.data['_x']).copy());counter+=1
        initial=xf;pos.append(xf[:,:2].copy());head.append(xf.copy())
    p=np.asarray(pos);err=np.linalg.norm(p[-1]-goals[:,:2],axis=-1);pair=np.linalg.norm(p[:,0]-p[:,1],axis=-1)-.2
    wall=np.linalg.norm(p[:,:,None,:]-np.asarray(obs)[None,None,:,:2],axis=-1)-.2;collision=bool(pair.min()<0 or wall.min()<0);success=bool(np.all(err<.1) and not collision)
    quiet=bool(np.max(np.linalg.norm(p[-17:]-p[-17],axis=-1))<.1)
    return dict(success=int(success),paper_converged=int(np.all(err<.1)),collision=int(collision),timeout=int(not success),numerical_failure=0,outcome='success' if success else 'timeout',episode_length=150,stationary_unreached_candidate=quiet and not success,paper_deadlock_certified=False,physical_time_seconds=15.),dict(positions=p,poses=head,goals=goals,optimizer_states=highres)

def run(c):
    start=time.monotonic()
    try:result,arrays=(circle(c) if c['method'].startswith('circle_') else intersection(c))
    except Exception as e:
        import traceback
        frames=e.__traceback__;local={}
        while frames:
            if frames.tb_frame.f_code.co_name in ('circle','intersection'):local=frames.tb_frame.f_locals
            frames=frames.tb_next
        positions=np.asarray(local.get('pos',[c['state']['positions']]))
        arrays=dict(positions=positions,goals=c['state']['goals'])
        for key in ('vel','actions'):
            if key in local:arrays['velocities' if key=='vel' else key]=np.asarray(local[key])
        result=dict(success=0,paper_converged=0,collision=0,timeout=0,numerical_failure=1,outcome='unknown',episode_length=len(positions)-1,error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()),partial_trace_retained=True)
    result['elapsed_seconds']=time.monotonic()-start
    return result,arrays

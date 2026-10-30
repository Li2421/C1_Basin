"""Stage-two tracking: independent HOCBF acceleration and discrete unicycle CBF."""
from experiment import *
import numpy as np
import clarabel
from scipy import sparse
from scipy.optimize import minimize
from shared_control.hard_projection import HardProjectionConfig

def solve_soc(target,A,lower,acceleration_bound=None):
    target=np.asarray(target,float);A=np.asarray(A,float).reshape(-1,2);lower=np.asarray(lower,float)
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_feas=1e-9;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
    def solve(relaxation=0.,minimax=False):
        d=3 if minimax else 2
        P=sparse.diags([0.,0.,0.] if minimax else [1.,1.],format='csc')
        q=np.asarray([0.,0.,1.]) if minimax else -target
        linear=np.c_[-A,-np.ones(len(A))] if minimax else -A
        rhs=-lower+relaxation
        if minimax:
            linear=np.vstack([linear,[0.,0.,-1.]]);rhs=np.r_[rhs,0.]
        blocks=[sparse.csc_matrix(linear)];values=[rhs];cones=[clarabel.NonnegativeConeT(len(rhs))]
        if acceleration_bound is not None:
            ball=np.zeros((3,d));ball[1:,:2]=-np.eye(2)
            blocks.append(sparse.csc_matrix(ball));values.append(np.asarray([acceleration_bound,0.,0.]));cones.append(clarabel.SecondOrderConeT(3))
        return clarabel.DefaultSolver(P,q,sparse.vstack(blocks,format='csc'),np.concatenate(values),cones,settings).solve()
    if np.all(A@target>=lower-1e-10) and (acceleration_bound is None or np.linalg.norm(target)<=acceleration_bound+1e-10):
        return target,dict(minimax_relaxation=0.,maximum_violation=0.,restored=False,status='Feasible_unconstrained_minimizer')
    sol=solve();restored=False;relax=0.
    if str(sol.status) in ('PrimalInfeasible','AlmostPrimalInfeasible'):
        first=solve(minimax=True)
        if str(first.status) not in ('Solved','AlmostSolved'):raise RuntimeError('Minimax SOCP '+str(first.status))
        relax=max(0.,float(first.x[-1]));sol=solve(relax+1e-8);restored=True
    if str(sol.status) not in ('Solved','AlmostSolved'):raise RuntimeError('Acceleration SOCP '+str(sol.status))
    x=np.asarray(sol.x[:2]);violation=float(np.maximum(lower-A@x,0).max(initial=0.))
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite acceleration')
    if acceleration_bound is not None and np.linalg.norm(x)>acceleration_bound+1e-6:raise RuntimeError('Acceleration bound certificate failed')
    if violation>relax+2e-6:raise RuntimeError('Linear constraint certificate failed')
    return x,dict(minimax_relaxation=relax,maximum_violation=violation,restored=restored,status=str(sol.status))

def wall_geometry(point,segment):
    a,b=segment;ab=b-a;parameter=float((point-a)@ab/(ab@ab));q=a+np.clip(parameter,0,1)*ab
    delta=point-q;distance=np.linalg.norm(delta)
    if distance<=1e-14:raise ValueError('Undefined wall normal')
    normal=delta/distance
    Hessian=np.zeros((2,2)) if 0<parameter<1 else (np.eye(2)-np.outer(normal,normal))/distance
    return distance,normal,Hessian

def acceleration_problem(p,v,i,c,wall_segments):
    cfg=HardProjectionConfig();circle=c.scene=='lit_circle_five';l0,l1=(6.,5.) if circle else (1.,2.)
    coefficient=1. if c.scene=='lit_doorway' else .5
    safe=2*c.agent_radius+c.agent_collision_margin+cfg.separation_buffer
    rows=[];lower=[];labels=[]
    for j in range(len(p)):
        if i==j:continue
        d=p[i]-p[j];dv=v[i]-v[j];h=d@d-safe*safe
        constant=2*(dv@dv)+2*l1*(d@dv)+l0*h
        rows.append(2*d);lower.append(-coefficient*constant);labels.append(['pair',i,j])
    for k,segment in enumerate(wall_segments):
        distance,normal,H=wall_geometry(p[i],segment)
        h=distance-c.agent_radius-c.wall_radius-c.wall_collision_margin-cfg.separation_buffer
        rows.append(normal);lower.append(-(v[i]@H@v[i]+l1*(normal@v[i])+l0*h));labels.append(['wall',i,k])
    # First-order speed CBF. Sampled speed violations are measured, never clipped.
    rows.append(-2*v[i]);lower.append(-(c.max_speed*c.max_speed-v[i]@v[i]));labels.append(['speed',i])
    return np.asarray(rows),np.asarray(lower),labels

def track_acceleration(desired,velocity,dt,bound):
    target=(desired-velocity)/dt
    if bound is not None:target=target*min(1.,bound/max(np.linalg.norm(target),1e-30))
    return target

LOOKAHEAD=.025

def unicycle_matrix(theta):
    return np.asarray([[np.cos(theta),-LOOKAHEAD*np.sin(theta)],[np.sin(theta),LOOKAHEAD*np.cos(theta)]])

def unicycle_next(position,theta,control,dt):
    v,omega=control
    return position+dt*v*np.asarray([np.cos(theta),np.sin(theta)]),theta+dt*omega

def unicycle_q(position,theta):return position+LOOKAHEAD*np.asarray([np.cos(theta),np.sin(theta)])

def discrete_problem(p,heading,i,c,wall_segments):
    cfg=HardProjectionConfig();inflated=c.agent_radius+LOOKAHEAD
    safe=2*inflated+c.agent_collision_margin+cfg.separation_buffer
    q=np.asarray([unicycle_q(x,t) for x,t in zip(p,heading)])
    old_pair=[(j,float((q[i]-q[j])@(q[i]-q[j])-safe*safe)) for j in range(len(p)) if j!=i]
    old_wall=[wall_geometry(q[i],segment)[0]-inflated-c.wall_radius-c.wall_collision_margin-cfg.separation_buffer for segment in wall_segments]
    labels=[['pair',i,j] for j,_ in old_pair]+[['wall',i,k] for k in range(len(wall_segments))]
    def values(control):
        pn,hn=unicycle_next(p[i],heading[i],control,c.dt);qn=unicycle_q(pn,hn)
        pair=[float((qn-q[j])@(qn-q[j])-safe*safe-.9*h) for j,h in old_pair]
        wall=[wall_geometry(qn,segment)[0]-inflated-c.wall_radius-c.wall_collision_margin-cfg.separation_buffer-.9*h for segment,h in zip(wall_segments,old_wall)]
        return np.asarray(pair+wall)
    def jacobian(control):
        velocity,omega=control;pn,hn=unicycle_next(p[i],heading[i],control,c.dt);qn=unicycle_q(pn,hn)
        J=np.column_stack([c.dt*np.asarray([np.cos(heading[i]),np.sin(heading[i])]),LOOKAHEAD*c.dt*np.asarray([-np.sin(hn),np.cos(hn)])])
        return np.asarray([2*(qn-q[j])@J for j,_ in old_pair]+[wall_geometry(qn,segment)[1]@J for segment in wall_segments])
    return values,jacobian,labels

def project_unicycle(desired,p,heading,i,c,wall_segments):
    T=unicycle_matrix(heading[i]);bounds=((-0.26,.26),(-3.8,3.8))
    nominal=np.clip(np.linalg.solve(T,desired),[-.26,-3.8],[.26,3.8]);target=T@nominal
    values,jacobian,labels=discrete_problem(p,heading,i,c,wall_segments)
    objective=lambda x:.5*float(np.sum((T@x-target)**2))
    gradient=lambda x:T.T@(T@x-target)
    candidates=[];attempts=[]
    for initial in (nominal,np.zeros(2),np.asarray([nominal[0],0.])):
        if np.all(values(initial)>=-1e-10) and objective(initial)<=1e-20:
            candidates.append(initial.copy());attempts.append(dict(status='Feasible_unconstrained_minimizer'));break
        fit=minimize(objective,initial,jac=gradient,method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=values,jac=jacobian)],options=dict(ftol=1e-12,maxiter=1000))
        violation=float(np.maximum(-values(fit.x),0).max(initial=0.))
        attempts.append(dict(success=bool(fit.success),message=str(fit.message),violation=violation))
        if fit.success and np.isfinite(fit.x).all() and violation<=1e-7:candidates.append(fit.x.copy())
    if not candidates:raise RuntimeError('Discrete unicycle same-problem solves failed: '+canonical(attempts))
    control=min(candidates,key=objective);active=np.flatnonzero(values(control)<=1e-7)
    A=jacobian(control)[active]@np.linalg.inv(T)
    return control,dict(attempts=attempts,maximum_violation=float(np.maximum(-values(control),0).max(initial=0.)),
        active_A=A.tolist(),active_labels=[labels[k] for k in active],nominal_front_velocity=target.tolist(),safe_front_velocity=(T@control).tolist(),global_optimality_certified=False)

"""Specified local SI rows; speed-only then safety minimax restoration."""
import bootstrap
from experiment import *
import numpy as np
import clarabel
from scipy import sparse
from distributed_safety import LocalSafety
from shared_control.hard_projection import HardProjectionConfig
from shared_control.hard_projection import CBFSolverError
from fast_projection import project_velocity
import accel_mode
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import tighter_identical_socp,identical_nonlinear_polish

def _local_problem_original(snapshot,i):
    hp=HardProjectionConfig();p=np.asarray(snapshot['positions']);c=snapshot['config'];rows=[];lower=[];labels=[]
    safe=2*c['agent_radius']+c['agent_collision_margin']+hp.separation_buffer
    for j in range(len(p)):
        if j==i:continue
        d=p[i]-p[j]
        if np.linalg.norm(d)>.5:continue
        rows.append(2*d);lower.append(-.5*hp.gamma*(d@d-safe*safe));labels.append(['pair',i,j])
    for k,(a,b) in enumerate(np.asarray(snapshot['walls'])):
        d=p[i]-(a+np.clip((p[i]-a)@(b-a)/((b-a)@(b-a)),0,1)*(b-a));distance=np.linalg.norm(d)
        if distance>.5:continue
        if distance<=1e-14:raise RuntimeError('Undefined static wall normal')
        rows.append(d/distance);lower.append(-hp.gamma_wall*(distance-c['agent_radius']-c['wall_radius']-c['wall_collision_margin']-hp.separation_buffer));labels.append(['wall',i,k])
    if snapshot.get('moving_position') is not None:
        d=p[i]-np.asarray(snapshot['moving_position']);distance=np.linalg.norm(d)
        if distance<=.5:
            if distance<=1e-14:raise RuntimeError('Undefined moving disk normal')
            radius=.3+.0125+c['agent_radius']+hp.separation_buffer
            rows.append(2*d);lower.append(-hp.gamma*(d@d-radius*radius));labels.append(['moving',i,0])
    if not rows:rows=[np.zeros(2)];lower=[-1.];labels=[['tautology']]
    return np.asarray(rows,float),np.asarray(lower,float),labels

def restore(target,A,lower,speed):
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
    def solve(mode,slack=0.,radius=speed):
        dim=3 if mode in ('speed_slack','safety_slack') else 2
        P=sparse.diags([0.,0.,0.] if dim==3 else [1.,1.],format='csc');q=np.array([0.,0.,1.]) if dim==3 else -target
        mat=-A if dim==2 else np.c_[-A,-np.ones(len(A)) if mode=='safety_slack' else np.zeros(len(A))]
        rhs=-lower+slack
        if dim==3:mat=np.vstack([mat,[0.,0.,-1.]]);rhs=np.r_[rhs,0.]
        blocks=[sparse.csc_matrix(mat)];values=[rhs];cones=[clarabel.NonnegativeConeT(len(rhs))]
        if mode not in ('safety_slack','safety_project'):
            ball=np.zeros((3,dim));ball[1:,:2]=-np.eye(2)
            if mode=='speed_slack':ball[0,2]=-1.
            blocks.append(sparse.csc_matrix(ball));values.append(np.array([radius,0.,0.]));cones.append(clarabel.SecondOrderConeT(3))
        return clarabel.DefaultSolver(P,q,sparse.vstack(blocks,format='csc'),np.concatenate(values),cones,settings).solve()
    solved=lambda s:str(s.status) in ('Solved','AlmostSolved')
    sol=solve('original');phase='original';slack=0.;radius=speed
    if str(sol.status) in ('PrimalInfeasible','AlmostPrimalInfeasible'):
        first=solve('speed_slack');phase='speed_only'
        if solved(first):radius=speed+max(0.,float(first.x[-1]))+1e-8;sol=solve('original',radius=radius)
        elif str(first.status) in ('PrimalInfeasible','AlmostPrimalInfeasible'):
            phase='safety_minimax';first=solve('safety_slack')
            if not solved(first):raise RuntimeError('Safety minimax '+str(first.status))
            slack=max(0.,float(first.x[-1]))+1e-8;sol=solve('safety_project',slack=slack)
        else:raise RuntimeError('Speed relaxation '+str(first.status))
    if not solved(sol):raise RuntimeError('SI safety SOCP '+str(sol.status))
    x=np.asarray(sol.x[:2]);effective=lower-slack
    if np.max(effective-A@x,initial=0.)>2e-6 or not np.isfinite(x).all():raise RuntimeError('Effective SI constraint certificate failed')
    return x,dict(restoration_phase=phase,maximum_violation=float(np.maximum(lower-A@x,0.).max(initial=0.)),effective_lower=effective.tolist(),speed_excess=max(0.,float(np.linalg.norm(x)-speed)),speed_ball_radius=radius)

class SwapSafety:
    def __init__(self,snap,i):self.A,self.lower,self.kinds=local_problem(snap,i);self.max_speed=snap['config']['max_speed'];self.config=HardProjectionConfig()
    def project(self,target):
        target=np.asarray(target,float).reshape(1,2)
        # Preserve the original SI solver and its same-problem retries when feasible.
        try:
            try:a,status,d=project_velocity(target,self.A,self.lower,self.max_speed,self.config)
            except CBFSolverError as first:
                try:a,status,d=tighter_identical_socp(target,self.A,self.lower,self.max_speed,self.config)
                except CBFSolverError:a,status,d=identical_nonlinear_polish(target,self.A,self.lower,self.max_speed,self.config,first.details.get('solver_output_candidate',target))
            return a,dict(restoration_phase='original',maximum_violation=float(np.maximum(self.lower-self.A@a.ravel(),0.).max(initial=0.)),effective_lower=self.lower.tolist(),speed_excess=max(0.,float(np.linalg.norm(a)-self.max_speed)))
        except CBFSolverError:
            # Distinguish convex infeasibility from numerical failure in restore.
            x,d=restore(target.ravel(),self.A,self.lower,self.max_speed)
            return x[None],d

def local_problem(snapshot,i):
    walls=np.asarray(snapshot['walls'])
    if not accel_mode.ENABLED or len(walls)==0:return _local_problem_original(snapshot,i)
    p=np.asarray(snapshot['positions']);a=walls[:,0];ab=walls[:,1]-a
    t=np.clip(((p[i]-a)*ab).sum(-1)/(ab*ab).sum(-1),0,1)
    keep=np.flatnonzero(np.linalg.norm(p[i]-(a+t[:,None]*ab),axis=-1)<=.5+1e-9)
    A,lower,labels=_local_problem_original(dict(snapshot,walls=walls[keep]),i)
    return A,lower,[['wall',l[1],int(keep[l[2]])]if l[0]=='wall'else l for l in labels]

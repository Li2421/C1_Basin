"""Equation-level ports of Wang2017 and Yanze Zhang ICRA2025.

These are independent reimplementations, not author code. Constants not specified
in source text are explicit defaults, to be frozen before independent testing.
"""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import expit
import clarabel
J=np.array([[0.,-1.],[1.,0.]])

class NumericalUnknown(RuntimeError):pass

def solve_qp(P,q,A,b,speed=None,velocity_offset=None,dt=1.):
    """Linear inequalities plus optional original radial speed bound."""
    n=len(q);A=np.asarray(A,float).reshape(-1,n);b=np.asarray(b,float)
    cones=[clarabel.NonnegativeConeT(len(b))];blocks=[sparse.csc_matrix(A)];rhs=[b]
    if speed is not None:
        ball=np.zeros((3,n));ball[1:,:2]=-dt*np.eye(2)
        blocks.append(sparse.csc_matrix(ball));rhs.append(np.r_[speed,np.zeros(2)if velocity_offset is None else velocity_offset]);cones.append(clarabel.SecondOrderConeT(3))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
    sol=clarabel.DefaultSolver(sparse.csc_matrix(P),np.asarray(q),sparse.vstack(blocks,format='csc'),np.concatenate(rhs),cones,settings).solve()
    if str(sol.status) not in ('Solved','AlmostSolved'):raise NumericalUnknown(str(sol.status))
    x=np.asarray(sol.x)
    if not np.isfinite(x).all()or np.max(A@x-b,initial=0)>2e-6:raise NumericalUnknown('QP residual')
    return x,str(sol.status)

def closest_segment(p,wall):
    a,b=np.asarray(wall);d=b-a;den=float(d@d)
    t=np.clip((p-a)@d/max(den,1e-30),0.,1.)
    hess=2*(np.eye(2)-np.outer(d,d)/den)if 0<t<1 and den>0 else 2*np.eye(2)
    return a+t*d,hess

class Wang:
    """Section IV decentralized DI certificates and complete Algorithm1 branches."""
    perturb_only=False
    def __init__(self,env,seed,options=None):
        self.alpha=1.;self.gamma=1.;self.kp=1.;self.kv=2.;self.kleft=2.;self.kright=.5;self.kdelta=1.;self.tol=1e-6
        for k,v in (options or {}).items():
            if k not in ('alpha','gamma'):raise ValueError(k)
            setattr(self,k,float(v))
        self.counts={str(i):0 for i in range(4)};self.lp_calls=0;self.calls=0;self.seconds=0.;self.failures=[];self.min_h=np.inf
    def barrier(self,d,v,Ds,total_alpha,share):
        dist=np.linalg.norm(d);gap=dist-Ds
        if gap<=0:raise NumericalUnknown('Wang DI barrier outside differentiable safe-distance domain')
        root=np.sqrt(2*total_alpha*gap);dot=d@v;h=root+dot/dist;self.min_h=min(self.min_h,h)
        cubic=share*self.gamma*h**3*dist
        other=share*(-dot**2/dist**2+v@v+total_alpha*dot/root)
        return -d,cubic+other,cubic
    def classify(self,A,b,u,nominal,velocity):
        if np.linalg.norm(nominal)<=self.tol or np.linalg.norm(u)>self.tol or np.linalg.norm(velocity)>self.tol:return 0
        self.lp_calls+=1
        result=linprog([0.,0.,1.],A_ub=np.c_[A,-np.ones(len(A))],b_ub=b,bounds=[(-self.alpha,self.alpha)]*2+[(None,None)],method='highs')
        if not result.success:raise NumericalUnknown('Wang width LP: '+result.message)
        # Prose in Section VII uses width<0; Algorithm1 printed sign is reversed.
        if result.fun>=-self.tol:return 3
        active=np.abs(A@u-b)<=self.tol;rank=np.linalg.matrix_rank(A[active],tol=self.tol)if active.any()else 0
        return 1 if rank==2 else 2 if rank==1 else 3
    def action(self,env,t):
        out=[];cfg=env.config
        for i,p in enumerate(env.positions):
            start=time.monotonic();v=env.velocities[i];nom=self.kp*(env.goals[i]-p)-self.kv*v
            if np.linalg.norm(env.goals[i]-p)<=cfg.goal_tolerance:nom=-self.kv*v
            nom=np.clip(nom,-self.alpha,self.alpha);A=[];b=[];cubic=[];relative=[]
            for j,pj in enumerate(env.positions):
                if i==j:continue
                row,bi,cu=self.barrier(p-pj,v-env.velocities[j],2*cfg.agent_radius+cfg.agent_collision_margin,2*self.alpha,.5)
                A.append(row);b.append(bi);cubic.append(cu);relative.append(pj-p)
            for wall in env.walls:
                q,_=closest_segment(p,wall);row,bi,cu=self.barrier(p-q,v,cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin,self.alpha,1.)
                A.append(row);b.append(bi);cubic.append(cu);relative.append(q-p)
            A=np.asarray(A).reshape(-1,2);b=np.asarray(b);cubic=np.asarray(cubic)
            def qp(target,rhs):
                self.calls+=1
                try:return solve_qp(np.eye(2),-target,np.r_[A,np.eye(2),-np.eye(2)],np.r_[rhs,[self.alpha]*4],cfg.max_speed,v,cfg.dt)[0]
                except Exception as exc:self.failures.append(dict(step=t,agent=i,error=str(exc)));raise
            u=qp(nom,b);typ=self.classify(A,b,u,nom,v);self.counts[str(typ)]+=1
            if typ==1 and not self.perturb_only:
                factors=np.array([self.kleft if np.cross(nom,r)>0 else self.kright for r in relative])
                u=qp(nom,b+(factors-1)*cubic) # Change only gamma term, not whole RHS.
            elif typ==2:u=qp(nom+self.kdelta*J@nom,b)
            out.append(v+cfg.dt*u);self.seconds+=time.monotonic()-start
        return np.asarray(out)
    def stats(self):return dict(solver=dict(calls=self.calls,seconds=self.seconds,failures=self.failures),classification_counts=self.counts,lp_calls=self.lp_calls,min_DI_barrier=self.min_h,parameters=dict(alpha=self.alpha,gamma=self.gamma,kp=self.kp,kv=self.kv,kleft=self.kleft,kright=self.kright,kdelta=self.kdelta,zero_tolerance=self.tol),port='Independent full Algorithm1 mechanism implementation with original DI barrier equations7/12; closest-wall adaptation; internal acceleration integrated to SI velocity, common radial next-velocity bound added. Finite-step port, not native continuous-time guarantee. Width sign follows prose, not conflicting Algorithm1 print.')

class WangCP(Wang):perturb_only=True

def risk_aux(x,goal,theta,center,radius,hess_h=None,eps=1e-5,scale=.25):
    """Exact two-dimensional SI specialization of Eqs11/13, with derivatives."""
    Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    v=2*(x-Q.T@goal);vtheta=2*J@Q.T@goal;d=x-center;w=2*d
    h=d@d-radius**2;cross=v[0]*w[1]-v[1]*w[0];D=.5*cross**2
    H=2*np.eye(2)if hess_h is None else hess_h
    dc=-2*J@w+H.T@J@v;dct=vtheta[0]*w[1]-vtheta[1]*w[0]
    psi=1/(1+(h/scale)**2);psip=-2*h/scale**2*psi**2
    value=psi*(D-eps);grad=psi*cross*dc+psip*(D-eps)*w;theta_grad=psi*cross*dct
    return value,grad,theta_grad,D,h,w

class RiskAdaptive:
    """ICRA2025 Eqs9–20; explicit fixed parameters, no success-label tuning."""
    def __init__(self,env,seed,options=None):
        self.theta=np.zeros(env.num_agents);self.calls=0;self.seconds=0.;self.failures=[];self.recovery=0;self.zetas=[]
        self.params=dict(p=100.,q=1.,alpha=1.,gamma=1.,beta=1.,phi=10.,c=9.75,t=10.,epsilon=1e-5,psi_scale=.25,omega_recovery=.1)
        for k,v in (options or {}).items():
            if k not in self.params:raise ValueError(k)
            self.params[k]=float(v)
    def action(self,env,t):
        prm=self.params;out=[];nexttheta=self.theta.copy();cfg=env.config
        for i,x in enumerate(env.positions):
            before=time.monotonic();goal=env.goals[i];th=self.theta[i];Q=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            error=x-goal;rotated=Q@x-goal;V=error@error;Vq=rotated@rotated;grad=2*error;gradq=2*Q.T@rotated;gtheta=gradq@J@x
            risk=[];constraints=[]
            for j,c in enumerate(env.positions):
                if j==i:continue
                r=2*cfg.agent_radius+cfg.agent_collision_margin;h=(x-c)@(x-c)-r*r;hdot=2*(x-c)@(env.velocities[i]-env.velocities[j]);risk.append(-hdot-prm['alpha']*h)
                constraints.append((c,r,.5,None))
            Ri=(np.mean(risk)if risk else 0.)+prm['phi'];z=float(expit(prm['t']*(Ri-prm['c'])));self.zetas.append(z)
            for wall in env.walls:
                c,H=closest_segment(x,wall);constraints.append((c,cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin,1.,H))
            A=[np.r_[(1-z)*grad+z*gradq,z*gtheta,-1.]];b=[-prm['gamma']*((1-z)*V+z*Vq)];Ds=[]
            for center,radius,responsibility,H in constraints:
                val,gh,gth,D,h,w=risk_aux(x,goal,th,center,radius,H,prm['epsilon'],prm['psi_scale']);Ds.append(D)
                A.append(np.r_[-w,0.,0.]);b.append(responsibility*prm['alpha']*h)
                A.append(np.r_[-z*gh,-z*gth,0.]);b.append(z*prm['beta']*val)
            self.calls+=1
            try:solution,status=solve_qp(np.diag([2.,2.,2*prm['q'],2*prm['p']]),np.zeros(4),A,b,cfg.max_speed)
            except NumericalUnknown as exc:
                # Remark4 only prescribes recovery when D=0, not every infeasible solve.
                if any(D<=1e-14 for D in Ds):
                    solution=np.array([0.,0.,prm['omega_recovery'],0.]);self.recovery+=1
                    self.failures.append(dict(step=t,agent=i,error=str(exc),paper_remark4_recovery=True))
                else:self.failures.append(dict(step=t,agent=i,error=str(exc),paper_remark4_recovery=False));raise
            out.append(solution[:2]);nexttheta[i]+=cfg.dt*solution[2];self.seconds+=time.monotonic()-before
        self.theta=nexttheta;return np.asarray(out)
    def stats(self):return dict(solver=dict(calls=self.calls,seconds=self.seconds,failures=self.failures),remark4_recoveries=self.recovery,zeta_mean=float(np.mean(self.zetas))if self.zetas else None,parameters=self.params,port='Independent equation-level implementation; original global V(Qx) rotation retained. Parameter defaults explicit, not claimed as author settings. Rational psi satisfies stated properties for h>=0. Walls generalized by nearest-segment h and exact local Hessian; original plant radial speed bound added. Mathematical guarantees not transferred.')

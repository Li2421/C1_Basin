"""JAX interface to the frozen joint score; exact forward, implicit KKT VJP.

No smoothing is added to witness selection or event masks. Derivatives are
piecewise derivatives on the selected constraint/witness branches. Host QP
calls are enclosed in custom VJPs, not detached from the objective.
"""
import numpy as np
from scipy.optimize import nnls
import jax
import jax.numpy as jnp
from scripts.audit_c1_joint_witness_risk import Projection,PROTOCOL
from single_integrator.cbf import project_velocity,CBFConfig

MAXR=1+PROTOCOL['eta']*PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']+1)/PROTOCOL['tau'])

def _solve(y,A,b,physical,controller):
    if controller:return project_velocity(y,A,b,.5,CBFConfig())[0].reshape(-1)
    return Projection(A,b,np.full(len(y)//2,.5) if physical else None)(y)

def _adjoint(y,A,b,p,cot,physical):
    slack=A@p-b;J=A.copy()
    if physical:
        balls=np.zeros((len(p)//2,len(p)))
        for i in range(len(balls)):balls[i,2*i:2*i+2]=-p[2*i:2*i+2]
        J=np.vstack([J,balls]);slack=np.r_[slack,(.25-np.sum(p.reshape(-1,2)**2,axis=1))/2]
    active=slack<1e-8;Ja=J[active]
    lam=nnls(Ja.T,p-y,maxiter=10000)[0] if len(Ja) else np.empty(0)
    H=np.eye(len(p));full=np.zeros(len(J));full[active]=lam
    if physical:
        for i,k in enumerate(full[len(b):]):H[2*i:2*i+2,2*i:2*i+2]+=k*np.eye(2)
    K=np.block([[H,-Ja.T],[Ja,np.zeros((len(Ja),len(Ja)))]])
    adj=np.linalg.lstsq(K.T,np.r_[cot,np.zeros(len(Ja))],rcond=1e-12)[0]
    dy=adj[:len(p)];db_full=np.zeros(len(J));db_full[active]=adj[len(p):]
    dA=full[:len(b),None]*dy[None,:]-db_full[:len(b),None]*p[None,:]
    return dy,dA,db_full[:len(b)]

def make_projection(physical=True,controller=False):
    def host(y,A,b):
        shape=y.shape;ys=y.reshape(-1,shape[-1]);As=A.reshape(len(ys),A.shape[-2],A.shape[-1]);bs=b.reshape(len(ys),-1)
        return np.stack([_solve(v,a,c,physical,controller) for v,a,c in zip(ys,As,bs)]).reshape(shape)
    @jax.custom_vjp
    def project(y,A,b):
        return jax.pure_callback(host,jax.ShapeDtypeStruct(y.shape,y.dtype),y,A,b)
    def fwd(y,A,b):
        p=project(y,A,b);return p,(y,A,b,p)
    def bwd(saved,cot):
        y,A,b,p=saved
        def backward(y,A,b,p,cot):
            shape=y.shape;m=shape[-1];ys=y.reshape(-1,m);As=A.reshape(len(ys),-1,m);bs=b.reshape(len(ys),-1)
            grads=[_adjoint(v,a,c,w,g,physical) for v,a,c,w,g in zip(ys,As,bs,p.reshape(-1,m),cot.reshape(-1,m))]
            return tuple(np.stack([r[i] for r in grads]).reshape(s) for i,s in enumerate([y.shape,A.shape,b.shape]))
        spec=tuple(jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (y,A,b))
        return jax.pure_callback(backward,spec,y,A,b,p,cot)
    project.defvjp(fwd,bwd);return project

physical_projection=make_projection();base_projection=make_projection(False)
controller_projection=make_projection(True,True)

def _zero_certificate(y,A,b):
    def host(y,A,b):
        m=y.shape[-1];out=[]
        for v,a,c in zip(y.reshape(-1,m),A.reshape(-1,A.shape[-2],m),b.reshape(-1,b.shape[-1])):
            boundary=a[c==0]
            out.append(bool(len(boundary)) and nnls(boundary.T,-v,maxiter=10000)[1]<=1e-10*max(1.,np.linalg.norm(v)))
        return np.asarray(out,bool).reshape(y.shape[:-1])
    return jax.pure_callback(host,jax.ShapeDtypeStruct(y.shape[:-1],jnp.bool_),
        *jax.tree_util.tree_map(jax.lax.stop_gradient,(y,A,b)))

def geometry(v,A,b):
    """Batched g, identical coordinate witnesses, thresholds and zero certificate."""
    norm=jnp.sqrt(jnp.sum(v*v,axis=-1));q=v/jnp.maximum(norm[...,None],1e-300)
    p=base_projection(q,A,b/.5)
    B=jnp.where(_zero_certificate(q,A,b),1.,jnp.sum((q-p)**2,axis=-1))
    m=v.shape[-1];axes=jnp.reshape(jnp.stack([-jnp.eye(m),jnp.eye(m)],axis=1),(2*m,m))*.5
    blocknorm=jnp.sqrt(jnp.maximum(jnp.sum(q.reshape(*q.shape[:-1],-1,2)**2,axis=-1),1e-300))
    c=jnp.min(.5/jnp.maximum(blocknorm,1e-300),axis=-1)
    targets=jnp.concatenate([jnp.broadcast_to(axes,(*v.shape[:-1],2*m,m)),(c[...,None]*q)[...,None,:]],axis=-2)
    aa=jnp.broadcast_to(A[...,None,:,:],(*targets.shape[:-1],*A.shape[-2:]));bb=jnp.broadcast_to(b[...,None,:],(*targets.shape[:-1],b.shape[-1]))
    w=physical_projection(targets,aa,bb);wn=jnp.sqrt(jnp.maximum(jnp.sum(w*w,axis=-1),1e-300))
    valid=(wn>1e-9)&~_zero_certificate(targets,aa,bb)
    cosine=jnp.sum(w*q[...,None,:],axis=-1)/jnp.maximum(wn,1e-300)
    M=jnp.max(jnp.where(valid,cosine,-jnp.inf),axis=-1)
    E=.05*jax.nn.softplus((.2-M)/.05)
    defined=(norm>1e-12)&jnp.any(valid,axis=-1)&jnp.all(b<=0,axis=-1)
    return jnp.where(defined,(B+.25*E)/MAXR,jnp.nan)

def trajectory(before,after,candidate,g,goals,success):
    """500 steps0..25s, g200 steps5..15s; exact original P/Gtwo scalar."""
    initial=jnp.sum(jnp.linalg.norm(before[0]-goals,axis=-1))+1e-5
    dist=jnp.linalg.norm(after-goals,axis=-1);V=jnp.concatenate([jnp.array([initial-1e-5]),dist.sum(-1)])/initial
    alive=jnp.concatenate([jnp.array([True]),~jnp.maximum.accumulate(success)[:-1]])
    U=1-jnp.prod(1-jax.nn.sigmoid((dist-.08)/.02),axis=-1)
    end=jnp.arange(1,501);start=jnp.maximum(end-80,0)
    rate=(V[start]-V[end])/((end-start)*.05)
    P=U*jax.nn.sigmoid((.005-rate)/.00125)*alive
    d=jnp.linalg.norm(before[100:300]-goals,axis=-1);u=1-jnp.prod(1-jax.nn.sigmoid((d-.08)/.02),axis=-1)
    ds=dist[299].sum()/initial;dl=dist[499].sum()/initial
    cl=jax.nn.sigmoid((.005-(ds-dl)/10)/.00125)
    cs=jax.nn.sigmoid((.005-(d.sum(-1)/initial-ds)/((300-jnp.arange(100,300))*.05))/.00125)
    raw=.05**2/(.05**2+jnp.sum((candidate[100:300]/.5)**2,axis=-1))
    S=u*raw*(cs*cl+(1-cs)*cl**2);mask=alive[100:300]
    gg=jnp.where(mask,g,0.);ss=jnp.where(mask,S,0.)
    pp=jnp.mean(P[100:300]);return jnp.stack([pp,jnp.mean(gg),jnp.mean(ss),pp+jnp.mean(gg+ss-gg*ss)])

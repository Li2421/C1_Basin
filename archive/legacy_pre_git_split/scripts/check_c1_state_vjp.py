"""Check geometry state derivatives and the largest rollout FD discrepancy."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_joint_bptt import ROOT,OUT,setup,noise_for
from single_integrator.c1.risk.joint_frozen import geometry
from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.joint_frozen_rollout import rollout
from single_integrator.environment import GiveWayEnv

params,field,plant,cbf=setup();env=GiveWayEnv(plant);rows=[]
for rid in [100000,100003,100034,100058]:
    z=np.load(ROOT/f'results/c1_frozen_unseen_64/traces/{rid}/zero.npz');x=jnp.asarray(z['positions_before'][100]);v=jnp.asarray(z['candidate'][100])
    def fn(x):
        A,b,_=barrier_constraints(x,jnp.asarray(env.walls),plant.to_dict(),cbf)
        return geometry(v[None],A[None],b[None])[0]
    grad=jax.grad(fn)(x);d=np.random.default_rng(rid).normal(size=(2,2));d/=np.linalg.norm(d);ad=float(jnp.sum(grad*d));fd=float((fn(x+1e-6*d)-fn(x-1e-6*d))/2e-6)
    rows.append(dict(rid=rid,ad=ad,fd=fd,relative_error=abs(ad-fd)/max(1.,abs(ad),abs(fd))))
rid=100003;z=np.load(ROOT/f'results/c1_frozen_unseen_64/traces/{rid}/zero.npz');noise=noise_for(rid)
fn=jax.jit(lambda p:rollout(p,field,jnp.asarray(z['positions_before'][0]),noise,plant,cbf)[0][1])
value,grad=jax.value_and_grad(fn)(params);norm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(grad))));direction=jax.tree_util.tree_map(lambda g:g/norm,grad);ref=[]
for h in [1e-4,3e-5,1e-5,3e-6,1e-6]:
    values=[float(fn(jax.tree_util.tree_map(lambda p,d:p+sign*h*d,params,direction))) for sign in [-1,1]]
    fd=(values[1]-values[0])/(2*h);ref.append(dict(h=h,fd=fd,ad=norm,relative_error=abs(fd-norm)/max(1.,abs(fd),norm)));print(ref[-1],flush=True)
shifted=jax.tree_util.tree_map(lambda p,d:p-1e-4*d,params,direction)
_,grad2=jax.value_and_grad(fn)(shifted);norm2=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(grad2))));direction2=jax.tree_util.tree_map(lambda g:g/norm2,grad2);off=[]
for h in [1e-5,3e-6,1e-6]:
    values=[float(fn(jax.tree_util.tree_map(lambda p,d:p+sign*h*d,shifted,direction2))) for sign in [-1,1]]
    fd=(values[1]-values[0])/(2*h);off.append(dict(h=h,fd=fd,ad=norm2,relative_error=abs(fd-norm2)/max(1.,abs(fd),norm2)));print('nonzero residual',off[-1],flush=True)
(OUT/'state_and_refined_fd.json').write_text(json.dumps(dict(state_gradients=rows,rollout_refinement=ref,nonzero_residual_refinement=off),indent=2)+'\n')

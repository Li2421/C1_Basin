"""Toy frozen Flow: analytic/JVP controller-output-bias derivative vs independent FD.

The scalar parameter adds a fixed vector to the final vector-field output in
normalized action coordinates. This is a local last-layer bias direction,
not an interpolation between different pretrained checkpoints. No env.step.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
ROOT=Path(__file__).resolve().parents[2]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
for p in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts')):sys.path.insert(0,p)

import jax
import jax.numpy as jnp
import numpy as np
from diagnostics.part2_closed_loop_sensitivity_validation_20261012.derivative_validation import projection_derivative
from diagnostics.placement_boundary_mechanism_audit_v1.closed_loop_probe import key_for
from validation_stage2.run_rollouts import make_runtime
from validation_stage2.factorial_controller import action


def biased(rt, theta, direction):
    """Clone the runtime without changing frozen checkpoint or evaluator."""
    b=copy.copy(rt)
    original=rt.velocity_fn
    b.velocity_fn=lambda obs,z,s: original(obs,z,s)+theta*jnp.asarray(direction)
    def nominal(env,key,traced=False):
        obs=jnp.asarray(b.observation(env)); z=jax.random.normal(key,(rt.agent.config['joint_act_dim'],))
        for k in range(rt.steps):
            z=z+b.velocity_fn(obs,z,jnp.asarray(k/rt.steps))/rt.steps
        raw=np.clip(rt.mean+rt.scale*np.asarray(z).reshape(rt.mean.shape),-1,1)
        raw*=np.minimum(1.,rt.config.max_speed/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-30))
        return obs,raw,None
    b.nominal=nominal
    return b


def native_tangent(rt,env,key,direction):
    obs=jnp.asarray(rt.observation(env));z0=jax.random.normal(key,(rt.agent.config['joint_act_dim'],))
    direction=jnp.asarray(direction)
    def endpoint(theta):
        z=z0
        for k in range(rt.steps):
            z=z+(rt.velocity_fn(obs,z,jnp.asarray(k/rt.steps))+theta*direction)/rt.steps
        a=jnp.clip(jnp.asarray(rt.mean).ravel()+jnp.asarray(rt.scale).ravel()*z,-1.,1.).reshape(rt.mean.shape)
        norms=jnp.linalg.norm(a,axis=1,keepdims=True)
        return (a*jnp.minimum(1.,rt.config.max_speed/jnp.maximum(norms,1e-30))).ravel()
    y,dy=jax.jvp(endpoint,(jnp.asarray(0.),),(jnp.asarray(1.),))
    return np.asarray(y),np.asarray(dy)


def integrate_tangent(rt,env,key,delta,ddelta,safety,project):
    obs=jnp.asarray(rt.observation(env))
    z0=np.asarray(jax.random.normal(key,(rt.agent.config['joint_act_dim'],))).reshape(rt.mean.shape)
    h=(rt.mean+rt.scale*z0).ravel().copy(); dh=np.zeros_like(h);records=[]
    scale=rt.scale.ravel();mean=rt.mean.ravel();direction=DIRECTION
    for k in range(rt.steps):
        s=k/rt.steps; ds=1/rt.steps;remain=1-s
        z=(h-mean)/scale
        fz,dfz=jax.jvp(lambda zz:rt.velocity_fn(obs,zz,jnp.asarray(s)),
                         (jnp.asarray(z),),(jnp.asarray(dh/scale),))
        f=scale*np.asarray(fz)+delta
        df=scale*(np.asarray(dfz)+direction)+ddelta
        if project:
            q=h+remain*f
            out,status=safety.project_field_target(q.reshape(rt.mean.shape))
            p,diag=projection_derivative(safety,q.reshape(rt.mean.shape),out)
            rho=min(1.,ds/remain)
            dh=(1-rho)*dh+rho*(p@(dh+remain*df))
            h=(1-rho)*h+rho*out.ravel()
            records.append(dict(weak=diag['weak'],kkt_residual=diag['kkt_residual'],status=status.get('status','')))
        else:
            h=h+ds*f;dh=dh+ds*df
    return h,dh,records


def main():
    global DIRECTION
    jax.config.update('jax_enable_x64',True)
    st=next(s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states'] if s['uid']=='hard_confirm_toy_1_820013')
    rt=make_runtime('toy_give_way');env=rt.make_env(st)
    root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    eta=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'][4],float)
    key=key_for(st['uid'],0,0,root)
    DIRECTION=np.array([.5,-.2,-.3,.4]);DIRECTION/=np.linalg.norm(DIRECTION)
    obs,safety,ref,safe,delta,_=rt.prepare(env,key,eta)
    reconstructed,dr=native_tangent(rt,env,key,DIRECTION)
    nominal_error=float(np.max(np.abs(ref.ravel()-reconstructed)))
    if nominal_error>2e-6:raise RuntimeError(f'native reconstruction mismatch {nominal_error}')
    p0,diag0=projection_derivative(safety,ref,safe)
    dsafe=p0@dr
    goal=rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values[0]
    eps=1e-6*rt.config.max_speed
    projection=(goal*dsafe.reshape(goal.shape)).sum(-1,keepdims=True)/(np.sum(goal*goal,axis=-1,keepdims=True)+eps*eps)*goal
    ddelta=(rt.basis.metadata.scale*eta[1]*(dsafe.reshape(goal.shape)-projection)).ravel()
    # Check basis differential independently; it is the controller->anchor->B route.
    rows=[]
    zero=np.zeros_like(delta.ravel())
    ft_base,dft_base,ft_records=integrate_tangent(rt,env,key,zero,zero,safety,True)
    tf_base,dtf_base,_=integrate_tangent(rt,env,key,delta.ravel(),ddelta,safety,False)
    ff_base,dff_base,ff_records=integrate_tangent(rt,env,key,delta.ravel(),ddelta,safety,True)
    for ch in ('TT','TF','FT','FF'):
        if ch=='TT':
            target=safe.ravel()+delta.ravel();out=safety.project(target.reshape(safe.shape))[0]
            p,diag=projection_derivative(safety,target.reshape(safe.shape),out)
            tangent=p@(dsafe+ddelta);records=[]
        elif ch=='TF':
            target=tf_base;out=safety.project(target.reshape(safe.shape))[0]
            p,diag=projection_derivative(safety,target.reshape(safe.shape),out)
            tangent=p@dtf_base;records=[]
        elif ch=='FT':
            target=ft_base+delta.ravel();out=safety.project(target.reshape(safe.shape))[0]
            p,diag=projection_derivative(safety,target.reshape(safe.shape),out)
            tangent=p@(dft_base+ddelta);records=ft_records
        else:
            tangent=dff_base;diag={'weak':any(r['weak'] for r in ff_records),'kkt_residual':max(r['kkt_residual'] for r in ff_records)};records=ff_records
        for step in (1e-3,3e-4):
            plus=action(biased(rt,+step,DIRECTION),env,key,eta,ch)[0].ravel()
            minus=action(biased(rt,-step,DIRECTION),env,key,eta,ch)[0].ravel()
            fd=(plus-minus)/(2*step)
            rows.append(dict(placement=ch,epsilon=step,tangent_norm=float(np.linalg.norm(tangent)),
                abs_error=float(np.linalg.norm(tangent-fd)),
                rel_error=float(np.linalg.norm(tangent-fd)/max(np.linalg.norm(fd),1e-12)),
                weak=bool(diag['weak']),kkt_residual=float(diag['kkt_residual']),
                internal_weak_steps=int(sum(r['weak'] for r in records)),native_replay_error=nominal_error,
                anchor_basis_term_norm=float(np.linalg.norm(ddelta)),
                native_anchor_term_norm=float(np.linalg.norm(dsafe))))
    out=Path(__file__).with_name('h_bias_results.json')
    out.write_text(json.dumps(dict(state_uid=st['uid'],eta=eta.tolist(),direction=DIRECTION.tolist(),rows=rows,
        method='JAX JVP of native 10-step Flow and all field responses; local KKT projector derivative; independent end-to-end centered FD',
        new_rollouts=0),indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':main()

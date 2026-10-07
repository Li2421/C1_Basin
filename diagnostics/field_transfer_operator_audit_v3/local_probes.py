"""Bounded, fixed-noise action probes and variational-equation verification.

No environment.step, new trajectory, new outcome label, or controller edit.
"""
from __future__ import annotations
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys
from pathlib import Path
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
for p in ['/home/zhihan/research',str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts')]:sys.path.insert(0,p)
import json
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import nnls
from validation_stage2.run_rollouts import make_runtime
from validation_stage2.factorial_controller import action
from cache_analysis import HERE,PARENT,CACHE,read,save,stats,rel,sha

def projection_derivative(safety,target,output):
    """Exact smooth KKT derivative with numerically inferred strict active set.

    For balls use c_i=(||u_i||²-v²)/2, so the Hessian contribution is lambda_i I.
    Weak/degenerate active sets are diagnosed, never declared differentiable.
    """
    target=target.ravel();u=output.ravel();d=len(u)
    slack=safety.A@u-safety.lower
    sp=np.linalg.norm(u.reshape(-1,2),axis=1)
    linear=np.flatnonzero(slack<1e-7);speed=np.flatnonzero(safety.max_speed-sp<1e-7)
    normals=[-safety.A[k] for k in linear]
    for i in speed:
        v=np.zeros(d);v[2*i:2*i+2]=u[2*i:2*i+2];normals.append(v)
    if not normals:return np.eye(d),dict(active_linear=0,active_speed=0,weak=False,kkt_residual=float(np.linalg.norm(target-u)))
    c=np.array(normals)
    lam,_=nnls(c.T,target-u,maxiter=10000)
    hh=np.ones(d)
    for k,i in enumerate(speed):hh[2*i:2*i+2]+=lam[len(linear)+k]
    hi=np.diag(1/hh)
    dp=hi-hi@c.T@np.linalg.pinv(c@hi@c.T,rcond=1e-10)@c@hi
    return dp,dict(active_linear=len(linear),active_speed=len(speed),weak=bool((lam<1e-7).any()),
                   kkt_residual=float(np.linalg.norm(c.T@lam-(target-u))),multipliers=lam.tolist())

def recurrence(rt,env,obs,key,delta,safety,grad):
    raw,tr=rt.integrate(env,obs,key,delta,safety,project=False,traced=True)
    ff,tf=rt.integrate(env,obs,key,delta,safety,project=True,traced=True)
    d=delta.size;eye=np.eye(d)
    transform=np.stack([rt.to_world((rt.scale.ravel()*v).reshape(delta.shape),env).ravel() for v in eye],axis=1)
    inv=np.linalg.inv(transform)
    results=[]
    for projected,trace in [(False,tr),(True,tf)]:
        ss=np.zeros((d,d));records=[];last_pre=None;last_dp=None
        for k in range(rt.steps):
            s=k/rt.steps;r=1-s;ds=1/rt.steps;h=trace['path'][k]
            z=(rt.to_local(h,env)-rt.mean)/rt.scale
            df=transform@np.asarray(grad(obs,jnp.asarray(z),jnp.asarray(s))).reshape(d,d)@inv
            if not projected:ss=(eye+ds*df)@ss+ds*eye
            else:
                f=np.asarray(rt.velocity_fn(obs,jnp.asarray(z),jnp.asarray(s))).reshape(delta.shape)
                target=h+r*(rt.to_world(rt.scale*f,env)+delta)
                dp,diagnostic=projection_derivative(safety,target,trace['targets'][k])
                qq=(eye+r*df)@ss+r*eye
                rho=min(1.,ds/r);ss=(1-rho)*ss+rho*dp@qq
                last_pre=qq;last_dp=dp
                records.append(dict(k=k,**diagnostic,dp_singular_values=np.linalg.svd(dp,compute_uv=False).tolist()))
        results.append(dict(M=ss,steps=records,last_pre=last_pre,last_dp=last_dp))
    return raw,ff,results

def branch_signature(safety,trace):
    return np.stack([np.r_[safety.A@q.ravel()-safety.lower<=1e-7,
                           safety.max_speed-np.linalg.norm(q,axis=1)<=1e-7] for q in trace['targets']])

def run():
    cfg=read(PARENT/'screening_manifest.json');etas=np.array(cfg['eta'])
    states=read(PARENT/'confirmation_manifest.json')['states']
    control=read(CACHE/'state_dependence_control/manifest.json')
    levels=[.01,.05,.1,.25,.5,1.]
    selected=[0,4,10]
    save('probe_protocol.json',dict(states=[s['uid'] for s in states],candidate_indices=selected,
        levels=levels,centers='delta=0',fixed_keys=control['actual_keys'],new_full_rollouts=0,
        purpose='Test finite-amplitude validity of zero-centered local transfer and analytic variational recurrence.',
        extra_scenarios='First two existing true-t0 states each from DB and Four in frozen stage2 pool.',source_sha256=sha(__file__)))
    records=[];scale_rows=[];errors=[];operators={};replay=[]
    for sc in ['toy_give_way','ring_exchange']:
        rt=make_runtime(sc);grad=jax.jit(jax.jacrev(rt.velocity_fn,argnums=1))
        for state in [s for s in states if s['scenario']==sc]:
            uid=state['uid'];env=rt.make_env(state);key=jnp.array(control['actual_keys'][sc],dtype=jnp.uint32)
            obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
            basis=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,axis=-1).reshape(-1,3)
            z=np.load(CACHE/'state_dependence_control'/f'{uid}.npz')
            raw0,ff0,rr=recurrence(rt,env,obs,key,np.zeros_like(safe),safety,grad)
            mraw=z['center0_M_flow'][2];mff=z['center0_M_exec'][3,2]
            err=rel(rr[0]['M'],mraw);ferr=rel(rr[1]['M'],mff)
            row=dict(scene=sc,state_uid=uid,raw_AD_FD_relative_error=err,FF_KKT_FD_relative_error=ferr,
                     FF_weak_steps=sum(x['weak'] for x in rr[1]['steps']),steps=rr[1]['steps'],
                     float64_enabled=bool(jax.config.x64_enabled),obs_dtype=str(obs.dtype),
                     velocity_dtype=str(rt.velocity_fn(obs,jnp.zeros_like(obs).ravel()[:safe.size].reshape(safe.shape),jnp.array(0.)).dtype),
                     flow_singular_values=np.linalg.svd(rr[0]['M'],compute_uv=False),FF_singular_values=np.linalg.svd(rr[1]['M'],compute_uv=False),
                     final_projection_factorization_error=rel(rr[1]['last_dp']@rr[1]['last_pre'],rr[1]['M']))
            records.append(row)
            operators[uid+'_flow']=rr[0]['M'];operators[uid+'_FF']=rr[1]['M'];operators[uid+'_last_DP']=rr[1]['last_dp'];operators[uid+'_last_pre']=rr[1]['last_pre']
            np.testing.assert_allclose(ff0.ravel(),z['exec_all'][3,1],atol=5e-9,rtol=0)
            _,tr0=rt.integrate(env,obs,key,np.zeros_like(safe),safety,traced=True);sig0=branch_signature(safety,tr0)
            for j in selected:
                delta=(basis@etas[j]).reshape(safe.shape)
                for alpha in levels:
                    try:
                        raw,_=rt.integrate(env,obs,key,alpha*delta,safety,project=False)
                        ff,tr=rt.integrate(env,obs,key,alpha*delta,safety,traced=True)
                        switched=bool(np.any(branch_signature(safety,tr)!=sig0))
                        for name,value,base,m in [('flow',raw,raw0,mraw),('FF',ff,ff0,mff)]:
                            actual=(value-base).ravel();pred=m@(alpha*delta.ravel());e=np.linalg.norm(actual-pred)
                            scale_rows.append(dict(scene=sc,state_uid=uid,eta_index=j,alpha=alpha,map=name,
                                delta_norm=np.linalg.norm(alpha*delta),error=e,response_norm=np.linalg.norm(actual),
                                relative_error=e/max(np.linalg.norm(actual),1e-8),branch_switch=switched if name=='FF' else False))
                    except Exception as exc:errors.append(dict(scene=sc,state_uid=uid,eta_index=j,alpha=alpha,error=str(exc)))
            replay.append(dict(scene=sc,state_uid=uid,error=float(np.max(abs(ff0.ravel()-z['exec_all'][3,1])))))
            print('action probes',uid,'AD',err,'FF recurrence',ferr,flush=True)
        # Keep scene-specific JAX precision exactly as the registered runtime.
    np.savez_compressed(HERE/'recurrence_matrices.npz',**operators)
    save('recurrence_checks.json',records);save('scale_probe_rows.json',scale_rows);save('probe_numerical_failures.json',errors)
    save('probe_replay_checks.json',replay)
    summary={}
    for sc in ['toy_give_way','ring_exchange']:
        for name in ['flow','FF']:
            for alpha in levels:
                rows=[r for r in scale_rows if r['scene']==sc and r['map']==name and r['alpha']==alpha]
                summary[f'{sc}:{name}:{alpha}']=dict(n=len(rows),error=stats([r['error'] for r in rows]),
                    relative_error=stats([r['relative_error'] for r in rows]),
                    over25pct=np.mean([r['relative_error']>.25 for r in rows]),
                    signature_switch_fraction=np.mean([r['branch_switch'] for r in rows]),
                    smooth_relative_error=stats([r['relative_error'] for r in rows if not r['branch_switch']]))
    save('scale_probe_summary.json',summary)
    extras=[]
    pool=read(FIELD/'validation_stage2/protocol.json')['pool']
    for sc in ['double_bottleneck','four_way_intersection']:
        rt=make_runtime(sc)
        for item in [p for p in pool if p['state']['scenario']==sc][:2]:
            state=item['state'];env=rt.make_env(state);key=jax.random.PRNGKey(202610071)
            obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
            basis=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,axis=-1).reshape(-1,3)
            eta=etas[4];delta=(basis@eta).reshape(safe.shape);d=delta.size;mat=[]
            base,_=rt.integrate(env,obs,key,delta,safety)
            forward=[];backward=[];fail=[]
            for epsilon in [.01,.001,.0001]:
                h=epsilon*rt.config.max_speed;columns=[]
                for v in np.eye(d):
                    try:
                        pp,_=rt.integrate(env,obs,key,delta+h*v.reshape(delta.shape),safety)
                        mm,_=rt.integrate(env,obs,key,delta-h*v.reshape(delta.shape),safety)
                        columns.append(((pp-mm)/(2*h)).ravel())
                        if epsilon==.0001:forward.append(((pp-base)/h).ravel());backward.append(((base-mm)/h).ravel())
                    except Exception as exc:columns.append(np.full(d,np.nan));fail.append(str(exc))
                mat.append(np.stack(columns,axis=1))
            row=dict(scene=sc,state_uid=state['uid'],eta=eta,failures=fail,M=mat[-1],
                epsilon_change=rel(mat[-2],mat[-1]),one_sided_mismatch=rel(np.array(forward),np.array(backward)),
                singular_values=np.linalg.svd(mat[-1],compute_uv=False) if np.isfinite(mat[-1]).all() else None,
                checkpoint=str(rt.checkpoint),checkpoint_sha256=sha(rt.checkpoint),float64_enabled=bool(jax.config.x64_enabled))
            extras.append(row);print('extra scenario',sc,state['uid'],flush=True)
    save('other_scenario_probes.json',extras)
    print('all bounded action probes complete; zero new rollouts',flush=True)

if __name__=='__main__':run()

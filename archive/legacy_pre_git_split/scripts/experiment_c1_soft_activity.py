"""Deterministic full-task before/after experiment with fixed C1 infrastructure."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.differentiable_rollout import ResidualFlowField, bounded_nominal
from single_integrator.c1.train import rollout_terms
from single_integrator.c1.socp import ExactProjection
from single_integrator.c1.risk.soft_activity import SoftRiskConfig,soft_risk_diagnostics
from single_integrator.c1.risk.risk_function import trajectory_risk
from single_integrator.c1.training.primal_dual import PrimalDualState,dual_update
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.evaluate import load_policy
from single_integrator.outcomes import first_event

INITIAL=np.array([[-.18,.008],[.18,-.005]])
SEED=1042

def noise_at(step):
    key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(SEED),0),step)
    return jax.random.normal(key,(1,4),dtype=jnp.float32)

def full_task(params,base,model,plant,cbf,risk,out,label):
    field=ResidualFlowField(base,model)
    field.baseline_sample=jax.jit(field.baseline_sample)
    field.correction=jax.jit(field.correction)
    score=jax.jit(lambda task,blocks,h,s:soft_risk_diagnostics(task,blocks,h,s,risk))
    env=GiveWayEnv(plant);env.reset(INITIAL)
    start_errors=np.linalg.norm(env.goals-INITIAL,axis=-1)
    records=[]
    def projection(target,A,b,speed):
        return jnp.asarray(project_velocity(np.asarray(target)[0],A,b,speed,cbf)[0].reshape(1,4))
    for step in range(plant.max_steps):
        snapshot=env.snapshot();A,b,geometry=barrier_constraints(snapshot,cbf)
        h=np.r_[geometry['pairwise_h'],geometry['wall_h'].ravel()]
        control=field.control(params,jnp.asarray(env.observation()[None]),noise_at(step),A,b,plant.max_speed,projection)
        applied=np.asarray(control['applied'])[0]
        safe=np.asarray(control['safe'])[0]
        s=A@applied-b
        diagnostic=score(control['candidate'].reshape(2,2),A.reshape(17,2,2),h,s)
        before=env.positions.copy()
        _,_,done,info=env.step(applied.reshape(2,2))
        records.append(dict(**{k:np.asarray(v)[0] for k,v in control.items()},
            **{k:np.asarray(v) for k,v in diagnostic.items()},h=h,
            positions_before=before,positions=env.positions.copy(),goal_errors=info['goal_errors'],
            window_progress=info['window_progress'],stuck_timer=info['stuck_timer'],
            candidate_deadlock=info['candidate_deadlock'],deadlock=info['deadlock'],
            min_swept_agent_distance=info['min_swept_agent_distance'],
            min_swept_wall_distance=info['min_swept_wall_distance'],
            J_def=np.sum((applied-safe)**2)))
        if done:break
    trace={key:np.stack([r[key] for r in records]) for key in records[0]}
    np.savez_compressed(out/(label+'_trajectory.npz'),**trace,initial_positions=INITIAL)
    finite=trace['margins'][np.isfinite(trace['margins'])]
    summary=dict(outcome=first_event(info),steps=len(records),seconds=len(records)*plant.dt,
        R_risk=float(trajectory_risk(trace['risk'],risk)),mean_risk=float(trace['risk'].mean()),
        J_def=float(trace['J_def'].mean()),min_interagent_distance=float(trace['min_swept_agent_distance'].min()+2*plant.agent_radius),
        min_wall_clearance=float(trace['min_swept_wall_distance'].min()),
        min_cbf_residual=float(trace['residual'].min()),
        max_speed=float(np.linalg.norm(trace['applied'].reshape(-1,2,2),axis=-1).max()),
        start_goal_errors=start_errors.tolist(),final_goal_errors=trace['goal_errors'][-1].tolist(),
        progress=float(start_errors.sum()-trace['goal_errors'][-1].sum()),
        hard_active_fraction=float(np.mean(np.any(trace['active_mask'],axis=1))),
        hard_deadlock_geometry_fraction=float(np.mean(np.any(trace['hard_deadlock_geometry'],axis=1))),
        mean_pair_h=float(trace['h'][:,0].mean()),mean_pair_slack=float(trace['residual'][:,0].mean()),
        mean_local_activity=float(trace['local_activity'].mean()),mean_cone_risk=float(trace['cone_risk'].mean()),
        mean_finite_margin=float(finite.mean()) if len(finite) else None)
    (out/(label+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(label,json.dumps(summary),flush=True)
    return summary,trace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['baseline','train'],required=True)
    p.add_argument('--out',type=Path,default=Path('results/c1_soft_activity_experiment'))
    p.add_argument('--updates',type=int,default=8)
    p.add_argument('--horizon',type=int,default=160)
    p.add_argument('--x-offset',type=float,default=0.)
    args=p.parse_args();args.out.mkdir(exist_ok=True)
    INITIAL[:,0] += args.x_offset
    jax.config.update('jax_enable_x64',True)
    checkpoint=Path('baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    base,metadata=load_policy(checkpoint);plant=Config(**metadata['evaluation_environment']);cbf=CBFConfig()
    risk=SoftRiskConfig(rho=.05,active_tol=1e-7,D0=.1,kappa=2.,alpha=.5,p=4.,tau_h=.01,tau_s=.01,candidate_sigmas=6.)
    model=ResidualCorrection(hidden_dims=(32,32))
    params=model.init(jax.random.PRNGKey(7),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
    contract=dict(method='c1_v0',risk_version='R_risk_v0_1',risk=asdict(risk),environment=plant.to_dict(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),noise_seed=SEED,initial_positions=INITIAL.tolist(),
        horizon=args.horizon,updates=args.updates,lr=3e-4,dual_lr=.5,residual_hidden_dims=[32,32])
    if args.phase=='baseline':
        (args.out/'config.json').write_text(json.dumps(contract,indent=2)+'\n')
        full_task(params,base,model,plant,cbf,risk,args.out,'baseline')
        return
    assert json.loads((args.out/'config.json').read_text())==contract,'configuration must match baseline'
    baseline_summary=json.loads((args.out/'baseline_summary.json').read_text())
    epsilon=.8*baseline_summary['R_risk']
    assert 0<epsilon<baseline_summary['R_risk']
    field=ResidualFlowField(base,model);solver=ExactProjection()
    field.baseline_sample=jax.jit(field.baseline_sample)
    field.correction=jax.jit(field.correction)
    noise=jnp.stack([noise_at(t) for t in range(args.horizon)],axis=1)
    initial=jnp.asarray(INITIAL[None]);dual=PrimalDualState();optimizer=optax.adam(3e-4);state=optimizer.init(params);history=[]
    def terms(phi):
        u,safe,r,d=rollout_terms(phi,field,solver,initial,noise,plant,cbf,risk,return_details=True)
        return (jnp.mean(jnp.sum((u-safe)**2,axis=-1)),jnp.mean(r)),dict(
            active_fraction=jnp.mean(jnp.any(d['active_mask'],axis=-1)),
            progress=jnp.linalg.norm(jnp.asarray(GiveWayEnv(plant).goals)-initial,axis=-1).sum()-jnp.linalg.norm(jnp.asarray(GiveWayEnv(plant).goals)-(d['positions'][:,-1]+plant.dt*u[:,-1].reshape(1,2,2)),axis=-1).sum())
    for update in range(args.updates):
        start=time.time()
        print('begin update',update,flush=True)
        (jdef,jlive),pullback,aux=jax.vjp(terms,params,has_aux=True)
        gd=pullback((jnp.array(1.),jnp.array(0.)))[0];gl=pullback((jnp.array(0.),jnp.array(1.)))[0]
        gradient=jax.tree_util.tree_map(lambda a,b:a+dual.dual*b,gd,gl)
        newdual=dual_update(dual,jlive,epsilon,.5)
        row=dict(update=update,J_def=float(jdef),J_live=float(jlive),epsilon=epsilon,
            lambda_used=dual.dual,lambda_after=newdual.dual,live_gradient_norm=float(optax.global_norm(gl)),
            **{k:float(v) for k,v in aux.items()},seconds=time.time()-start)
        history.append(row);print(json.dumps(row),flush=True)
        delta,state=optimizer.update(gradient,state,params);params=optax.apply_updates(params,delta);dual=newdual
        (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        with (args.out/'short_run_params.pkl').open('wb') as f:pickle.dump(jax.device_get(params),f)
    final_summary,_=full_task(params,base,model,plant,cbf,risk,args.out,'after')
    report=dict(baseline=baseline_summary,after=final_summary,epsilon=epsilon,history=history,
        risk_decreased=final_summary['R_risk']<baseline_summary['R_risk'],
        progress_improved=final_summary['progress']>baseline_summary['progress'],
        deadlock_escaped=baseline_summary['outcome']=='safe_deadlock' and final_summary['outcome']=='success')
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

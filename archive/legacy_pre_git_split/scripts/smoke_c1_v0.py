"""Deterministic short C1 V0 check; never a benchmark or long training run."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dataclasses import asdict
from single_integrator.c1.train import rollout_terms
from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.socp import ExactProjection
from single_integrator.c1.risk.risk_function import RiskConfig
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy

def main():
    jax.config.update('jax_enable_x64', True)
    baseline, metadata = load_policy(Path('baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl'))
    plant, cbf = Config(**metadata['evaluation_environment']), CBFConfig()
    risk = RiskConfig(rho=.05, active_tol=1e-7, D0=.1, kappa=2., alpha=.5, p=4.)
    model = ResidualCorrection(hidden_dims=(32,32))
    params = model.init(jax.random.PRNGKey(7),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
    field, solver = ResidualFlowField(baseline,model), ExactProjection()
    initial = jnp.array([[[-.18,.008],[.18,-.005]]])
    noise = jax.random.normal(jax.random.PRNGKey(1042),(1,4,4),dtype=jnp.float32)
    def terms(phi):
        u,safe,r,d = rollout_terms(phi,field,solver,initial,noise,plant,cbf,risk,return_details=True)
        aux = dict(correction_norm=jnp.linalg.norm(d['correction']),
                   active_steps=jnp.count_nonzero(jnp.any(d['active_mask'],axis=-1)),
                   min_cbf_residual=jnp.min(jnp.einsum('btij,btj->bti',d['A'],d['applied'])-d['b']),
                   max_speed=jnp.max(jnp.linalg.norm(d['applied'].reshape(1,4,2,2),axis=-1)))
        return (jnp.mean(jnp.sum((u-safe)**2,axis=-1)),jnp.mean(r)),aux
    u,safe,r = rollout_terms(params,field,solver,initial,noise,plant,cbf,risk)
    identity_error=float(jnp.max(jnp.abs(u-safe)))
    assert identity_error < 2e-6, identity_error
    baseline_risk=float(jnp.mean(r))
    assert baseline_risk > 0, 'fixture must have nonzero baseline risk'
    epsilon=.8*baseline_risk
    print('BASELINE',baseline_risk,'epsilon',epsilon,'identity_error',identity_error,flush=True)
    optimizer=optax.adam(3e-4);state=optimizer.init(params);dual=0.;rows=[]
    for step in range(12):
        (jdef,jlive),pullback,aux=jax.vjp(terms,params,has_aux=True)
        gd=pullback((jnp.array(1.),jnp.array(0.)))[0]
        gl=pullback((jnp.array(0.),jnp.array(1.)))[0]
        gradient=jax.tree_util.tree_map(lambda d,l:d+dual*l,gd,gl)
        next_dual=max(0.,dual+.5*(float(jlive)-epsilon))
        row=dict(update=step,J_def=float(jdef),J_live=float(jlive),lambda_used=dual,
                 lambda_after=next_dual,live_gradient_norm=float(optax.global_norm(gl)),
                 **{key:float(value) for key,value in aux.items()})
        rows.append(row);print(json.dumps(row),flush=True)
        delta,state=optimizer.update(gradient,state,params)
        params=optax.apply_updates(params,delta);dual=next_dual
    (final_def,final_live),final_aux=terms(params)
    # Measure the correction on the initial state using exactly the same sample.
    from single_integrator.c1.train import observation
    from single_integrator.c1.differentiable_rollout import barrier_constraints
    from single_integrator.environment import GiveWayEnv
    env=GiveWayEnv(plant)
    A,b,_=jax.vmap(lambda p:barrier_constraints(p,jnp.array(env.walls),plant.to_dict(),cbf))(initial)
    control=field.control(params,observation(initial,jnp.zeros_like(initial),jnp.array(env.goals)),noise[:,0],A,b,plant.max_speed,solver)
    result=dict(method='c1_v0',risk_version='R_risk_v0',risk=asdict(risk),initial_positions=np.asarray(initial).tolist(),
                noise_seed=1042,residual_seed=7,residual_hidden_dims=[32,32],
                primal_lr=3e-4,dual_lr=.5,updates=12,horizon=4,
                baseline_R_risk=baseline_risk,epsilon=epsilon,
                zero_residual_identity_error=identity_error,history=rows,
                final_J_def=float(final_def),final_J_live=float(final_live),
                final_correction_norm=float(jnp.linalg.norm(control['correction'])),
                final_diagnostics={key:float(value) for key,value in final_aux.items()},
                safety='both projections checked at every step using authoritative tolerances')
    assert rows[0]['lambda_after'] > rows[0]['lambda_used']
    assert rows[0]['live_gradient_norm'] > 1e-8
    assert result['final_correction_norm'] > 0
    assert result['final_J_live'] < baseline_risk
    assert result['final_J_def'] < 1e-3  # declared smoke-only deformation limit
    out=Path('results/c1_v0_smoke');out.mkdir(exist_ok=True)
    for label,phi in [('final',params)]:
        _,_,_,diagnostics=rollout_terms(phi,field,solver,initial,noise,plant,cbf,risk,return_details=True)
        np.savez_compressed(out/(label+'_diagnostics.npz'),**jax.device_get(diagnostics))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL',json.dumps({k:v for k,v in result.items() if k.startswith('final_')}),flush=True)

if __name__=='__main__':
    main()

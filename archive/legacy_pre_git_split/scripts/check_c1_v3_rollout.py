"""Real 850-step Flow/CBF V3 forward, environment and BPTT smoke audit."""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.train_v3 import setup, noise, source_hashes
from single_integrator.c1.rollout_v3 import rollout
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.training.step_control import finite, guarded_step
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update
from single_integrator.environment import GiveWayEnv
from single_integrator.cbf import barrier_constraints, project_velocity
from single_integrator.c1.risk.joint_frozen import trajectory as legacy_score
from single_integrator.c1.risk.risk_v3 import progress_steps
from c1_heldout_safety import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    params,field,plant,cbf,_ = setup()
    data = json.loads(args.sets.read_text())
    item = data['pools']['train'][0]
    initial=jnp.asarray(item['initial']); draws=noise(data['calibration_noise_seeds'][0],item['rid'])
    fn=jax.jit(lambda p:rollout(p,field,initial,draws,plant,cbf))
    started=time.monotonic()
    base,trace=fn(params)
    if not finite((base,trace)):
        raise FloatingPointError('nonfinite V3 full rollout')
    trace=jax.tree_util.tree_map(np.asarray,trace)
    n=int(base['steps']);env=GiveWayEnv(plant);env.reset(np.asarray(initial))
    step_error=0.;safe_error=0.;statuses=[];stagnation=0
    for t in range(n):
        step_error=max(step_error,float(np.max(np.abs(env.positions-trace['before'][t]))))
        # Independently invoke runtime baseline sampling and safety projection.
        raw=field.baseline_sample(jnp.asarray(env.observation()[None]),draws[t:t+1])
        from single_integrator.c1.differentiable_rollout import bounded_nominal
        nominal=np.asarray(bounded_nominal(raw,.5))[0]
        A,b,_=barrier_constraints(env.snapshot(),cbf)
        expected=project_velocity(nominal,A,b,.5,cbf)[0].reshape(4)
        safe_error=max(safe_error,float(np.max(np.abs(expected-trace['applied'][t]))))
        _,_,done,info=env.step(trace['applied'][t].reshape(2,2))
        statuses.append(info['task_success']);stagnation+=bool(info['candidate_deadlock'])
        if done and t!=n-1:
            raise AssertionError('V3 terminal mask disagrees with environment')
    safety=check(dict(positions_before=trace['before'][:n],positions_after=trace['after'][:n],
                      applied=trace['applied'][:n]),env,cbf)
    assert step_error<1e-8 and safe_error<2e-6
    assert all(safety[k]==0 for k in ('agent_collision_steps','wall_collision_steps',
        'outside_endpoints','cbf_violations','speed_violations'))
    p,_=progress_steps(jnp.asarray(trace['before']),jnp.asarray(trace['after']),
                       jnp.asarray(env.goals),jnp.asarray(trace['success']))
    old=legacy_score(jnp.asarray(trace['before'][:500]),jnp.asarray(trace['after'][:500]),
        jnp.asarray(trace['candidate'][:500]),jnp.asarray(trace['g'][:200]),
        jnp.asarray(env.goals),jnp.asarray(trace['success'][:500]))
    np.testing.assert_allclose(float(p[100:300].mean()),float(old[0]),atol=1e-12)
    result=dict(scope='One training start, engineering smoke; not independent efficacy evidence',
        baseline={k:float(v) for k,v in base.items()},steps=n,runtime_state_error=step_error,
        runtime_control_error=safe_error,safety=safety,stagnation_seconds=stagnation*.05,
        source_hashes=source_hashes(),legacy_window_Pg=float(old[:2].sum()))
    atomic_save(args.out/'forward.json',result)
    print(dict(forward=result,elapsed=time.monotonic()-started),flush=True)
    # Fixed lambda=1 diagnoses the full risk derivative independently of the
    # intentional lambda=0 first step in primal-dual training.
    def objective(phi):
        values,_=fn(phi)
        return values['J_live']+values['J_def'],values
    before,grad=jax.value_and_grad(objective,has_aux=True)(params)
    assert finite((before,grad))
    norm=float(optax.global_norm(grad));assert norm>0
    optimizer=optax.adam(1e-5)
    updated,_,after,decision=guarded_step(params,optimizer.init(params),grad,optimizer,
                                         objective,before,max_backtracks=8)
    assert decision['status']=='accepted'
    assert float(after[0])<=float(before[0])
    # This is an observed check, not an extra constraint on C1 updates.
    result.update(gradient_norm=norm,step_control=decision,post={k:float(v) for k,v in after[1].items()},
                  loss_before=float(before[0]),loss_after=float(after[0]),elapsed=time.monotonic()-started,
                  dual_from_zero=dual_update(PrimalDualState(),base['J_live'],.8*float(base['J_live']),.01).dual)
    atomic_save(args.out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

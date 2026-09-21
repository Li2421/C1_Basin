"""Fresh closed-loop execution of the selected fixed-noise interventions."""
import json
import hashlib
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_clean_geometry import OUT,BASE,SEED,START,ROOT,save
from audit_c1_multistep_objectives import ACTIONS
from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity
from single_integrator.evaluate import load_policy

def main():
    choices=json.loads((OUT/'choices.json').read_text())
    selected=sorted({(r['rid'],r['action']) for r in choices})
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    frames=0
    for rid,action in selected:
        z=np.load(BASE/'traces'/f'{rid:04d}_{SEED}_{action}.npz')
        env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False))
        env.reset(z['positions_before'][0]);cfg=CBFConfig();key=jax.random.fold_in(jax.random.PRNGKey(SEED),rid)
        for t in range(len(z['success'])):
            if t<START: u=z['applied'][t]
            else:
                raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
                A,b,_=barrier_constraints(env.snapshot(),cfg)
                safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
                delta=ACTIONS[action] if t<140 else np.zeros(4)
                v=safe+delta
                u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(delta) else safe
                np.testing.assert_array_equal(v,z['candidate'][t])
            np.testing.assert_array_equal(u,z['applied'][t])
            _,_,done,info=env.step(u.reshape(2,2))
            np.testing.assert_array_equal(env.positions,z['positions_after'][t])
            assert info['task_success']==z['success'][t] and info['deadlock']==z['deadlock'][t]
            assert not info['wall_collision'] and not info['agent_collision']
            frames+=1
        print('verified',rid,action,flush=True)
    old=json.loads((ROOT/'results/c1_completed_waiting_risk_audit/verification.json').read_text())['sha256']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in old}
    assert hashes==old
    save(OUT/'verification.json',dict(unique_selected_branches=len(selected),frames=frames,
        fresh_closed_loop_execution_exact_match=True,no_collision=True,original_core_and_geometry_hashes_unchanged=True,hashes=hashes))

if __name__=='__main__':main()

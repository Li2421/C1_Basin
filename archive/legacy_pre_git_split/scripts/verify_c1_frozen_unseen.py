"""Independent saved-trace checks and selected-branch policy/environment replay."""
import json
import hashlib
import time
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from evaluate_c1_frozen_unseen import OUT,ROOT,manifest,PREFIX_SEED,PLAN_SEED
from audit_c1_candidate_coverage import metrics,delta_at,START,LONG
from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity
from single_integrator.evaluate import load_policy

def main():
    p=json.loads((OUT/'protocol.json').read_text());assert p['hashes']==manifest()
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    cfg=CBFConfig();score_error=0.;position_error=0.;control_error=0.;checked=0;replayed=0
    for rid,initial in zip(p['ids'],p['initials']):
        deadline=time.monotonic()+7200
        while not (OUT/'states'/f'{rid}.json').exists():
            if time.monotonic()>deadline:raise TimeoutError(f'Incomplete state {rid}')
            time.sleep(1)
        state=json.loads((OUT/'states'/f'{rid}.json').read_text())
        base=np.load(OUT/'traces'/str(rid)/'zero.npz')
        for c in p['library']:
            path=OUT/'traces'/str(rid)/c['cid']
            row=json.loads(path.with_name(path.name+'.json').read_text())
            with np.load(path.with_name(path.name+'.npz')) as z:
                np.testing.assert_array_equal(z['positions_before'][0],initial)
                np.testing.assert_array_equal(z['positions_after'][:START],base['positions_after'][:START])
                if row['controller_error']:continue
                cost,out=metrics(z);assert out==row['outcome']
                if cost is not None:
                    score_error=max(score_error,max(abs(cost[k]-row['costs'][k]) for k in cost))
                    # Deliberately poison positions after the prediction endpoint:
                    # scalar scores must not see eventual outcome positions.
                    poisoned={k:z[k].copy() for k in z.files}
                    poisoned['positions_after'][LONG:]=123.
                    cost2,_=metrics(poisoned)
                    assert cost==cost2
                checked+=1
        selected=state['selected']
        if selected is None:continue
        c=next(c for c in p['library'] if c['cid']==selected['cid'])
        z=np.load(OUT/'traces'/str(rid)/(c['cid']+'.npz'))
        env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False));env.reset(np.asarray(initial))
        for t in range(len(z['success'])):
            key=jax.random.fold_in(jax.random.PRNGKey(PREFIX_SEED if t<START else PLAN_SEED),rid)
            raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
            A,b,_=barrier_constraints(env.snapshot(),cfg)
            safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
            delta=delta_at(c,t);v=safe+delta
            u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(delta) else safe
            _,_,done,info=env.step(u.reshape(2,2))
            control_error=max(control_error,float(np.max(np.abs(u-z['applied'][t]))))
            position_error=max(position_error,float(np.max(np.abs(env.positions-z['positions_after'][t]))))
            for field,key in [('success','task_success'),('deadlock','deadlock'),('candidate_deadlock','candidate_deadlock')]:
                assert info[key]==z[field][t]
            if done:assert t==len(z['success'])-1
        replayed+=1
        print('verified',rid,'traces',checked,'replays',replayed,flush=True)
    assert score_error<1e-12 and position_error<1e-10 and control_error<1e-10
    result=dict(manifest_verified=True,trace_metrics_checked=checked,selected_policy_replays=replayed,
        identical_prefix_verified=True,post_25s_positions_do_not_affect_score=True,
        max_score_difference=score_error,max_position_difference=position_error,max_control_difference=control_error,
        script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()

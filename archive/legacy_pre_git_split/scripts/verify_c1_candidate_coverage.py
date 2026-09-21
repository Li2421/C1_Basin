"""Independent argmin, information-boundary and selected-branch execution checks."""
import json
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_candidate_coverage import OUT,BASE,SEED,START,LONG,metrics,delta_at,save
from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity
from single_integrator.evaluate import load_policy
from audit_c1_joint_witness_risk import ROOT

def main():
    rows=json.loads((OUT/'rows.json').read_text());assert len(rows)==322
    summary=json.loads((OUT/'summary.json').read_text());pairs=[];separation={}
    for rid in [68,208]:
        rr=[r for r in rows if r['rid']==rid];eligible=[r for r in rr if r['costs'] is not None]
        ranked=sorted(eligible,key=lambda r:(r['costs']['score'],r['cid']))
        assert ranked[0]['cid']==summary[str(rid)]['selected']['cid']
        good=[r for r in ranked if r['outcome']['success']];bad=[r for r in ranked if not r['outcome']['success']]
        separation[rid]=dict(best_success_score=good[0]['costs']['score'] if good else None,
            best_failed_score=bad[0]['costs']['score'] if bad else None,
            best_success_margin=bad[0]['costs']['score']-good[0]['costs']['score'] if good and bad else None,
            all_successes_above_all_failures=bool(good and bad and good[-1]['costs']['score']<bad[0]['costs']['score']),
            best_failed=bad[0] if bad else None)
        pairs.append(ranked[0])
        if good and good[0]['cid']!=ranked[0]['cid']:pairs.append(good[0])
        if bad and bad[0]['cid']!=ranked[0]['cid']:pairs.append(bad[0])
        # Poison states beyond the scoring lookahead, retaining trajectory length.
        baseline=next(r for r in rr if r['cid']=='zero')
        z0=np.load(OUT/'traces'/f'{rid:04d}_zero.npz');z={k:z0[k].copy() for k in z0.files}
        original,_=metrics(z);z['positions_after'][LONG:]+=100
        changed,_=metrics(z);assert original==changed
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');frames=0
    for row in pairs:
        rid=row['rid'];z=np.load(OUT/'traces'/f"{rid:04d}_{row['cid']}.npz")
        env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False));env.reset(z['positions_before'][0])
        cfg=CBFConfig();key=jax.random.fold_in(jax.random.PRNGKey(SEED),rid)
        for t in range(len(z['success'])):
            if t<START:u=z['applied'][t]
            else:
                raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
                A,b,_=barrier_constraints(env.snapshot(),cfg)
                safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
                delta=delta_at(row['intervention'],t);v=safe+delta
                u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(delta) else safe
                np.testing.assert_array_equal(v,z['candidate'][t])
            _,_,_,info=env.step(u.reshape(2,2));np.testing.assert_array_equal(env.positions,z['positions_after'][t])
            assert info['task_success']==z['success'][t] and info['deadlock']==z['deadlock'][t];frames+=1
        print('replayed',rid,row['cid'],flush=True)
    save(OUT/'ranking_verification.json',dict(separation=separation,independent_argmin=True,
        no_post25_position_leak=True,replayed_branches=len(pairs),replayed_frames=frames,exact_closed_loop_replay=True))

if __name__=='__main__':main()

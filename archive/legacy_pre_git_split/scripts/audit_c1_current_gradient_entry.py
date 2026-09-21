"""Audit actual frozen scoring entry points, never substitute the legacy risk.

A non-traceable necessary component blocks a full rollout gradient; this script
reports that block, rather than manufacturing a stopped/finite-difference VJP.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_joint_witness_risk import ROOT,LocalRisk
from audit_c1_candidate_coverage import metrics
from audit_c1_completed_waiting_risk import MAXR
from single_integrator.cbf import CBFConfig,barrier_constraints
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.train import observation
from single_integrator.evaluate import load_policy

OUT=ROOT/'results/c1_pretraining_audits';BASE=ROOT/'results/c1_frozen_unseen_64'

def attempt(name,fn,arg):
    try:
        value,grad=jax.value_and_grad(fn)(arg)
        leaves=jax.tree_util.tree_leaves(grad)
        return dict(name=name,backprop_completed=True,value=float(value),finite=all(np.isfinite(np.asarray(a)).all() for a in leaves),
            gradient_norm=float(np.sqrt(sum(np.sum(np.asarray(a)**2) for a in leaves))))
    except Exception as e:
        return dict(name=name,backprop_completed=False,error_type=type(e).__name__,error=str(e))

def main():
    assert (OUT/'mismatch/summary.json').exists(),'Run audits in requested order'
    jax.config.update('jax_enable_x64',True)
    with np.load(BASE/'traces/100000/zero.npz') as data:z={k:data[k] for k in data.files}
    env=GiveWayEnv(Config(corridor_half_length=1.3));cfg=CBFConfig();t=100
    A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cfg)
    local=LocalRisk(A,b,np.full(2,.5));value=local.score(z['candidate'][t])['risk']/MAXR
    assert abs(value-z['g'][0])<1e-12
    tests=[attempt('current_local_g_wrt_v',lambda v:local.score(v)['risk']/MAXR,jnp.asarray(z['candidate'][t]))]
    tests.append(attempt('current_trajectory_score_wrt_controls',lambda v:metrics(dict(z,candidate=v))[0]['score'],jnp.asarray(z['candidate'])))
    baseline,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    model=ResidualCorrection();field=ResidualFlowField(baseline,model)
    params=model.init(jax.random.PRNGKey(0),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
    params=jax.tree_util.tree_map(lambda a:jnp.asarray(a,jnp.float64),params)
    obs=observation(jnp.asarray(z['positions_before'][t:t+1]),jnp.asarray(z['applied'][t-1:t].reshape(1,2,2)),jnp.asarray(env.goals))
    safe=jnp.asarray(z['candidate'][t:t+1]);assert np.max(np.abs(field.correction(params,obs,safe)))==0
    def residual_risk(phi):
        v=safe+field.correction(phi,obs,safe)
        return local.score(v[0])['risk']/MAXR
    tests.append(attempt('current_g_wrt_actual_G_phi_at_zero_initialization',residual_risk,params))
    result=dict(status='blocked_before_full_rollout_backprop' if not all(r['backprop_completed'] for r in tests) else 'entry_checks_pass_only',
        tests=tests,forward_g_matches_frozen=True,
        active_training_risk='train.py CLI supports v0_1/v1/v1_1/v2; default v2 calls episode_rollout.soft_risk_diagnostics and trajectory_risk_v2, not LocalRisk or P+Gtwo.',
        horizon_mismatch='Existing v2 requires full max_steps and first-event termination including deadlock; frozen scorer uses decision5s, endpoint25s and does not terminate at deadlock.',
        full_objective_gradient_finite=None,explosion_or_vanishing=None,term_gradient_dominance=None,negative_gradient_risk_decrease=None,
        reason='Necessary current local and trajectory score entry points have no JAX backward implementation; full current-risk training objective is not wired. These nulls mean not measurable, not passed.',
        pilot_started=False,main_code_modified=False,
        mathematical_caveat='This diagnoses the current implementation, not impossibility of differentiating the underlying convex projection.',
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['single_integrator/c1/train.py','single_integrator/c1/episode_rollout.py','scripts/audit_c1_joint_witness_risk.py','scripts/audit_c1_candidate_coverage.py']})
    (OUT/'gradient_entry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Frozen synthetic/recorded counterexamples for VI R_CERT semantics."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.vi_r_cert import _vi_margins, trajectory
from single_integrator.c1.rollout_vi_r_cert import WITNESS_TARGETS
from single_integrator.c1.training.persistence import atomic_save


DT=.05
GOALS=np.array([[1.09,0.],[-1.09,0.]])


def physical(actions, initial=np.array([[-.4,0.],[.4,0.]])):
    actions=np.asarray(actions,float).reshape(-1,2,2)
    states=np.concatenate([initial[None],initial[None]+DT*np.cumsum(actions,axis=0)])
    return states[:-1],states[1:]


def monitor(actions, before, after, horizon):
    actions=np.asarray(actions).reshape(horizon,2,2)
    states=np.concatenate([before[:1],after])
    errors=np.linalg.norm(states-GOALS,axis=-1)
    speed=np.linalg.norm(actions,axis=-1).max(axis=-1)
    candidate=np.zeros(horizon,bool)
    for k in range(39,horizon):
        candidate[k]=(errors[k+1].max()>.08 and
                      np.abs(errors[k-39]-errors[k+1]).max()<.01 and
                      speed[k]<.025)
    run=0; first=None
    for k,value in enumerate(candidate):
        run=run+1 if value else 0
        if run>=101 and first is None:first=k
    stalled=(horizon>=40 and np.all(speed[-40:]<.05) and
             np.all(np.abs(errors[-1]-errors[-40])<.02))
    return dict(historical_strict=first is not None,
                first_deadlock_action=first, tail_stall_atoms=bool(stalled),
                max_candidate_run=int(max((len(x) for x in ''.join(
                    '1' if z else '0' for z in candidate).split('0')),default=0)),
                tail_max_speed=float(speed[-40:].max()),
                tail_goal_error_change=np.abs(errors[-1]-errors[-40]).tolist())


def risk_for(actions, original_timeout):
    before,after=physical(actions)
    flat=np.asarray(actions).reshape(850,4)
    witnesses=np.broadcast_to(np.asarray(WITNESS_TARGETS),(850,8,4))
    result=trajectory(jnp.asarray(before),jnp.asarray(after),jnp.asarray(flat),
        jnp.asarray(flat),jnp.asarray(witnesses),jnp.asarray(GOALS),
        jnp.ones(850,bool),jnp.zeros(850,bool),original_timeout)
    return {k:float(result[k]) for k in ('direct_R_CERT','augmented_R_CERT')}


def main():
    jax.config.update('jax_enable_x64',True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,
        default=ROOT/'results/c1_vi_r_cert_counterexamples_v1.json')
    out=parser.parse_args().out
    if out.exists():raise FileExistsError('counterexample output already exists')
    zero=np.zeros((850,4));zb,za=physical(zero)
    zero_monitor=monitor(zero,zb,za,850)
    zero_monitor.update(original_other_timeout=False,stalled_deadlock=False,
                        primary=True)
    zero_monitor['risk']=risk_for(zero,False)

    jitter=np.zeros((850,4));jitter[:,0]=.03*(-1.)**np.arange(850)
    jitter[:,2]=-.03*(-1.)**np.arange(850)
    jb,ja=physical(jitter);jitter_monitor=monitor(jitter,jb,ja,850)
    jitter_monitor.update(original_other_timeout=True,
                          stalled_deadlock=jitter_monitor['tail_stall_atoms'],
                          primary=jitter_monitor['tail_stall_atoms'])
    jitter_monitor['risk']=risk_for(jitter,True)

    outsider=np.zeros((850,4));outsider[:,2]=.1*(-1.)**np.arange(850)
    ob,oa=physical(outsider);outsider_monitor=monitor(outsider,ob,oa,850)
    outsider_monitor.update(original_other_timeout=True,
                            stalled_deadlock=False,primary=False)

    retreat=np.zeros((850,4));retreat[:,0]=-.1;retreat[:,2]=.1
    rb,ra=physical(retreat);retreat_monitor=monitor(retreat,rb,ra,850)
    retreat_monitor.update(original_other_timeout=True,
                           stalled_deadlock=False,primary=False)

    delayed=np.zeros((1000,4));delayed[:830,0]=.1*(-1.)**np.arange(830)
    delayed[:830,2]=-.1*(-1.)**np.arange(830)
    db,da=physical(delayed)
    short_monitor=monitor(delayed[:850],db[:850],da[:850],850)
    long_monitor=monitor(delayed,db,da,1000)
    short_monitor.update(original_other_timeout=True,stalled_deadlock=False,
                         primary=False)
    long_monitor.update(original_other_timeout=False,stalled_deadlock=False,
                        primary=True)

    # Symmetric VI terms can cancel even with a nonzero term-level response.
    witnesses=np.broadcast_to(np.asarray(WITNESS_TARGETS),(850,8,4))
    def symmetric_objective(w):
        result=trajectory(jnp.asarray(zb),jnp.asarray(za),w,jnp.zeros((850,4)),
            jnp.asarray(witnesses),jnp.asarray(GOALS),jnp.ones(850,bool),
            jnp.zeros(850,bool),False)
        return result['augmented_R_CERT']
    symmetric_gradient=jax.grad(symmetric_objective)(jnp.zeros((850,4)))
    symmetric_gradient_host=np.asarray(symmetric_gradient)

    w=jnp.array([2.,10.]);v=jnp.array([[.5,0.],[-.5,0.],[0.,0.],[0.,0.]])
    plateau=np.asarray(_vi_margins(
        w,v,lambda q:jnp.maximum(jnp.linalg.norm(q,axis=-1)-1.,0.)**2))* .25

    # Recorded successful trajectory: quantify genuine waiting without a latch.
    trace_path=ROOT/'results/c1_vi_r_cert_probe_v1/reference_1_84220.npz'
    with np.load(trace_path) as tr:
        speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1)
        candidate=np.asarray(tr['candidate_deadlock'],bool)
        wait=np.max(np.sum(speed<.025,axis=0))
        recorded=dict(trace=str(trace_path.relative_to(ROOT)),actions=len(speed),
            terminal='success',candidate_samples=int(candidate.sum()),
            minimum_agent_speed=float(speed.min()),
            per_agent_low_speed_samples=np.sum(speed<.025,axis=0).tolist(),
            at_least_one_agent_low_speed_samples=int(np.sum(np.any(speed<.025,axis=1))),
            simple_low_speed_count_upper_bound=int(wait),historical_strict=False)

    report=dict(
        zero_control_blocking=zero_monitor,
        nonzero_control_jitter_stagnation=jitter_monitor,
        one_agent_stuck_moving_outsider=outsider_monitor,
        retreat_without_stop=retreat_monitor,
        horizon_delay=dict(H850=short_monitor,H1000=long_monitor),
        symmetry=dict(augmented_w_gradient_norm=(float(np.linalg.norm(
            symmetric_gradient_host)) if np.isfinite(symmetric_gradient_host).all()
            else None), nonfinite_components=int(np.sum(
                ~np.isfinite(symmetric_gradient_host))),
            exact_zero=bool(np.all(symmetric_gradient_host==0.))),
        projection_plateau=dict(
            setup='U=[-2,2]x{0}, w=(2,10), B=unit ball',
            exact_certificate=4.,unnormalized_vi=plateau.tolist(),
            all_vi_nonpositive=bool(np.all(plateau<=0))),
        recorded_normal_yielding=recorded,
        interpretation=dict(
            local_deadlock='one stopped agent is not D when the moving outsider breaks max-speed atoms',
            jitter='0.03 m/s defeats strict speed but is caught by stalled timeout',
            horizon='a low-speed phase beginning too near H can first deadlock only after H and evade the H tail test',
            retreat='persistent retreat is task abandonment but not the frozen deadlock event'))
    atomic_save(out,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

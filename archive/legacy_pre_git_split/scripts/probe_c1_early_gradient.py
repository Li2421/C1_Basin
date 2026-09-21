"""Independent-noise early-gradient diagnostic, not shared-policy efficacy."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from single_integrator.c1.train_deadlock_union import setup, noise, digest, source_hashes
from single_integrator.c1.rollout_early_diagnostic import rollout
from single_integrator.c1.early_intervention import execute_prefix
from single_integrator.c1.risk.ordered_guidance import trajectory
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.train import observation
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('new output required')
    if jax.default_backend() != 'gpu': raise RuntimeError('Slurm GPU required')
    params, field, plant, cbf, baseline = setup()
    goals = jnp.asarray(GiveWayEnv(plant).goals)
    kwargs = dict(dt=plant.dt, max_speed=plant.max_speed, goal_tolerance=plant.goal_tolerance,
                  hold_seconds=plant.deadlock_hold_seconds, progress_window_seconds=plant.progress_window_seconds,
                  progress_epsilon=plant.progress_epsilon, speed_epsilon_fraction=plant.speed_epsilon_fraction)
    rng = np.random.default_rng(2026091805)
    x=rng.uniform(.55,1.05,(20,2)); y=rng.uniform(-.025,.025,(20,2))
    starts=np.stack([np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]],1)
    prediction_seeds=[8241830,8241831]; evaluation_seeds=list(range(8241840,8241844))
    args.out.mkdir(parents=True)
    paths=['scripts/probe_c1_early_gradient.py','single_integrator/c1/early_intervention.py',
           'single_integrator/c1/rollout_early_diagnostic.py','single_integrator/c1/evaluate_deadlock_union.py',
           'single_integrator/c1/risk/ordered_guidance.py','single_integrator/c1/risk/exact_margin.py']
    sources={**source_hashes(),**{p:digest(ROOT/p) for p in paths}}
    protocol=dict(scope='per-start local derivative development; not shared-policy efficacy',
        starts=starts.tolist(),prediction_seeds=prediction_seeds,evaluation_seeds=evaluation_seeds,
        arms=['reference','negative_early','positive_early','random_early','negative_full'],
        early_steps=100,early_seconds=5.,gradient='mean of two independent prediction-noise gradients of early-only intervention risk',
        scale='0.005 m/s predicted same-state residual component RMS; parameter L2 cap 0.1',
        random_direction='fixed seeded isotropic parameter direction with independent same-state RMS normalization',
        full_arm='same early-objective negative direction applied all episode, not independently retuned',
        zero_gradient='retain zero intervention, never replace with heuristic',
        planned_outcome_replays=400,prediction_gradient_rollouts=40,prediction_value_check_rollouts=80,
        difficulty_gate='reference deadlock >=10% and at least five distinct deadlock starts',
        interpretation='require deadlock reduction AND success increase; report paired start-cluster uncertainty and timeout conversions; no automatic full-training launch',
        environment=plant.to_dict(),cbf=cbf.to_dict(),baseline_sha256=digest(baseline),sources=sources)
    atomic_save(args.out/'protocol.json',protocol)
    for path,expected in sources.items():
        data=(ROOT/path).read_bytes()
        if digest(ROOT/path)!=expected: raise RuntimeError('source changed')
        target=args.out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)

    def objective(phi, initial, draws):
        terms,tr=rollout(phi,field,initial,draws,plant,cbf,intervention_steps=100)
        risk=trajectory(tr['before'],tr['after'],tr['applied'],goals,tr['alive'],terminal_timeout=terms['timeout'],**kwargs)['J_live']
        velocity=jnp.concatenate([jnp.zeros((1,2,2)),tr['applied'][:99].reshape(-1,2,2)])
        obs=observation(tr['before'][:100],velocity,goals)
        return risk,(obs,tr['safe'][:100],terms['steps'])
    vg=jax.jit(jax.value_and_grad(objective,has_aux=True))
    value_only=jax.jit(lambda phi,initial,draws:objective(phi,initial,draws)[0])
    jvp=jax.jit(lambda direction,obs,safe:jax.jvp(lambda p:field.correction(p,obs,safe),(params,),(direction,))[1])
    records=[];gradients=[];begun=time.monotonic();replay_seconds=0.
    for rid,initial in enumerate(starts):
        samples=[vg(params,jnp.asarray(initial),noise(seed,30000+rid)) for seed in prediction_seeds]
        gradient=jax.tree_util.tree_map(lambda a,b:(a+b)/2,samples[0][1],samples[1][1])
        norm=float(optax.global_norm(gradient))
        if not np.isfinite(norm): raise FloatingPointError('nonfinite gradient')
        negative=jax.tree_util.tree_map(lambda g:-g/max(norm,1e-300),gradient)
        random=jax.tree_util.tree_map(lambda p:jnp.asarray(rng.normal(size=p.shape),dtype=p.dtype),params)
        random_norm=float(optax.global_norm(random))
        random=jax.tree_util.tree_map(lambda p:p/random_norm,random)
        def scale(direction):
            outputs=[jvp(direction,s[0][1][0],s[0][1][1]) for s in samples]
            sensitivity=float(jnp.sqrt(jnp.mean(jnp.concatenate(outputs)**2)))
            return min(.005/max(sensitivity,1e-300),.1),sensitivity
        alpha,sensitivity=scale(negative);alpha=alpha if norm>0 else 0.
        random_alpha,random_sensitivity=scale(random)
        candidate=lambda direction,step:jax.tree_util.tree_map(lambda p,d:p+step*d,params,direction)
        minus=candidate(negative,alpha);plus=candidate(negative,-alpha);random_phi=candidate(random,random_alpha)
        mean_value=lambda phi:float(np.mean([float(value_only(phi,jnp.asarray(initial),noise(seed,30000+rid))) for seed in prediction_seeds]))
        gradients.append(dict(rid=rid,gradient_norm=norm,mean_prediction_risk=float(np.mean([float(s[0][0]) for s in samples])),
            negative_prediction_risk=mean_value(minus),positive_prediction_risk=mean_value(plus),
            alpha=alpha,sensitivity=sensitivity,random_alpha=random_alpha,random_sensitivity=random_sensitivity))
        atomic_save(args.out/'gradients.json',gradients)
        arms=[('reference',params,0),('negative_early',minus,100),('positive_early',plus,100),
              ('random_early',random_phi,100),('negative_full',minus,plant.max_steps)]
        for seed in evaluation_seeds:
            base_trace=None
            for name,phi,steps in arms:
                started=time.monotonic()
                row,tr=execute_prefix(params,phi,field,initial,noise(seed,30000+rid),plant,cbf,prefix_steps=steps)
                replay_seconds+=time.monotonic()-started
                outcome='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
                label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
                    goal_errors=np.linalg.norm(tr['positions_after']-np.asarray(goals),axis=-1)),outcome,plant.dt)
                if base_trace is None: base_trace=tr
                n=min(100,len(tr['applied']),len(base_trace['applied']))
                row.update(rid=rid,seed=seed,arm=name,outcome=label,
                    either_deadlock=label in ('safe_deadlock','stalled_deadlock'),
                    actual_early_action_rms=float(np.sqrt(np.mean((tr['applied'][:n]-base_trace['applied'][:n])**2))))
                records.append(row);np.savez_compressed(args.out/f'{name}_{rid}_{seed}.npz',**tr)
                atomic_save(args.out/'records.json',records)
                if replay_seconds>1800: raise TimeoutError('replay budget exceeded; incomplete, no efficacy claim')
        print(json.dumps(dict(starts_completed=rid+1,replays=len(records),elapsed=time.monotonic()-begun)),flush=True)
    atomic_save(args.out/'complete.json',dict(replays=len(records),elapsed=time.monotonic()-begun,
                                             replay_seconds=replay_seconds,independent_test_opened=False))


if __name__=='__main__': main()

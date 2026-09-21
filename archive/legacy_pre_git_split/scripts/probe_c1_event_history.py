"""Initial risk prediction and matched early action-gradient development tests."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,digest,source_hashes
from single_integrator.c1.rollout_event_history import rollout
from single_integrator.c1.action_probe import execute_offsets,calibrate_offsets
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out
    if out.exists():raise FileExistsError('new output directory required')
    if jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    parity=json.loads((ROOT/'results/c1_action_probe_parity_v1/complete.json').read_text())
    if not parity['passed']:raise RuntimeError('interface parity gate not passed')
    adapter=json.loads((ROOT/'results/c1_event_history_prefix_v1/complete.json').read_text())
    if not adapter['passed']:raise RuntimeError('forecast prefix gate not passed')
    params,field,plant,cbf,baseline=setup()
    goals=jnp.asarray(GiveWayEnv(plant).goals)
    kw=dict(dt=plant.dt,max_speed=plant.max_speed,goal_tolerance=plant.goal_tolerance,
        hold_seconds=plant.deadlock_hold_seconds,progress_window_seconds=plant.progress_window_seconds,
        progress_epsilon=plant.progress_epsilon,speed_epsilon_fraction=plant.speed_epsilon_fraction)
    rng=np.random.default_rng(2026091867)
    x=rng.uniform(.55,1.05,(25,2));y=rng.uniform(-.025,.025,(25,2))
    starts=np.stack([np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]],1)
    prediction_seeds=list(range(82810,82812));evaluation_seeds=list(range(82820,82822))
    arms=['reference','negative','positive','random']
    out.mkdir(parents=True)
    sources=source_hashes()
    for p in ['scripts/probe_c1_event_history.py','scripts/analyze_c1_event_history.py',
              'C1_EVENT_HISTORY_PROBE_PROTOCOL.md','single_integrator/c1/action_probe.py',
              'single_integrator/c1/rollout_event_history.py',
              'single_integrator/c1/evaluate_deadlock_union.py',
              'single_integrator/c1/risk/event_history.py']:
        sources[p]=digest(ROOT/p)
    atomic_save(out/'protocol.json',dict(scope='new fixed-horizon historical-event risk: 50 paired cases, 200 arm replays; not trained-policy efficacy',
        starts=starts.tolist(),rid_offset=63000,prediction_seeds=prediction_seeds,evaluation_seeds=evaluation_seeds,
        arms=arms,early_steps=100,initial_G_phi='frozen zero-output initialization',
        random_seed=2026091868,target_projected_rms=.0025,component_cap=.025,
        planned_outcome_replays=200,prediction_gradient_rollouts=50,
        environment=plant.to_dict(),cbf=cbf.to_dict(),baseline_sha256=digest(baseline),sources=sources))
    for p,sha in sources.items():
        assert digest(ROOT/p)==sha
        dst=out/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    (out/'reference_params.pkl').write_bytes(pickle.dumps(jax.tree_util.tree_map(np.asarray,params)))
    def objective(offsets,initial,draws):
        full=jnp.pad(offsets,((0,plant.max_steps-100),(0,0)))
        terms,tr=rollout(params,field,initial,draws,plant,cbf,action_offsets=full)
        errors=jnp.linalg.norm(tr['after']-goals,axis=-1)
        speed=jnp.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1)
        stalled=terms['timeout'] & jnp.all(speed[-40:]<.05) & jnp.all(jnp.abs(errors[-1]-errors[-40])<.02)
        expected=terms['deadlock'] | stalled
        return terms['J_live'],(tr['candidate'][:100],tr['A'][:100],tr['b'][:100],expected,terms['steps'])
    vg=jax.jit(jax.value_and_grad(objective,has_aux=True))
    zero=jnp.zeros((100,4),jnp.float64);random_rng=np.random.default_rng(2026091868)
    diagnostics=[];records=[];begun=time.monotonic();replay_seconds=0.
    for rid,initial in enumerate(starts):
        samples=[vg(zero,jnp.asarray(initial),noise(s,63000+rid)) for s in prediction_seeds]
        gradient=np.mean([np.asarray(s[1]) for s in samples],axis=0)
        norm=float(np.linalg.norm(gradient))
        if not np.isfinite(norm):raise FloatingPointError('nonfinite action gradient')
        for sample in samples:
            if bool(sample[0][0]>1)!=bool(sample[0][1][3]):raise RuntimeError('forecast risk/event mismatch')
        prediction_samples=[tuple(np.asarray(a) for a in s[0][1][:3]) for s in samples]
        random=random_rng.normal(size=(100,4))
        directions=dict(negative=-gradient,positive=gradient,random=random if norm>0 else np.zeros_like(random))
        offsets={'reference':np.zeros((100,4))};calibration={}
        for arm,direction in directions.items():
            offsets[arm],calibration[arm]=calibrate_offsets(direction,prediction_samples,cbf)
        diagnostic=dict(rid=rid,gradient_norm=norm,prediction_risks=[float(s[0][0]) for s in samples],
            prediction_risk=float(np.mean([float(s[0][0]) for s in samples])),calibration=calibration,
            prediction_deadlocks=[bool(s[0][1][3]) for s in samples],prediction_first_event_steps=[int(s[0][1][4]) for s in samples])
        diagnostics.append(diagnostic);atomic_save(out/'diagnostics.json',diagnostics)
        np.savez_compressed(out/f'offsets_{rid}.npz',gradient=gradient,**offsets)
        for seed in evaluation_seeds:
            for arm in arms:
                started=time.monotonic()
                row,tr=execute_offsets(params,field,initial,noise(seed,63000+rid),plant,cbf,offsets[arm])
                replay_seconds+=time.monotonic()-started
                outcome='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
                label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
                    goal_errors=np.linalg.norm(tr['positions_after']-np.asarray(goals),axis=-1)),outcome,plant.dt)
                changes=tr['direct_action_changes']
                row.update(rid=rid,seed=seed,arm=arm,outcome=label,
                    either_deadlock=label in ('safe_deadlock','stalled_deadlock'),
                    direct_action_sum_squares=float(np.sum(changes**2)),direct_action_components=int(changes.size))
                records.append(row);np.savez_compressed(out/f'{arm}_{rid}_{seed}.npz',**tr)
                atomic_save(out/'records.json',records)
                if replay_seconds>1800:raise TimeoutError('30 minute planning budget exceeded; no completion claim')
        print(json.dumps(dict(starts_completed=rid+1,replays=len(records),elapsed=time.monotonic()-begun)),flush=True)
    assert all(digest(ROOT/p)==sha for p,sha in sources.items()),'source changed during experiment'
    atomic_save(out/'complete.json',dict(replays=len(records),elapsed=time.monotonic()-begun,
        replay_seconds=replay_seconds,independent_test_opened=False))


if __name__=='__main__':main()

"""Fixed 2x2 risk/step-rule diagnostic, with independent physical outcomes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from single_integrator.c1.train_deadlock_union import setup,noise,digest,source_hashes
from single_integrator.c1.rollout_early_diagnostic import rollout
from single_integrator.c1.early_intervention import execute_prefix
from single_integrator.c1.risk.ordered_guidance import trajectory as ordered
from single_integrator.c1.risk.terminal_progress_guard import trajectory as guarded
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.train import observation
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('new output required')
    if jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    params,field,plant,cbf,baseline=setup();goals=jnp.asarray(GiveWayEnv(plant).goals)
    kw=dict(dt=plant.dt,max_speed=plant.max_speed,goal_tolerance=plant.goal_tolerance,
            hold_seconds=plant.deadlock_hold_seconds,progress_window_seconds=plant.progress_window_seconds,
            progress_epsilon=plant.progress_epsilon,speed_epsilon_fraction=plant.speed_epsilon_fraction)
    rng=np.random.default_rng(2026091807);x=rng.uniform(.55,1.05,(20,2));y=rng.uniform(-.025,.025,(20,2))
    starts=np.stack([np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]],1)
    prediction_seeds=[8241860,8241861];evaluation_seeds=list(range(8241870,8241874))
    arms=['reference','ordered_fixed','ordered_backtrack','guarded_fixed','guarded_backtrack']
    args.out.mkdir(parents=True)
    paths=['scripts/probe_c1_guard_factorial.py','scripts/analyze_c1_guard_factorial.py',
           'single_integrator/c1/rollout_early_diagnostic.py','single_integrator/c1/early_intervention.py',
           'single_integrator/c1/evaluate_deadlock_union.py','single_integrator/c1/risk/ordered_guidance.py',
           'single_integrator/c1/risk/exact_margin.py','single_integrator/c1/risk/terminal_progress_guard.py']
    sources={**source_hashes(),**{p:digest(ROOT/p) for p in paths}}
    atomic_save(args.out/'protocol.json',dict(scope='local 2x2 development ablation; not shared training',
        starts=starts.tolist(),prediction_seeds=prediction_seeds,evaluation_seeds=evaluation_seeds,arms=arms,
        early_steps=100,scale='same-state residual component RMS .005 m/s; parameter L2 cap .1',
        backtracking='try initial step then halvings through 1/32; accept first actual mean risk decrease >1e-9*max(1,abs(before)); otherwise unchanged',
        no_candidates='backtracking optimizes parameters on fixed prediction noise; no outcome/noise/path selection',
        planned_outcome_replays=400,prediction_gradient_rollouts=80,max_prediction_value_rollouts=480,
        difficulty_gate='reference >=10% deadlocks and >=5 independent deadlock starts',
        efficacy_gate='paired deadlock decrease with CI excluding zero, success increase, no total-failure increase, and original deadlock-to-success >=50%; diagnostic only',
        environment=plant.to_dict(),cbf=cbf.to_dict(),baseline_sha256=digest(baseline),sources=sources))
    for path in sources:
        target=args.out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
    def objective(phi,initial,draws,index):
        terms,tr=rollout(phi,field,initial,draws,plant,cbf,intervention_steps=100)
        base=ordered(tr['before'],tr['after'],tr['applied'],goals,tr['alive'],terminal_timeout=terms['timeout'],**kw)['J_live']
        candidate=guarded(tr['before'],tr['after'],tr['applied'],goals,tr['alive'],terminal_timeout=terms['timeout'],**kw)['J_live']
        risk=jnp.where(index==0,base,candidate)
        velocity=jnp.concatenate([jnp.zeros((1,2,2)),tr['applied'][:99].reshape(-1,2,2)])
        obs=observation(tr['before'][:100],velocity,goals)
        return risk,(obs,tr['safe'][:100])
    vg=jax.jit(jax.value_and_grad(objective,has_aux=True))
    value=jax.jit(lambda phi,x,z,i:objective(phi,x,z,i)[0])
    jvp=jax.jit(lambda d,obs,safe:jax.jvp(lambda p:field.correction(p,obs,safe),(params,),(d,))[1])
    records=[];diagnostics=[];begun=time.monotonic();replay_seconds=0.
    for rid,initial in enumerate(starts):
        draws=[noise(s,50000+rid) for s in prediction_seeds];initial=jnp.asarray(initial)
        policies={'reference':params}
        for index,name in enumerate(('ordered','guarded')):
            samples=[vg(params,initial,z,jnp.asarray(index)) for z in draws]
            gradient=jax.tree_util.tree_map(lambda a,b:(a+b)/2,samples[0][1],samples[1][1])
            norm=float(optax.global_norm(gradient))
            if not np.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            direction=jax.tree_util.tree_map(lambda g:-g/max(norm,1e-300),gradient)
            sensitivity=float(jnp.sqrt(jnp.mean(jnp.concatenate([jvp(direction,s[0][1][0],s[0][1][1]) for s in samples])**2)))
            alpha=min(.005/max(sensitivity,1e-300),.1) if norm>0 else 0.
            before=float(np.mean([float(s[0][0]) for s in samples]));trials=[];accepted=params;accepted_alpha=0.
            for iteration in range(6):
                step=alpha*2**(-iteration)
                proposal=jax.tree_util.tree_map(lambda p,d:p+step*d,params,direction)
                after=float(np.mean([float(value(proposal,initial,z,jnp.asarray(index))) for z in draws])) if alpha else before
                if not np.isfinite(after):raise FloatingPointError('nonfinite objective')
                if iteration==0:policies[name+'_fixed']=proposal
                trials.append(dict(alpha=step,risk=after))
                if after<before-1e-9*max(1.,abs(before)):
                    accepted=proposal;accepted_alpha=step;break
                if not alpha:break
            policies[name+'_backtrack']=accepted
            diagnostics.append(dict(rid=rid,risk=name,gradient_norm=norm,sensitivity=sensitivity,
                initial_alpha=alpha,accepted_alpha=accepted_alpha,before=before,trials=trials))
        atomic_save(args.out/'diagnostics.json',diagnostics)
        atomic_save(args.out/f'local_params_{rid}.pkl',dict(params=jax.tree_util.tree_map(np.asarray,policies),
                    rid=rid,initial=np.asarray(initial),scope='diagnostic per-start fits, not deployable shared checkpoint'),binary=True)
        for seed in evaluation_seeds:
            reference_trace=None
            for name in arms:
                start=time.monotonic()
                row,tr=execute_prefix(params,policies[name],field,initial,noise(seed,50000+rid),plant,cbf,
                                      prefix_steps=0 if name=='reference' else 100)
                replay_seconds+=time.monotonic()-start
                outcome='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
                label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
                    goal_errors=np.linalg.norm(tr['positions_after']-np.asarray(goals),axis=-1)),outcome,plant.dt)
                if reference_trace is None:reference_trace=tr
                n=min(100,len(tr['applied']),len(reference_trace['applied']))
                row.update(rid=rid,seed=seed,arm=name,outcome=label,either_deadlock=label in ('safe_deadlock','stalled_deadlock'),
                    actual_early_action_rms=float(np.sqrt(np.mean((tr['applied'][:n]-reference_trace['applied'][:n])**2))))
                records.append(row);np.savez_compressed(args.out/f'{name}_{rid}_{seed}.npz',**tr)
                atomic_save(args.out/'records.json',records)
                if replay_seconds>1800:raise TimeoutError('replay budget exceeded; incomplete')
        print(json.dumps(dict(starts_completed=rid+1,replays=len(records),elapsed=time.monotonic()-begun)),flush=True)
    atomic_save(args.out/'complete.json',dict(replays=len(records),elapsed=time.monotonic()-begun,
                                            replay_seconds=replay_seconds,independent_test_opened=False))


if __name__=='__main__':main()

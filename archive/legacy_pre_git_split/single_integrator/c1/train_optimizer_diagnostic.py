"""Matched 16-update optimizer diagnostic; no independent test execution."""
import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from single_integrator.c1.train_deadlock_union import setup, noise, digest, source_hashes
from single_integrator.c1.rollout_deadlock_union import rollout
from single_integrator.c1.risk.ordered_guidance import trajectory
from single_integrator.c1.episode_pool import EpisodePool
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.training.step_control_restart import guarded_step, finite
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update
from single_integrator.c1.training.selection import better_feasible_candidate
from single_integrator.c1.acceptance import assess
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--updates',type=int,choices=(16,),default=16)
    parser.add_argument('--optimizer',choices=('adam','sgd'),required=True)
    parser.add_argument('--warm-start',type=Path,required=True)
    parser.add_argument('--training-seed',type=int,default=2026091842)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('new output required')
    probe=json.loads((args.probe/'complete.json').read_text())
    prior_analysis=json.loads((args.probe/'analysis.json').read_text())
    if probe.get('replays')!=400 or not prior_analysis['difficulty_passed']:
        raise ValueError('requires completed difficult 2x2 development experiment')
    gate=prior_analysis['development_gate_passed']
    # New bounded optimizer audit, not a declaration that the risk passed
    # its efficacy gate. This module cannot open independent evaluation.
    params,field,plant,cbf,baseline=setup(0,0)
    if jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    original_params=params
    if args.warm_start:
        previous=pickle.loads(args.warm_start.read_bytes())
        if previous['config']['baseline_sha256']!=digest(baseline) or previous['config']['environment']!=plant.to_dict():
            raise ValueError('warm-start baseline/environment mismatch')
        params=jax.tree_util.tree_map(jnp.asarray,previous['params'])
    benchmark=json.loads((ROOT/'results/c1_episode_pool_benchmark.json').read_text())
    if not benchmark['agreement_passed'] or benchmark['speedup']<=1:
        raise RuntimeError('parallel path requires numerical agreement and measured speedup')
    pool=EpisodePool('ordered',workers=4)
    oldsets=json.loads((ROOT/'results/c1_deadlock_union/sets.json').read_text())
    data=dict(train=oldsets['pools']['train'],validation=oldsets['pools']['validation'],
              validation_noise_seeds=[82551,82552])
    args.out.mkdir(parents=True)
    atomic_save(args.out/'sets.json',data)
    config=dict(risk='relu(1 + rho_exact * exp(sign(rho_exact) * smooth_sum))',epsilon=.01,lr=.0001,
        dual_lr=.1,initial_dual=1.,batch_size=8,updates=args.updates,validation_every=4,optimizer=args.optimizer,development_only=True,
        baseline_sha256=digest(baseline),environment=plant.to_dict(),seed=0,training_rng_seed=args.training_seed,
        sets_sha256=digest(args.out/'sets.json'),probe_sha256=digest(args.probe/'complete.json'),
        warm_start_sha256=digest(args.warm_start) if args.warm_start else None,
        warm_start_path=str(args.warm_start) if args.warm_start else None,
        warm_start_selected_update=previous.get('selection',{}).get('update') if args.warm_start else None,
        optimizer_audit_budget=16,prior_factorial_gate_passed=gate,
        workers=4,parallel_benchmark_sha256=digest(ROOT/'results/c1_episode_pool_benchmark.json'),
        parallelism='Independent scalar rollouts, same batch loss/gradient mean; benchmark verified',
        source_hashes={**source_hashes(),**{p:digest(ROOT/p) for p in (
            'single_integrator/c1/train_optimizer_diagnostic.py','single_integrator/c1/episode_pool.py',
            'single_integrator/c1/risk/ordered_guidance.py','single_integrator/c1/risk/exact_margin.py',
            'single_integrator/c1/evaluate_deadlock_union.py','single_integrator/c1/acceptance.py',
            'single_integrator/c1/paired_statistics.py','scripts/c1_heldout_safety.py')}},
        test_budget='none: development only, no test states loaded',
        difficulty='Before training: unfiltered validation Safety >=10% deadlock and >=10 deadlock start clusters',
        selection='minimum validation J_def among J_live<=.01',
        local_gate_passed=gate,entry_reason='matched optimizer stability audit, not efficacy-gated full training',
        objective='min J_def subject to E R <= epsilon')
    atomic_save(args.out/'config.json',config)
    for path,expected in config['source_hashes'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError('source changed')
        target=args.out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
    begun=time.monotonic()
    def measure(phi,pool,seeds):
        items=[item for item in pool for seed in seeds]
        draws=[seed for item in pool for seed in seeds]
        values=workers.rows(phi,items,draws)
        rows=[dict(rid=i['rid'],noise_seed=s,**value[1]) for i,s,value in zip(items,draws,values)]
        if not finite([v[1] for v in values]):raise FloatingPointError('nonfinite rollout')
        return dict(**{k:float(np.mean([r[k] for r in rows])) for k in values[0][1]},episodes=rows)
    workers=pool
    baseline_val=measure(original_params,data['validation'],data['validation_noise_seeds'])
    for r in baseline_val['episodes']:
        if bool(r['either_deadlock']) != (r['J_live']>1.):
            raise AssertionError('exact score and original detector disagree on development')
    difficult_clusters=len({r['rid'] for r in baseline_val['episodes'] if r['either_deadlock']})
    difficulty=baseline_val['either_deadlock']>=.1 and difficult_clusters>=10
    atomic_save(args.out/'difficulty.json',dict(passed=difficulty,deadlock_clusters=difficult_clusters,**baseline_val))
    if not difficulty:raise RuntimeError('Safety development difficulty gate failed; do not run easy comparison')
    eps=config['epsilon'];optimizer=(optax.adam(config['lr']) if args.optimizer=='adam' else optax.sgd(config['lr']));state=optimizer.init(params)
    dual=PrimalDualState(1.);rng=np.random.default_rng(args.training_seed);history=[];validations=[];best=None
    def validate(update,precomputed=None):
        nonlocal best
        row=measure(params,data['validation'],data['validation_noise_seeds']) if precomputed is None else precomputed
        row.update(update=update,epsilon=eps,constraint=row['J_live']-eps)
        validations.append(row);atomic_save(args.out/'validation.json',validations)
        if better_feasible_candidate(row,best):
            best=row
            atomic_save(args.out/'best_feasible.pkl',dict(params=jax.device_get(params),config=config,selection=row),binary=True)
        print(json.dumps(dict(validation={k:v for k,v in row.items() if k!='episodes'},elapsed=time.monotonic()-begun)),flush=True)
    validate(0,None if args.warm_start else baseline_val)
    for step in range(args.updates):
        ids=rng.choice(len(data['train']),size=config['batch_size'],replace=False)
        items=[data['train'][int(i)] for i in ids];seed=int(rng.integers(1000000,2**31-1))
        def objective(phi):
            return workers.evaluate(phi,items,seed,dual.dual,eps)
        before,g=workers.evaluate(params,items,seed,dual.dual,eps,gradient=True)
        used=dual.dual
        params,state,after,decision=guarded_step(params,state,g,optimizer,objective,before)
        dual=dual_update(dual,before[1]['J_live'],eps,config['dual_lr'])
        history.append(dict(update=step,rids=[i['rid'] for i in items],noise_seed=seed,lambda_used=used,
            lambda_after=dual.dual,gradient_norm=float(optax.global_norm(g)),decision=decision,
            pre={k:float(v) for k,v in before[1].items()},post={k:float(v) for k,v in after[1].items()}))
        atomic_save(args.out/'checkpoint.pkl',dict(params=jax.device_get(params),optimizer=jax.device_get(state),
            dual=dual.dual,rng_state=rng.bit_generator.state,completed_updates=step+1,history=history,config=config),binary=True)
        atomic_save(args.out/'history.json',history)
        print(json.dumps(dict(update=step+1,**history[-1]['post'],elapsed=time.monotonic()-begun)),flush=True)
        interval=4
        if (step+1)%interval==0 or step+1==args.updates:validate(step+1)
    atomic_save(args.out/'training_complete.json',dict(updates=len(history),validation_feasible=best is not None,
        elapsed_seconds=time.monotonic()-begun))
    workers.close()
    atomic_save(args.out/'complete.json',dict(status='development_optimizer_diagnostic_complete',
        optimizer=args.optimizer,completed_updates=len(history),validation_feasible=best is not None,
        test_opened=False,elapsed_seconds=time.monotonic()-begun))


if __name__=='__main__':main()


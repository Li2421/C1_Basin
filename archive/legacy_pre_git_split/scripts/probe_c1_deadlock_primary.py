"""Local first-event BPTT diagnostic; not training or held-out efficacy evidence."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.train_completion import setup, noise, digest
from single_integrator.c1.rollout_deadlock_primary import rollout
from single_integrator.c1.evaluate_completion import execute
from single_integrator.c1.training.step_control_restart import guarded_step, finite
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update
from single_integrator.c1.training.persistence import atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=8)
    args = p.parse_args()
    if args.out.exists() or args.updates < 1:
        raise ValueError('new output directory and positive update count required')
    folder = ROOT/'results/c1_independent_distribution/extra_validation_baseline0'
    paths = sorted(folder.glob('Safety_*.npz'))
    for path in paths:
        with np.load(path) as z:
            if np.any(z['deadlock']):
                initial = np.array(z['positions_before'][0])
                break
    else:
        raise ValueError('no existing development deadlock')
    _, rid, seed = path.stem.split('_')
    params, field, plant, cbf, baseline = setup()
    plant = replace(plant, terminate_on_deadlock=True)
    draws = noise(int(seed), int(rid))
    args.out.mkdir(parents=True)
    atomic_save(args.out/'protocol.json', dict(scope='Local diagnosis on first sorted previously observed deadlock; no generalization claim',
        trace=str(path), trace_sha256=digest(path), baseline_sha256=digest(baseline),
        updates=args.updates, epsilon=.05, lr=.0001, initial_dual=1., dual_lr=.1,
        risk='scene-independent smooth temporal upper certificate', backend=jax.default_backend(),
        sources={s:digest(ROOT/s) for s in ('single_integrator/c1/risk/deadlock_primary.py',
            'single_integrator/c1/rollout_deadlock_primary.py', 'scripts/probe_c1_deadlock_primary.py')}))
    fn = jax.jit(lambda phi: rollout(phi, field, jnp.asarray(initial), draws, plant, cbf)[0])
    optimizer = optax.adam(.0001)
    state, dual = optimizer.init(params), PrimalDualState(1.)
    history = []
    started = time.monotonic()
    # Independent authoritative environment replay before/after the diagnostic.
    row, trace = execute(params, field, initial, draws, plant, cbf)
    atomic_save(args.out/'before_execution.json', row)
    np.savez_compressed(args.out/'before_execution.npz', **trace)
    print(dict(before_execution=row, seconds=time.monotonic()-started), flush=True)
    for step in range(args.updates):
        def objective(phi):
            result = fn(phi)
            return result['J_def']+dual.dual*(result['J_live']-.05), result
        before, grad = jax.value_and_grad(objective, has_aux=True)(params)
        if not finite((before, grad)):
            raise FloatingPointError('nonfinite objective or gradient')
        if step == 0 and (bool(before[1]['deadlock']) != row['any_deadlock'] or int(before[1]['steps']) != row['steps']):
            raise AssertionError('direct environment and differentiable first-event rollout disagree')
        used = dual.dual
        params, state, after, decision = guarded_step(params, state, grad, optimizer, objective, before)
        dual = dual_update(dual, before[1]['J_live'], .05, .1)
        record = dict(update=step+1, lambda_used=used, lambda_after=dual.dual,
            before={k:float(v) for k,v in before[1].items()},
            after={k:float(v) for k,v in after[1].items()},
            gradient_norm=float(optax.global_norm(grad)), decision=decision,
            elapsed_seconds=time.monotonic()-started)
        history.append(record)
        atomic_save(args.out/'history.json', history)
        print(record, flush=True)
    row, trace = execute(params, field, initial, draws, plant, cbf)
    atomic_save(args.out/'after_execution.json', row)
    np.savez_compressed(args.out/'after_execution.npz', **trace)
    atomic_save(args.out/'complete.json', dict(updates=len(history), after_execution=row,
        scope='Single already-seen case only; neither final C1 training nor efficacy'))
    print(dict(after_execution=row, seconds=time.monotonic()-started), flush=True)


if __name__ == '__main__':
    main()

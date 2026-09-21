"""400 fixed development rollouts: disjoint prediction and outcome noise.

Safety is frozen throughout. This probes conditional risk association, not
trained C1 efficacy. No trajectory selection, intervention, or test opening.
"""
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
from single_integrator.c1.train_deadlock_union import setup, noise, digest, source_hashes
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.risk.ordered_guidance import trajectory
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('new output required')
    if jax.default_backend() != args.backend:
        raise RuntimeError('requested backend required')
    phi, field, plant, cbf, baseline = setup()
    rng = np.random.default_rng(2026091802)
    x = rng.uniform(.55, 1.05, (40, 2))
    y = rng.uniform(-.025, .025, (40, 2))
    starts = np.stack([np.c_[-x[:, 0], y[:, 0]], np.c_[x[:, 1], y[:, 1]]], axis=1)
    prediction_seeds = list(range(8241810, 8241814))
    outcome_seeds = list(range(8241820, 8241826))
    goals = jnp.asarray(GiveWayEnv(plant).goals)
    kwargs = dict(dt=plant.dt, max_speed=plant.max_speed, goal_tolerance=plant.goal_tolerance,
                  hold_seconds=plant.deadlock_hold_seconds,
                  progress_window_seconds=plant.progress_window_seconds,
                  progress_epsilon=plant.progress_epsilon,
                  speed_epsilon_fraction=plant.speed_epsilon_fraction)
    score = jax.jit(lambda before, after, u, alive, timeout:
        trajectory(before, after, u, goals, alive, terminal_timeout=timeout, **kwargs)['J_live'])
    args.out.mkdir(parents=True)
    paths = ['scripts/probe_c1_conditional_risk_cpu.py',
             'single_integrator/c1/evaluate_deadlock_union.py',
             'single_integrator/c1/risk/ordered_guidance.py',
             'single_integrator/c1/risk/exact_margin.py']
    atomic_save(args.out/'protocol.json', dict(
        purpose='development conditional risk association; not final test or policy efficacy',
        policy='Safety zero residual', backend=args.backend, starts=starts.tolist(),
        prediction_seeds=prediction_seeds, outcome_seeds=outcome_seeds,
        environment=plant.to_dict(), cbf=cbf.to_dict(), baseline_sha256=digest(baseline),
        sources={**source_hashes(), **{p: digest(ROOT/p) for p in paths}},
        analysis='Compare one and four prediction-noise mean risks with independent six-noise deadlock frequency; start-cluster bootstrap; report ties and zero scores',
        total_rollouts=400, outcome_definition='strict or original stalled terminal event; other timeout separate',
        no_success_claim='Association only; no threshold tuning or score-to-probability calibration'))
    begun = time.monotonic()
    records = []
    for rid, initial in enumerate(starts):
        for seed in prediction_seeds + outcome_seeds:
            row, trace = execute(phi, field, initial, noise(seed, 20000+rid), plant, cbf)
            outcome = ('safe_deadlock' if row['any_deadlock'] else
                       'success' if row['success'] else 'other_timeout')
            label, _ = classify_timeout_trace(dict(
                max_speed=np.linalg.norm(trace['applied'].reshape(-1, 2, 2), axis=-1).max(axis=-1),
                goal_errors=np.linalg.norm(trace['positions_after']-np.asarray(goals), axis=-1)),
                outcome, plant.dt)
            n = row['steps']; pad = plant.max_steps-n
            padded = lambda values: np.concatenate([values, np.repeat(trace['positions_after'][-1][None], pad, axis=0)])
            risk = float(score(jnp.asarray(padded(trace['positions_before'])),
                               jnp.asarray(padded(trace['positions_after'])),
                               jnp.asarray(np.pad(trace['applied'], ((0, pad), (0, 0)))),
                               jnp.arange(plant.max_steps)<n, jnp.asarray(row['timeout'])))
            record = dict(row, rid=rid, seed=seed, role='prediction' if seed in prediction_seeds else 'outcome',
                          outcome=label, either_deadlock=label in ('safe_deadlock', 'stalled_deadlock'), risk=risk)
            records.append(record)
            np.savez_compressed(args.out/f'trace_{rid}_{seed}.npz', **trace)
            atomic_save(args.out/'records.json', records)
        print(json.dumps(dict(starts_completed=rid+1, rollouts=len(records),
                              elapsed=time.monotonic()-begun)), flush=True)
    atomic_save(args.out/'complete.json', dict(rollouts=len(records), elapsed=time.monotonic()-begun,
                                              independent_test_opened=False))


if __name__ == '__main__':
    main()

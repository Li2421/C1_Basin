"""Paired local gradient intervention, not shared-policy held-out efficacy."""
import argparse
import json
import time
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.train_deadlock_union import setup, noise, digest, source_hashes
from single_integrator.c1.rollout_deadlock_union import rollout
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.risk.deadlock_union import trajectory as old_risk
from single_integrator.c1.risk.reachability_union import trajectory as new_risk
from single_integrator.c1.train import observation
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', type=int, default=40)
    p.add_argument('--replay-budget-seconds', type=float, default=1800.)
    p.add_argument('--candidate', choices=('bounded', 'exact', 'ordered'), default='bounded')
    args = p.parse_args()
    candidate_risk = new_risk
    if args.candidate == 'exact':
        from single_integrator.c1.risk.exact_margin import trajectory as candidate_risk
    elif args.candidate == 'ordered':
        from single_integrator.c1.risk.ordered_guidance import trajectory as candidate_risk
    if args.out.exists() or args.cases < 2 or args.cases % 2:
        raise ValueError('new output and positive even case count required')
    params, field, plant, cbf, baseline = setup()
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required for this authorized experiment')
    sets_path = ROOT/'results/c1_deadlock_union/sets.json'
    calibration_path = ROOT/'results/c1_deadlock_union/baseline0_seed0/calibration.json'
    sets = json.loads(sets_path.read_text())
    pool = {x['rid']: x for x in sets['pools']['train']}
    calibration = json.loads(calibration_path.read_text())['episodes']
    chosen = []
    # Outcome stratification is only for diagnosis, never an estimated population rate.
    # Distinct initial states; first sorted entries, fixed before intervention results.
    used = set()
    for deadlocked in (True, False):
        candidates = sorted(calibration, key=lambda x: (x['rid'], x['noise_seed']))
        group = []
        for x in candidates:
            if bool(x['either_deadlock']) == deadlocked and x['rid'] not in used:
                group.append(dict(rid=x['rid'], noise_seed=x['noise_seed'],
                                  expected_deadlock=deadlocked, initial=pool[x['rid']]['initial']))
                used.add(x['rid'])
                if len(group) == args.cases//2:
                    break
        if len(group) != args.cases//2:
            raise ValueError('insufficient distinct development cases')
        chosen.extend(group)
    # Interleave deadlock and nondeadlock so a time cap cannot remove all controls.
    chosen = [x for pair in zip(chosen[:args.cases//2], chosen[args.cases//2:]) for x in pair]
    goals = jnp.asarray(GiveWayEnv(plant).goals)
    kwargs = dict(dt=plant.dt, max_speed=plant.max_speed, goal_tolerance=plant.goal_tolerance,
                  hold_seconds=plant.deadlock_hold_seconds, progress_window_seconds=plant.progress_window_seconds,
                  progress_epsilon=plant.progress_epsilon, speed_epsilon_fraction=plant.speed_epsilon_fraction)

    @jax.jit
    def scores(before, after, applied, alive, timeout):
        kw = dict(terminal_timeout=timeout, **kwargs)
        old = old_risk(before, after, applied, goals, alive, **kw)
        new = (candidate_risk(before, after, applied, goals, alive, **kw) if args.candidate != 'bounded'
               else candidate_risk(before, after, applied, goals, alive, margin_width=.5, tail_budget=.01, **kw))
        return jnp.array([old['J_live'], new['J_live']])

    def objective(phi, initial, draws, index):
        terms, trace = rollout(phi, field, initial, draws, plant, cbf)
        risks = scores(trace['before'], trace['after'], trace['applied'], trace['alive'], terms['timeout'])
        return risks[index], (risks, terms['steps'], terms['either_deadlock'])

    vg = jax.jit(jax.value_and_grad(objective, has_aux=True))
    direction_outputs = jax.jit(lambda phi, direction, obs, safe:
        jax.jvp(lambda v: field.correction(v, obs, safe), (phi,), (direction,))[1])
    args.out.mkdir(parents=True)
    atomic_save(args.out/'protocol.json', dict(
        scope='Local per-case phi gradients on stratified TRAIN cases; not training, test efficacy, or deployment candidate selection',
        cases=chosen, total_replays=9*args.cases, risks=['original_sum', args.candidate+'_union'],
        candidate=args.candidate, compatibility_note='risk_bounded column stores selected candidate score',
        bounded_parameters=dict(margin_width=.5, tail_budget=.01, temperature=.02),
        steps='Negative gradient scaled by first 5 s same-state residual JVP RMS: .001, .005, .01 m/s; positive .005 control',
        zero_gradient='Exact zero direction kept and reported, never replaced with heuristic',
        parameter_displacement_cap=.1, replay_budget_seconds=args.replay_budget_seconds,
        baseline_sha256=digest(baseline), sets_sha256=digest(sets_path), calibration_sha256=digest(calibration_path),
        sources={**source_hashes(), **{s:digest(ROOT/s) for s in (
            'scripts/probe_c1_gradient_closed_loop.py', 'single_integrator/c1/risk/reachability_union.py',
            'single_integrator/c1/risk/exact_margin.py',
            'single_integrator/c1/risk/ordered_guidance.py',
            'single_integrator/c1/risk/reachability_margin.py')}}, backend=jax.default_backend()))
    started = time.monotonic()
    replay_seconds = 0.
    records = []
    gradients = []

    def replay(phi, case, name, base_trace=None):
        nonlocal replay_seconds
        begin = time.monotonic()
        row, trace = execute(phi, field, np.asarray(case['initial']), noise(case['noise_seed'], case['rid']), plant, cbf)
        replay_seconds += time.monotonic()-begin
        outcome = 'safe_deadlock' if row['any_deadlock'] else ('success' if row['success'] else 'other_timeout')
        label, _ = classify_timeout_trace(dict(max_speed=np.linalg.norm(trace['applied'].reshape(-1,2,2), axis=-1).max(axis=1),
            goal_errors=np.linalg.norm(trace['positions_after']-np.asarray(goals), axis=-1)), outcome, plant.dt)
        n = row['steps']
        pad = lambda x: np.concatenate([x, np.repeat(x[-1:], plant.max_steps-n, axis=0)])
        values = np.asarray(scores(pad(trace['positions_before']), pad(trace['positions_after']),
            np.pad(trace['applied'], ((0,plant.max_steps-n),(0,0))), np.arange(plant.max_steps)<n, row['timeout']))
        rec = dict(rid=case['rid'], noise_seed=case['noise_seed'], variant=name,
                   label=label, risk_original=float(values[0]), risk_bounded=float(values[1]), **row)
        if base_trace is not None:
            early = min(100, len(trace['applied']), len(base_trace['applied']))
            diff = trace['applied'][:early]-base_trace['applied'][:early]
            rec['early_applied_delta_rms'] = float(np.sqrt(np.mean(diff**2)))
            rec['early_applied_delta_max'] = float(np.max(np.abs(diff)))
        records.append(rec)
        np.savez_compressed(args.out/f"{case['rid']}_{case['noise_seed']}_{name}.npz", **trace)
        atomic_save(args.out/'records.json', records)
        return rec, trace

    stopped = False
    for ci, case in enumerate(chosen):
        if replay_seconds >= args.replay_budget_seconds:
            stopped = True
            break
        base, trace = replay(params, case, 'Safety')
        actual_deadlock = base['any_deadlock'] or base['label'] == 'stalled_deadlock'
        if actual_deadlock != case['expected_deadlock']:
            raise AssertionError('Frozen calibration and authoritative baseline disagree')
        early = min(100, len(trace['applied']))
        velocity = np.concatenate([np.zeros((1,2,2)), trace['applied'][:-1].reshape(-1,2,2)])
        obs = observation(jnp.asarray(trace['positions_before'][:early]), jnp.asarray(velocity[:early]), goals)
        safe = jnp.asarray(trace['safe'][:early])
        for index, risk_name in enumerate(('original', 'bounded')):
            begin = time.monotonic()
            (value, aux), gradient = vg(params, jnp.asarray(case['initial']), noise(case['noise_seed'], case['rid']), jnp.asarray(index))
            norm = float(optax.global_norm(gradient))
            if not np.isfinite(norm):
                raise FloatingPointError('nonfinite gradient')
            if int(aux[1]) != base['steps'] or bool(aux[2]) != actual_deadlock:
                raise AssertionError('Differentiable and authoritative rollout disagree')
            direction = jax.tree_util.tree_map(lambda g: -g/max(norm, 1e-300), gradient)
            sensitivity = float(jnp.sqrt(jnp.mean(direction_outputs(params, direction, obs, safe)**2)))
            info = dict(rid=case['rid'], risk=risk_name, risk_value=float(value), gradient_norm=norm,
                        early_residual_sensitivity=sensitivity, seconds=time.monotonic()-begin)
            gradients.append(info)
            atomic_save(args.out/'gradients.json', gradients)
            for sign, target in ((1,.001),(1,.005),(1,.01),(-1,.005)):
                if replay_seconds >= args.replay_budget_seconds:
                    stopped = True
                    break
                alpha = min(target/max(sensitivity, 1e-300), .1) if norm > 0 else 0.
                candidate = jax.tree_util.tree_map(lambda x, d: x+sign*alpha*d, params, direction)
                name = f'{risk_name}_{"minus" if sign == 1 else "plus"}_{target:g}'
                rec, _ = replay(candidate, case, name, trace)
                rec.update(parameter_step=sign*alpha, target_early_residual_rms=target,
                           linearized_risk_change=-sign*alpha*norm,
                           risk_change=rec['risk_'+('original' if index == 0 else 'bounded')]-float(value))
                atomic_save(args.out/'records.json', records)
            if stopped:
                break
        atomic_save(args.out/'progress.json', dict(completed_cases=ci+1, total_cases=len(chosen),
            replays=len(records), replay_seconds=replay_seconds, elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(case=ci+1, replays=len(records), replay_seconds=round(replay_seconds,1),
                              elapsed_seconds=round(time.monotonic()-started,1))), flush=True)
        if stopped:
            break
    variants = sorted({r['variant'] for r in records})
    summary = {}
    for variant in variants:
        pairs = [(r, next(b for b in records if b['rid']==r['rid'] and b['variant']=='Safety'))
                 for r in records if r['variant']==variant]
        dead = lambda r: r['any_deadlock'] or r['label']=='stalled_deadlock'
        summary[variant] = dict(n=len(pairs), deadlocks=sum(dead(r) for r,b in pairs),
            successes=sum(r['success'] for r,b in pairs), ordinary_timeout=sum(r['label']=='other_timeout' for r,b in pairs),
            baseline_deadlocks=sum(dead(b) for r,b in pairs),
            deadlock_to_success=sum(dead(b) and r['success'] for r,b in pairs),
            deadlock_to_ordinary_timeout=sum(dead(b) and r['label']=='other_timeout' for r,b in pairs),
            new_deadlocks=sum(not dead(b) and dead(r) for r,b in pairs))
    atomic_save(args.out/'complete.json', dict(status='budget_stopped' if stopped else 'complete',
        scope='Stratified local intervention only; no shared-policy generalization or feasible C1 claim',
        replays=len(records), replay_seconds=replay_seconds, elapsed_seconds=time.monotonic()-started,
        variants=summary))


if __name__ == '__main__':
    main()

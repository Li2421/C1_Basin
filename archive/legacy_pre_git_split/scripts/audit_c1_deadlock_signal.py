"""Development-only audit of deadlock signal, masking and prospective ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.risk.completion_certificate import trajectory
from single_integrator.c1.risk.deadlock_primary import trajectory as primary
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace


def auc(rows, score, label):
    y = np.array([r[label] for r in rows], bool)
    if not y.any() or y.all():
        return None
    ranks = rankdata([r[score] for r in rows])
    return float((ranks[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cohort', choices=('development', 'historical'), default='development')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    jax.config.update('jax_enable_x64', True)
    root = ROOT/'results/c1_independent_distribution'
    config = json.loads((root/'sets.json').read_text())['environment']
    plant = Config(**config)
    goals = jnp.asarray(GiveWayEnv(plant).goals)
    kwargs = dict(dt=plant.dt, max_speed=plant.max_speed,
        goal_tolerance=plant.goal_tolerance, hold_seconds=plant.deadlock_hold_seconds,
        progress_window_seconds=plant.progress_window_seconds,
        progress_epsilon=plant.progress_epsilon, speed_epsilon_fraction=plant.speed_epsilon_fraction)

    @jax.jit
    def score(x, y, u, alive):
        r = trajectory(x, y, u, goals, alive, **kwargs)
        # Independent trajectory partials, not a policy BPTT gradient.
        def d(a, b, c):
            return trajectory(a, b, c, goals, alive, **kwargs)['deadlock_bound']
        grad = jax.grad(d, argnums=(0, 1, 2))(x, y, u)
        support = sum(jnp.sum(jnp.any(jnp.abs(g.reshape(len(g), -1)) > 1e-12, axis=1)) for g in grad)
        norm = jnp.sqrt(sum(jnp.sum(g*g) for g in grad))
        smooth = primary(x, y, u, goals, alive, **kwargs)
        def smooth_score(a, b, c):
            return primary(a, b, c, goals, alive, **kwargs)['J_live']
        smooth_grad = jax.grad(smooth_score, argnums=(0, 1, 2))(x, y, u)
        smooth_support = sum(jnp.sum(jnp.any(jnp.abs(g.reshape(len(g), -1)) > 1e-12, axis=1)) for g in smooth_grad)
        return r['J_live'], r['deadlock_bound'], r['timeout_bound'], support, norm, smooth['J_live'], smooth_support

    rows = []
    sources = [(seed, root/f'extra_validation_baseline{seed}', '*.npz', None) for seed in (0, 1)]
    if args.cohort == 'historical':
        base = ROOT/'results/c1_four_objectives_multiseed/evaluation'
        sources = [(0, base/'baseline', '4*.npz', 'Safety'),
                   (0, base/'Pg_seed0', '4*.npz', 'HistoricalPg')]
    for seed, folder, pattern, fixed_method in sources:
        for path in sorted(folder.glob(pattern)):
            with np.load(path) as z:
                n = len(z['positions_after'])
                pad = plant.max_steps-n
                x = np.concatenate([z['positions_before'], np.repeat(z['positions_after'][-1:], pad, axis=0)])
                y = np.concatenate([z['positions_after'], np.repeat(z['positions_after'][-1:], pad, axis=0)])
                u = np.concatenate([z['applied'], np.zeros((pad, 4))])
                success, dead = np.asarray(z['success']), np.asarray(z['deadlock'])
            vals = list(map(float, score(jnp.asarray(x), jnp.asarray(y), jnp.asarray(u), jnp.arange(plant.max_steps)<n)))
            joint, d, t, support, norm, smoothed, smooth_support = vals
            outcome = 'safe_deadlock' if dead.any() else ('success' if success.any() else 'other_timeout')
            label, _ = classify_timeout_trace(dict(
                max_speed=np.linalg.norm(u[:n].reshape(n,2,2),axis=-1).max(axis=-1),
                goal_errors=np.linalg.norm(y[:n]-np.asarray(goals),axis=-1)), outcome, plant.dt)
            k = round(20/plant.dt)
            prefix = score(jnp.asarray(x[:k]), jnp.asarray(y[:k]), jnp.asarray(u[:k]), jnp.arange(k)<n)
            # Only a deadlock score is used prospectively. Full-episode outcome
            # and completion gate must never be included in the prefix score.
            at_risk = n > k and not dead[:k].any() and not success[:k].any()
            row = dict(baseline_seed=seed, case=path.stem,
                method=fixed_method or path.stem.split('_')[0], any_deadlock=bool(dead.any()),
                six_class_outcome=label, stalled_deadlock=label=='stalled_deadlock',
                either_deadlock=label in ('safe_deadlock','stalled_deadlock'),
                timeout=bool(not success.any() and not dead.any()), success=bool(success.any()),
                joint=joint, deadlock_score=d, timeout_score=t,
                deadline_masks_deadlock=bool(t>d), deadlock_partial_support=int(support),
                deadlock_partial_norm=norm, prefix20_deadlock_score=float(prefix[1]),
                smooth_deadlock_score=smoothed, smooth_partial_support=int(smooth_support),
                prefix20_smooth_score=float(prefix[5]),
                at_risk_at20=bool(at_risk), future_deadlock=bool(dead[k:].any()),
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if row['any_deadlock'] and d < 1-1e-8:
                raise AssertionError(('deadlock upper bound violation', row))
            rows.append(row)
    summary = {}
    for name in sorted(set(r['method'] for r in rows)):
        group = [r for r in rows if r['method']==name]
        future = [r for r in group if r['at_risk_at20']]
        dead = [r for r in group if r['any_deadlock']]
        summary[name] = dict(episodes=len(group), deadlocks=len(dead),
            deadlock_auc=auc(group, 'deadlock_score', 'any_deadlock'),
            joint_auc_for_deadlock=auc(group, 'joint', 'any_deadlock'),
            deadlocks_masked_by_timeout=sum(r['deadline_masks_deadlock'] for r in dead),
            deadlock_partial_support=[r['deadlock_partial_support'] for r in dead],
            smooth_deadlock_auc=auc(group, 'smooth_deadlock_score', 'any_deadlock'),
            smooth_partial_support=[r['smooth_partial_support'] for r in dead],
            smooth_mean=float(np.mean([r['smooth_deadlock_score'] for r in group])),
            stalled_deadlocks=sum(r['stalled_deadlock'] for r in group),
            stalled_below_one=sum(r['stalled_deadlock'] and r['smooth_deadlock_score'] < 1 for r in group),
            stalled_zero_risk=sum(r['stalled_deadlock'] and r['smooth_deadlock_score'] == 0 for r in group),
            either_deadlock_auc=auc(group, 'smooth_deadlock_score', 'either_deadlock'),
            prospective20_at_risk=len(future), prospective20_events=sum(r['future_deadlock'] for r in future),
            prospective20_smooth_auc=auc(future, 'prefix20_smooth_score', 'future_deadlock'),
            prospective20_auc=auc(future, 'prefix20_deadlock_score', 'future_deadlock'))
    result = dict(scope='Already-seen development validation only; not independent efficacy',
        cohort=args.cohort, backend=jax.default_backend(), summary=summary, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.out, result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

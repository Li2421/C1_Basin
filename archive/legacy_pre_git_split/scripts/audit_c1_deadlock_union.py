"""Check the two original deadlock endpoints on already-seen saved traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.risk.deadlock_union import trajectory
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import Config, GiveWayEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    jax.config.update('jax_enable_x64', True)
    plant = Config(**json.loads((ROOT/'results/c1_independent_distribution/sets.json').read_text())['environment'])
    goals = jnp.asarray(GiveWayEnv(plant).goals)
    kwargs = dict(dt=plant.dt, max_speed=plant.max_speed,
        goal_tolerance=plant.goal_tolerance, hold_seconds=plant.deadlock_hold_seconds,
        progress_window_seconds=plant.progress_window_seconds,
        progress_epsilon=plant.progress_epsilon, speed_epsilon_fraction=plant.speed_epsilon_fraction)

    @jax.jit
    def score(x, y, u, alive, timeout):
        risk = trajectory(x, y, u, goals, alive, terminal_timeout=timeout, **kwargs)
        # Partial derivatives of the saved trajectory, NOT policy gradients.
        fn = lambda a,b,c: trajectory(a,b,c,goals,alive,terminal_timeout=timeout,**kwargs)['J_live']
        grad = jax.grad(fn, argnums=(0,1,2))(x,y,u)
        return dict(J_live=risk['J_live'], strict_bound=risk['strict_bound'],
            stalled_bound=risk['stalled_bound'], hard_risk=risk['hard_risk'],
            partial_norm=jnp.sqrt(sum(jnp.sum(g*g) for g in grad)),
            partial_support=sum(jnp.sum(jnp.any(jnp.abs(g.reshape(len(g),-1))>1e-12,axis=1)) for g in grad))

    rows = []
    root = ROOT/'results/c1_four_objectives_multiseed/evaluation'
    for method, folder in [('Safety', 'baseline'), ('HistoricalPg', 'Pg_seed0')]:
        paths = sorted((root/folder).glob('4*.npz'))
        if not paths:
            raise ValueError('Missing predeclared historical traces')
        for path in paths:
            with np.load(path) as z:
                x,y,u = (z[k] for k in ('positions_before','positions_after','applied'))
                dead, success = bool(z['deadlock'].any()), bool(z['success'].any())
                if dead and success:
                    raise ValueError('Expected original first-event traces')
            n = len(y)
            if not 0 < n <= plant.max_steps:
                raise ValueError('Invalid trace length')
            timeout = not dead and not success
            outcome = 'safe_deadlock' if dead else ('success' if success else 'other_timeout')
            label,_ = classify_timeout_trace(dict(max_speed=np.linalg.norm(u.reshape(n,2,2),axis=-1).max(axis=1),
                goal_errors=np.linalg.norm(y-np.asarray(goals),axis=-1)),outcome,plant.dt)
            pad = np.repeat(y[-1:],plant.max_steps-n,axis=0)
            risk = {k:float(v) for k,v in score(jnp.asarray(np.concatenate([x,pad])),
                jnp.asarray(np.concatenate([y,pad])),jnp.asarray(np.concatenate([u,np.zeros((plant.max_steps-n,4))])),
                jnp.arange(plant.max_steps)<n, jnp.asarray(timeout)).items()}
            if not all(np.isfinite(v) for v in risk.values()):
                raise FloatingPointError(path)
            positive = label in ('safe_deadlock','stalled_deadlock')
            if positive and risk['J_live'] < 1-1e-8:
                raise AssertionError(('union certificate failed',path,risk))
            if not timeout and risk['stalled_bound'] != 0:
                raise AssertionError('Terminal stall leaked into a non-timeout')
            rows.append(dict(method=method,case=path.stem,outcome=label,
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**risk))
    summary = {}
    for method in ('Safety','HistoricalPg'):
        summary[method] = {}
        for label in sorted({r['outcome'] for r in rows if r['method']==method}):
            group = [r for r in rows if r['method']==method and r['outcome']==label]
            summary[method][label] = dict(n=len(group),
                risk_mean=float(np.mean([r['J_live'] for r in group])),
                risk_min=min(r['J_live'] for r in group),
                strict_below_one=sum(r['strict_bound']<1 for r in group),
                union_below_one=sum(r['J_live']<1 for r in group),
                zero_partial_gradient=sum(r['partial_norm']==0 for r in group))
    sources = [Path(__file__), ROOT/'single_integrator/c1/risk/deadlock_union.py',
        ROOT/'single_integrator/c1/risk/deadlock_primary.py',
        ROOT/'single_integrator/diagnostics/stalled_outcomes.py']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(args.out,dict(scope='Historical seen traces; retrospective certificate coverage, not prediction or efficacy',
        source_sha256={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        summary=summary, rows=rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()

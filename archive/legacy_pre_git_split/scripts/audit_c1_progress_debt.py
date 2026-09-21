"""Check debt certificates on all 384 frozen traces, without fitting anything."""
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
from single_integrator.c1.risk.progress_debt import trajectory
from single_integrator.environment import Config, GiveWayEnv
from scripts.summarize_c1_risk_outcomes import auc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--risk', choices=['debt', 'temporal'], default='debt')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    jax.config.update('jax_enable_x64', True)
    goals = jnp.asarray(GiveWayEnv(Config(corridor_half_length=1.3)).goals)
    from single_integrator.c1.risk.temporal_certificate import trajectory as temporal
    score = jax.jit(lambda before, after, applied, alive:
        temporal(before, after, applied, goals, alive) if args.risk == 'temporal'
        else trajectory(before, after, goals, alive))
    source = ROOT/'results/c1_risk_deadlock_correspondence/rows.json'
    rows = json.loads(source.read_text())
    assert len(rows) == 384
    output = []
    for row in rows:
        path = ROOT/f'results/c1_four_objectives_multiseed/evaluation/{row["model"]}/{row["rid"]}_{row["execution_seed"]}.npz'
        with np.load(path) as z:
            before, after, success = z['positions_before'], z['positions_after'], z['success']
            n = len(after)
            applied = np.concatenate([z['applied'].reshape(n, 2, 2), np.zeros((850-n, 2, 2))])
            assert n == 850 or success[-1]
            before = np.concatenate([before, np.repeat(after[-1][None], 850-n, axis=0)])
            after = np.concatenate([after, np.repeat(after[-1][None], 850-n, axis=0)])
        alive = np.arange(850) < n
        result = score(jnp.asarray(before), jnp.asarray(after), jnp.asarray(applied), jnp.asarray(alive))
        scalars = {k: float(result[k]) for k in ['J_live', 'timeout_bound', 'deadlock_bound', 'deadline_slack']}
        assert all(np.isfinite(v) for v in scalars.values())
        if row['failure']:
            assert scalars['timeout_bound'] >= 1-1e-9, (path, scalars)
        if row['any_deadlock']:
            assert scalars['deadlock_bound'] >= 1-1e-9, (path, scalars)
        output.append({k: row[k] for k in ['model', 'rid', 'execution_seed', 'success', 'deadlock', 'any_deadlock', 'timeout', 'failure']} | scalars)
    summary = {}
    for model in sorted({r['model'] for r in output}):
        rr = [r for r in output if r['model'] == model]
        result = {}
        for key in ['J_live', 'timeout_bound', 'deadlock_bound']:
            result[key] = dict(mean=float(np.mean([r[key] for r in rr])),
                failure_auc=auc([r['failure'] for r in rr], [r[key] for r in rr]),
                deadlock_auc=auc([r['any_deadlock'] for r in rr], [r[key] for r in rr]),
                distributions={label: dict(n=sum(r[label] for r in rr),
                    mean=float(np.mean([r[key] for r in rr if r[label]])),
                    min=float(np.min([r[key] for r in rr if r[label]])),
                    max=float(np.max([r[key] for r in rr if r[label]])))
                    for label in ['success', 'deadlock', 'timeout']})
        summary[model] = result
    files = [Path(__file__), ROOT/'single_integrator/c1/risk/progress_debt.py', source, ROOT/'single_integrator/c1/risk/temporal_certificate.py']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(scope='Already-seen trajectories; certificate audit, not prediction or learned efficacy',
        source_hashes={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        risk=args.risk, episodes=len(output), certificate_violations=0, summary=summary, rows=output), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


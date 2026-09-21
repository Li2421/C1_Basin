"""CPU-only full-horizon boundary equivalence; not efficacy or held-out testing."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup, noise, digest, source_hashes
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.early_intervention import execute_prefix
from single_integrator.c1.training.persistence import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('use a new output directory')
    if jax.default_backend() != 'cpu':
        raise RuntimeError('this check must not use GPU')
    reference, field, plant, cbf, baseline = setup()
    checkpoint = ROOT / 'results/c1_deadlock_union/baseline0_seed0/best_feasible.pkl'
    saved = pickle.loads(checkpoint.read_bytes())
    if saved['config']['baseline_sha256'] != digest(baseline):
        raise ValueError('checkpoint baseline mismatch')
    alternative = jax.tree_util.tree_map(jnp.asarray, saved['params'])
    sets = ROOT / 'results/c1_deadlock_union/sets.json'
    item = json.loads(sets.read_text())['pools']['train'][0]
    seed = 8241801
    draws = noise(seed, item['rid'])
    args.out.mkdir(parents=True)
    sources = source_hashes()
    for name in ['single_integrator/c1/evaluate_deadlock_union.py',
                 'single_integrator/c1/early_intervention.py',
                 'scripts/check_c1_prefix_physics.py']:
        sources[name] = digest(ROOT / name)
    atomic_save(args.out / 'protocol.json', dict(
        scope='development boundary equivalence only', backend='cpu',
        checkpoint_sha256=digest(checkpoint), sets_sha256=digest(sets),
        source_hashes=sources, rid=item['rid'], noise_seed=seed,
        environment=plant.to_dict(), cbf=cbf.to_dict(), rtol=1e-10, atol=1e-12))
    begun = time.monotonic()
    results = {}
    for name, params, prefix in [('reference', reference, None),
                                 ('prefix_zero', reference, 0),
                                 ('alternative', alternative, None),
                                 ('prefix_full', alternative, plant.max_steps)]:
        if prefix is None:
            row, trace = execute(params, field, item['initial'], draws, plant, cbf)
        else:
            row, trace = execute_prefix(reference, alternative, field, item['initial'],
                                       draws, plant, cbf, prefix_steps=prefix)
        results[name] = row, trace
        atomic_save(args.out / (name + '.json'), row)
        np.savez_compressed(args.out / (name + '.npz'), **trace)
        print(json.dumps(dict(completed=name, steps=row['steps'],
                              elapsed=time.monotonic()-begun)), flush=True)
    comparisons = {}
    for a, b in [('reference', 'prefix_zero'), ('alternative', 'prefix_full')]:
        row_a, trace_a = results[a]
        row_b, trace_b = results[b]
        assert row_a == {k: v for k, v in row_b.items() if k != 'intervention'}
        errors = {}
        for key in trace_a:
            np.testing.assert_allclose(trace_a[key], trace_b[key], rtol=1e-10, atol=1e-12)
            errors[key] = float(np.max(np.abs(trace_a[key].astype(float)-trace_b[key].astype(float))))
        comparisons[a + '_vs_' + b] = errors
    atomic_save(args.out / 'complete.json', dict(
        passed=True, comparisons=comparisons, elapsed_seconds=time.monotonic()-begun,
        efficacy_demonstrated=False, independent_test_opened=False))


if __name__ == '__main__':
    main()

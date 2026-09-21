"""Freeze disjoint wide-start calibration/selection sets before v2 training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.environment import Config, GiveWayEnv


def wide_starts(rng, size):
    starts = np.empty((size, 2, 2), dtype=np.float64)
    starts[:, :, 0] = rng.uniform(.55, 1.05, (size, 2)) * [-1, 1]
    starts[:, :, 1] = rng.uniform(-.025, .025, (size, 2))
    return starts


def check_disjoint(groups):
    seen = set()
    for name, starts in groups:
        rows = [tuple(np.asarray(row, dtype=np.float64).ravel()) for row in starts]
        if len(set(rows)) != len(rows) or seen.intersection(rows):
            raise ValueError(f'duplicate or overlapping initial positions in {name}')
        seen.update(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--size', type=int, default=64, help='Episodes in each set.')
    parser.add_argument('--seed', type=int, default=2026091301)
    args = parser.parse_args()
    if args.size < 1:
        raise ValueError('size must be positive')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('out-dir must be empty; frozen sets cannot be overwritten')
    environment_path = ROOT / 'datasets/give_way_si_short_v1/environment.json'
    plant = Config(**json.loads(environment_path.read_text())['evaluation_environment'])
    children = np.random.SeedSequence(args.seed).spawn(4)
    groups = [(name, wide_starts(np.random.default_rng(children[i]), args.size))
              for i, name in enumerate(('calibration', 'validation'))]
    # Compare against all original corpus starts and the locked benchmark.
    corpus = []
    for path in sorted((ROOT / 'datasets/give_way_si_short_v1/raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            corpus.append(data['initial_positions'])
    corpus = np.unique(np.asarray(corpus).reshape(-1, 4), axis=0).reshape(-1, 2, 2)
    suite_path = ROOT / 'baseline_309_314/planning/wide_initial_states_200.npz'
    with np.load(suite_path, allow_pickle=False) as data:
        benchmark = data['test_initial_positions']
    # Historical groups may overlap each other; only new sets must be disjoint.
    historical = np.unique(np.concatenate((corpus, benchmark)).reshape(-1, 4), axis=0).reshape(-1, 2, 2)
    check_disjoint([('historical', historical), *groups])
    for _, starts in groups:
        for start in starts:
            GiveWayEnv(plant).reset(start)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol='v2_wide_frozen_v1', seed=args.seed, size_per_set=args.size,
                    horizon=plant.max_steps, environment=plant.to_dict(),
                    numpy_version=np.__version__,
                    distribution='independent |x| U[0.55,1.05], y U[-0.025,0.025]; fixed agent directions',
                    selection='unconditional draws; no policy outcome selection',
                    validation_role='checkpoint selection; not an untouched final test',
                    calibration_role='wide-start constraint calibration; differs from risk-enriched training batches',
                    independence='exact starts disjoint from each other, original corpus, and fixed 200-case benchmark',
                    benchmark_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(), files={})
    for i, (name, starts) in enumerate(groups):
        noise = np.random.default_rng(children[i + 2]).standard_normal(
            (args.size, plant.max_steps, 4), dtype=np.float32)
        path = args.out_dir / f'{name}.npz'
        np.savez_compressed(path, initial_positions=starts, noise=noise,
                            metadata_json=json.dumps(dict(kind=name, **{k: v for k, v in manifest.items() if k != 'files'})))
        manifest['files'][name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

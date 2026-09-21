"""Freeze the original benchmark's independent-x distribution before training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.environment import Config, GiveWayEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    template = json.loads((ROOT/'results/c1_completion_development/sets.json').read_text())
    counts = [('train', 32), ('validation', 16), ('test', 256)]
    rng = np.random.default_rng(2026091631)
    n = sum(count for _, count in counts)
    magnitudes = rng.uniform(.55, 1.05, (n, 2))
    ys = rng.uniform(-.025, .025, (n, 2))
    positions = np.stack([np.c_[-magnitudes[:, 0], ys[:, 0]],
                          np.c_[magnitudes[:, 1], ys[:, 1]]], axis=1)
    env = GiveWayEnv(Config(**template['environment']))
    for x in positions:
        env.reset(x)
    assert np.unique(positions.reshape(n, 4), axis=0).shape[0] == n
    assert not np.any(positions[:, 0, 0] == -positions[:, 1, 0])
    pools, offset = {}, 0
    for name, count in counts:
        pools[name] = [dict(rid=10000+i, initial=positions[i].tolist()) for i in range(offset, offset+count)]
        offset += count
    # Exclude the original benchmark and both earlier frozen development/test
    # splits exactly. This checks state overlap, not test outcomes.
    prior = []
    suite = ROOT/'baseline_309_314/planning/wide_initial_states_200.npz'
    with np.load(suite) as z:
        prior.extend(z['val_initial_positions']); prior.extend(z['test_initial_positions'])
    for path in [ROOT/'results/c1_certificate_development/sets.json', ROOT/'results/c1_v3_smoke/sets.json']:
        data = json.loads(path.read_text())
        prior.extend(item['initial'] for pool in data['pools'].values() for item in pool)
    separation = float(np.min(np.max(np.abs(positions[:, None]-np.asarray(prior)[None]), axis=(2, 3))))
    assert separation > 0
    data = dict(version=template['version'], data_seed=2026091631,
        distribution='independent |x_0|, |x_1| U(.55,1.05); independent y U(-.025,.025)',
        environment=template['environment'], prefix_steps=0, score_steps=[0, 850],
        calibration_noise_seeds=[71101, 71102], validation_noise_seeds=[71201, 71202],
        test_noise_seeds=[71301, 71302], min_distance_from_checked_prior_starts=separation,
        pools=pools, purpose='Original wide-distribution match; final test remains sealed',
        original_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2)+'\n')
    print(dict(path=str(args.out), sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), counts=dict(counts)))


if __name__ == '__main__':
    main()

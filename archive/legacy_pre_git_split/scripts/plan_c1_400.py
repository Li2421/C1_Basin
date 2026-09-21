"""Run C1 planning for both selected Flow-BC seeds on the fixed 400-case protocol."""
import argparse
import json
import hashlib
import os
import pickle
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace, OUTCOMES, PROTOCOL


def validate_residual_pair(paths):
    """Reject swapped/duplicated baseline seeds and mixed risk definitions."""
    metadata = []
    for seed, path in enumerate(paths):
        with Path(path).open('rb') as handle:
            saved = pickle.load(handle)
        meta = saved['metadata']
        expected = ROOT / f'baseline_309_314/checkpoints/seed{seed}/ckpt_0025000.pkl'
        digest = hashlib.sha256(expected.read_bytes()).hexdigest()
        if meta.get('method') != 'c1_v0' or meta.get('baseline_checkpoint_sha256') != digest:
            raise ValueError(f'residual-seed{seed} must use frozen baseline seed {seed}')
        metadata.append(meta)
    for key in ('risk_version', 'risk', 'environment', 'cbf', 'architecture'):
        if metadata[0].get(key) != metadata[1].get(key):
            raise ValueError(f'the two C1 arms disagree on {key}')
    for key in ('updates', 'batch_size', 'horizon', 'lr', 'dual_lr', 'epsilon_fraction',
                'risk_start_fraction', 'start_distribution', 'validation_sha256', 'validation_every'):
        if metadata[0].get('training', {}).get(key) != metadata[1].get('training', {}).get(key):
            raise ValueError(f'the two C1 arms disagree on training {key}')
    if metadata[0].get('training', {}).get('calibration', {}).get('sha256') != metadata[1].get('training', {}).get('calibration', {}).get('sha256'):
        raise ValueError('the two C1 arms disagree on calibration set')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual-seed0', type=Path, required=True)
    parser.add_argument('--residual-seed1', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    validate_residual_pair((args.residual_seed0, args.residual_seed1))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('out-dir must be empty')
    args.out_dir.mkdir(parents=True)
    suite = ROOT / 'baseline_309_314/planning/wide_initial_states_200.npz'
    reports = []
    classified = []
    for seed, residual in enumerate((args.residual_seed0, args.residual_seed1)):
        folder = args.out_dir / f'seed{seed}_c1'
        c1_python = Path(os.environ.get('C1_PYTHON', str(ROOT / '.venv-c1/bin/python')))
        subprocess.run([str(c1_python), '-m', 'single_integrator.c1.evaluate', '--residual', str(residual),
                        '--out-dir', str(folder), '--initial-states', str(suite), '--split', 'test', '--seed', '42'],
                       cwd=ROOT, check=True)
        reports.append(json.loads((folder/'summary.json').read_text()))
        ids = [item['rollout_id'] for item in reports[-1]['rollouts']]
        if sorted(ids) != list(range(200)):
            raise ValueError('formal aggregation requires all 200 distinct cases per baseline seed')
        for summary in reports[-1]['rollouts']:
            rid = summary['rollout_id']
            with np.load(folder/f'rollout_{rid:04d}.npz', allow_pickle=False) as trace:
                label, details = classify_timeout_trace(trace, summary['outcome'],
                    json.loads((folder/'config.json').read_text())['environment']['dt'])
            classified.append(dict(seed=seed, rollout_id=rid, original_outcome=summary['outcome'],
                                   outcome=label, **details))
    arms = [report['aggregate'] for report in reports]
    aggregate = {label + '_count': sum(item['outcome'] == label for item in classified)
                 for label in OUTCOMES}
    aggregate['n_rollouts'] = 400
    (args.out_dir/'stalled_outcomes_v3.json').write_text(json.dumps(dict(protocol=PROTOCOL, rows=classified), indent=2)+'\n')
    (args.out_dir/'summary.json').write_text(json.dumps(dict(method='c1_v0', outcome_protocol=PROTOCOL,
        per_seed_first_event=arms, per_seed=[{label+'_count':sum(r['seed']==seed and r['outcome']==label for r in classified)
                                            for label in OUTCOMES} for seed in (0,1)], aggregate=aggregate,
        reference=dict(mac_only_success=309, safety_success=314, n_rollouts=400)), indent=2)+'\n')


if __name__ == '__main__':
    main()

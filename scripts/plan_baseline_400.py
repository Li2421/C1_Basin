"""Reproduce the selected 200 starts x 2 frozen checkpoints, two arms."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    from single_integrator.diagnostics.stalled_outcomes import reclassify_evaluation, OUTCOMES
    selected = ROOT / 'baseline_309_314'
    hashes = ['8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32',
              '3f58b12266d19358a8ef5673e3e4018c1248c526de7d71877218ea176bd53383']
    suite = selected / 'planning/wide_initial_states_200.npz'
    assert hashlib.sha256(suite.read_bytes()).hexdigest() == '30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb'
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Output directory must be empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    totals = {arm: dict.fromkeys(OUTCOMES, 0) for arm in ('mac_only', 'mac_cbf')}
    reports = []
    for seed in (0, 1):
        checkpoint = selected / f'checkpoints/seed{seed}/ckpt_0025000.pkl'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == hashes[seed]
        folder = args.out_dir.resolve() / f'seed{seed}_paired'
        subprocess.run([sys.executable, '-m', 'single_integrator.evaluate_cbf',
                        '--checkpoint', str(checkpoint), '--dataset', str(ROOT/'datasets/give_way_si_short_v1'),
                        '--initial_states', str(suite), '--split', 'test', '--seed', '42',
                        '--n_rollouts', '200', '--out_dir', str(folder)], cwd=ROOT, check=True)
        report = reclassify_evaluation(folder)
        reports.append(report)
        for arm in totals:
            for label in OUTCOMES:
                totals[arm][label] += report['arms'][arm]['outcome_counts'][label]
    (args.out_dir/'stalled_outcomes_v3.json').write_text(json.dumps(dict(reports=reports, totals=totals), indent=2)+'\n')


if __name__ == '__main__':
    main()

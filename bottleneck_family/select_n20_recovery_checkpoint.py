"""Rank N=20 recovery snapshots using only the declared non-opposing DEV screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bottleneck_family.select_n20_checkpoint import MODES


def run(root: Path, train: Path, output: Path) -> dict:
    losses = {int(row['step']): float(row['fixed_dev_loss'])
              for line in (train / 'metrics.jsonl').read_text().splitlines()
              if (row := json.loads(line))['step'] > 0}
    candidates = []
    for step in range(1000, 8001, 1000):
        checkpoint = train / f'snapshot_{step:07d}.pkl'
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        folder = root / f'n20_select_v5_s{step}'
        groups = {}
        for mode in MODES:
            rows = []
            for case in range(4):
                path = folder / f'{mode}_{case:04d}/summary.json'
                summary = json.loads(path.read_text())
                if (summary['N'] != 20 or summary['mode'] != mode
                        or summary['checkpoint_sha256'] != sha
                        or summary['split'] != 'dev'
                        or summary['fresh_seed'] != 162020
                        or summary['fresh_start'] != case
                        or not summary['goal_stop']
                        or len(summary['rollouts']) != 1):
                    raise ValueError(f'screen provenance mismatch: {path}')
                rows.append(summary['rollouts'][0])
            groups[mode] = rows
        all_rows = [row for rows in groups.values() for row in rows]
        per_mode = {mode: dict(runs=4, success=sum(bool(r['success']) for r in rows),
                               collision=sum(bool(r['collision']) for r in rows),
                               numerical=sum(r['numerical_error'] is not None for r in rows),
                               mean_goal_occupancy=sum(r['individual_goal_fraction'] for r in rows)/4)
                    for mode, rows in groups.items()}
        collision = sum(bool(r['collision']) for r in all_rows)
        numerical = sum(r['numerical_error'] is not None for r in all_rows)
        worst = min(v['success'] for v in per_mode.values())
        total = sum(v['success'] for v in per_mode.values())
        occupancy = sum(r['individual_goal_fraction'] for r in all_rows)/32
        correction = sum(r['mean_projection_norm'] for r in all_rows)/32
        rank = (int(collision + numerical == 0), worst, total,
                occupancy, -correction, -losses[step])
        candidates.append(dict(step=step, checkpoint=str(checkpoint), sha256=sha,
                               per_mode=per_mode, total_success=total,
                               worst_mode_success=worst, collision=collision,
                               numerical=numerical, mean_goal_occupancy=occupancy,
                               mean_projection_norm=correction,
                               offline_dev_loss=losses[step], rank=rank))
    candidates.sort(key=lambda row: row['rank'], reverse=True)
    report = dict(schema='gap1_n20_recovery_checkpoint_selection_v1',
                  selection_distribution='fresh non-opposing DEV seed 162020; four states per mode',
                  no_opposing_test_used=True, chosen=candidates[0], candidates=candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.train, args.output)
    print(json.dumps([{key: row[key] for key in
                       ('step', 'total_success', 'worst_mode_success', 'collision',
                        'numerical', 'mean_goal_occupancy', 'offline_dev_loss')}
                      for row in report['candidates']], indent=2))


if __name__ == '__main__':
    main()

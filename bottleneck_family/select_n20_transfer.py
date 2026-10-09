"""Rank N=20 transfer snapshots using only non-opposing DEV controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODES = ('single_even', 'single_odd', 'one_way', 'one_way_mirror')


def run(root: Path, train: Path, output: Path, *, steps: tuple[int, ...],
        cases: int) -> dict:
    if not steps or cases not in (2, 4):
        raise ValueError('nonempty snapshot list and two or four DEV cases required')
    candidates = []
    for step in steps:
        checkpoint = train / f'snapshot_{step:07d}.pkl'
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        groups = {}
        for mode in MODES:
            rows = []
            for case in range(cases):
                path = root / f'n20_select_v10_s{step}' / f'{mode}_{case:04d}/summary.json'
                summary = json.loads(path.read_text())
                if (summary['N'] != 20 or summary['mode'] != mode
                        or summary['checkpoint_sha256'] != sha
                        or summary['split'] != 'dev'
                        or summary['observation_version'] != 'competence_v2'
                        or summary['fresh_seed'] != 162020
                        or summary['fresh_start'] != case
                        or summary['fresh_count'] != 1
                        or summary['samples_per_step'] != 1
                        or not summary['goal_stop']
                        or len(summary['rollouts']) != 1):
                    raise ValueError(f'screen provenance mismatch: {path}')
                rows.append(summary['rollouts'][0])
            groups[mode] = rows
        all_rows = [row for rows in groups.values() for row in rows]
        per_mode = {mode: dict(runs=cases,
                               success=sum(bool(r['success']) for r in rows),
                               collision=sum(bool(r['collision']) for r in rows),
                               numerical=sum(r['numerical_error'] is not None for r in rows),
                               mean_goal_occupancy=sum(r['individual_goal_fraction']
                                                       for r in rows) / cases)
                    for mode, rows in groups.items()}
        collision = sum(bool(r['collision']) for r in all_rows)
        numerical = sum(r['numerical_error'] is not None for r in all_rows)
        worst = min(v['success'] for v in per_mode.values())
        total = sum(v['success'] for v in per_mode.values())
        correction = sum(r['mean_projection_norm'] for r in all_rows) / len(all_rows)
        rank = (int(collision + numerical == 0), worst, total,
                -correction, -step)
        candidates.append(dict(step=step, checkpoint=str(checkpoint), sha256=sha,
                               per_mode=per_mode, total_success=total,
                               worst_mode_success=worst, collision=collision,
                               numerical=numerical, mean_projection_norm=correction,
                               rank=rank))
    candidates.sort(key=lambda row: row['rank'], reverse=True)
    report = dict(schema='gap1_n20_transfer_checkpoint_selection_v1',
                  selection_distribution=f'non-opposing DEV seed 162020; {cases} cases per mode',
                  no_opposing_test_used=True, chosen=candidates[0],
                  candidates=candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    parser.add_argument('--cases', type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    report = run(args.root, args.train, args.output,
                 steps=tuple(args.steps), cases=args.cases)
    print(json.dumps([dict(step=row['step'], success=row['total_success'],
                           worst=row['worst_mode_success'],
                           correction=row['mean_projection_norm'])
                      for row in report['candidates']], indent=2))


if __name__ == '__main__':
    main()

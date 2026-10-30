"""Audit every post-freeze N=20 opposing TEST state with the existing rubric."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_scale_opposing import audit_trace


def run(root: Path, checkpoint: Path, output: Path, *, cases: int = 24,
        fresh_seed: int = 2026102012) -> dict:
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    for case in range(cases):
        folder = root / f'case_{case:04d}'
        summary = json.loads((folder / 'summary.json').read_text())
        if (summary['N'] != 20 or summary['mode'] != 'opposing'
                or summary['checkpoint_sha256'] != sha
                or summary['split'] != 'test'
                or summary['observation_version'] != 'competence_v2'
                or summary['fresh_seed'] != fresh_seed
                or summary['fresh_start'] != case
                or summary['fresh_count'] != 1
                or summary['samples_per_step'] != 1
                or summary['latent_mode'] != 'per_step'
                or not summary['goal_stop']
                or len(summary['rollouts']) != 1):
            raise ValueError(f'opposing TEST provenance mismatch: {folder}')
        traces = list((folder / 'traces').glob('*.npz'))
        if len(traces) != 1:
            raise ValueError(f'exactly one full simulator trace required: {folder}')
        row = audit_trace(traces[0])
        result = summary['rollouts'][0]
        if (row['steps'] != result['steps']
                or row['termination'] != result['termination']
                or row['collision'] != result['collision']):
            raise ValueError(f'trace/summary mismatch: {folder}')
        row['case'] = case
        row['trace'] = str(traces[0])
        rows.append(row)
    labels = sorted(set(row['screening_label'] for row in rows))
    counts = {label:sum(row['screening_label'] == label for row in rows)
              for label in labels}
    candidates = [r for r in rows if r['screening_label'] ==
                  'clean_bounded_gridlock_candidate']
    with_interaction = [r for r in rows if r['interaction_step'] is not None]
    result = dict(schema='gap1_n20_frozen_opposing_audit_v1',
                  frozen_checkpoint=str(checkpoint),
                  frozen_checkpoint_sha256=sha,
                  test_distribution=f'fresh TEST seed {fresh_seed}; {cases} states',
                  cases=cases,counts=counts,
                  candidate_gridlock_count=len(candidates),
                  candidate_with_partial_goal_occupancy=sum(
                      0 < r['final_goal_fraction'] < 1 for r in candidates),
                  interaction_cases=len(with_interaction),
                  mean_pre_interaction_progress_rate=float(np.mean([
                      r['pre_interaction_progress_rate'] for r in with_interaction]))
                  if with_interaction else None,
                  mean_post_interaction_progress_rate=float(np.mean([
                      r['post_interaction_progress_rate'] for r in with_interaction]))
                  if with_interaction else None,
                  mean_tail_goal_progress_per_agent=float(np.mean([
                      r['tail_goal_progress_per_agent'] for r in rows])),
                  rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=24)
    parser.add_argument('--fresh-seed', type=int, default=2026102012)
    args = parser.parse_args()
    report = run(args.root, args.checkpoint, args.output,
                 cases=args.cases, fresh_seed=args.fresh_seed)
    print(json.dumps({k:report[k] for k in
                      ('cases', 'counts', 'candidate_gridlock_count',
                       'interaction_cases', 'mean_pre_interaction_progress_rate',
                       'mean_post_interaction_progress_rate')}, indent=2))


if __name__ == '__main__':
    main()

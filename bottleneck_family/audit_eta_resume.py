"""List exact missing Gap1 eta pair indices after preemptible Slurm arrays.

An interrupted pair keeps its already committed per-seed rollout records.
Re-running an index resumes only missing continuation seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(indices: list[int]) -> str:
    if not indices:
        return ''
    spans = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            spans.append(str(start) if start == previous else f'{start}-{previous}')
            start = index
        previous = index
    spans.append(str(start) if start == previous else f'{start}-{previous}')
    return ','.join(spans)


def certified(seeds: dict) -> bool:
    valid = [row for row in seeds.values() if not row['numerical_failure']]
    successes = sum(bool(row['success']) for row in valid)
    failures = len(valid) - successes
    return len(valid) == 16 or failures >= 2 or successes >= 15


def audit(design: Path, n: int, stage: str) -> dict:
    manifest = json.loads((design / 'design_manifest.json').read_text())
    eta = json.loads((design / 'eta_pool.json').read_text())['eta']
    total = sum(state['N'] == n for state in manifest['states']) * len(eta)
    missing, numerical_incomplete, complete = [], [], []
    for index in range(total):
        path = design / f'rollouts/n{n}/pair_{index:04d}/result.json'
        if not path.exists():
            missing.append(index)
            continue
        result = json.loads(path.read_text())
        if result['pair_index'] != index or result['N'] != n:
            raise ValueError(f'identity mismatch: {path}')
        seeds = result['seeds']
        if stage == 'logical_robust':
            if certified(seeds):
                complete.append(index)
            elif len(seeds) == 16:
                numerical_incomplete.append(index)
            else:
                missing.append(index)
        elif stage == 'full_q16':
            (complete if len(seeds) == 16 else missing).append(index)
            if len(seeds) == 16 and any(row['numerical_failure'] for row in seeds.values()):
                numerical_incomplete.append(index)
        else:
            raise ValueError(stage)
    return dict(N=n, stage=stage, total=total, complete=len(complete),
                missing_indices=missing, missing_array_spec=compact(missing),
                numerical_incomplete_indices=numerical_incomplete)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--agents', type=int, required=True)
    parser.add_argument('--stage', choices=('logical_robust', 'full_q16'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.design, args.agents, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('missing_indices', 'numerical_incomplete_indices')}))


if __name__ == '__main__':
    main()

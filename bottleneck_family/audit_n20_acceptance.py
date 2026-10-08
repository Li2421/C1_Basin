"""Audit the independent N=20 non-opposing acceptance rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bottleneck_family.select_n20_checkpoint import MODES


def run(root: Path, checkpoint: Path, output: Path, *, cases: int = 12,
        fresh_seed: int = 398201, tolerance: float = .08) -> dict:
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    groups = {}
    for mode in MODES:
        rows = []
        for case in range(cases):
            folder = root / f'{mode}_{case:04d}'
            summary = json.loads((folder / 'summary.json').read_text())
            if (summary['N'] != 20 or summary['mode'] != mode
                    or summary['checkpoint_sha256'] != sha
                    or summary['split'] != 'dev'
                    or summary['observation_version'] != 'competence_v2'
                    or summary['fresh_seed'] != fresh_seed
                    or summary['fresh_start'] != case
                    or summary['fresh_count'] != 1
                    or summary['samples_per_step'] != 1
                    or summary['latent_mode'] != 'per_step'
                    or not summary['goal_stop']
                    or len(summary['rollouts']) != 1):
                raise ValueError(f'acceptance provenance mismatch: {folder}')
            trace_paths = list((folder / 'traces').glob('*.npz'))
            if len(trace_paths) != 1:
                raise ValueError(f'expected one full simulator trace: {folder}')
            with np.load(trace_paths[0], allow_pickle=False) as trace:
                positions = np.asarray(trace['positions'])
                goals = np.asarray(trace['goals'])
                correction = np.asarray(trace['projection_norm'])
                pair = np.asarray(trace['active_pair_count'])
                wall = np.asarray(trace['active_wall_count'])
            row = summary['rollouts'][0]
            if len(positions) != row['steps'] + 1 or len(correction) != row['steps']:
                raise ValueError(f'trace/summary horizon mismatch: {folder}')
            distance = np.linalg.norm(positions - goals[None], axis=2)
            initially_active = distance[0] > tolerance
            if not np.any(initially_active):
                raise ValueError(f'no active agent in control: {folder}')
            crossed = np.any((positions[:, :, 0] * goals[None, :, 0] > 0)
                             & (np.abs(positions[:, :, 0]) > .5), axis=0)
            late_start = max(0, len(positions) - 2001)
            rows.append(dict(case=case,success=bool(row['success']),
                             collision=bool(row['collision']),
                             numerical=row['numerical_error'] is not None,
                             termination=row['termination'],steps=int(row['steps']),
                             active_agents=int(initially_active.sum()),
                             active_gate_crossings=int(np.sum(crossed & initially_active)),
                             active_goal_reached=int(np.sum(
                                 (distance[-1] <= tolerance) & initially_active)),
                             all_goal_occupancy=float(np.mean(distance[-1] <= tolerance)),
                             final_active_goal_distance_mean=float(np.mean(
                                 distance[-1, initially_active])),
                             final_active_goal_distance_max=float(np.max(
                                 distance[-1, initially_active])),
                             final_2000_active_goal_progress_mean=float(np.mean(
                                 distance[late_start, initially_active]
                                 - distance[-1, initially_active])),
                             wall_active_fraction=float(np.mean(wall > 0)),
                             pair_active_fraction=float(np.mean(pair > 0)),
                             mean_projection_norm=float(np.mean(correction)),
                             min_swept_wall_clearance=row['min_swept_wall_clearance'],
                             min_swept_agent_clearance=row['min_swept_agent_clearance']))
        groups[mode] = rows
    per_mode = {}
    for mode, rows in groups.items():
        per_mode[mode] = dict(runs=len(rows),
                              success=sum(row['success'] for row in rows),
                              collision=sum(row['collision'] for row in rows),
                              numerical=sum(row['numerical'] for row in rows),
                              active_gate_crossings=sum(row['active_gate_crossings'] for row in rows),
                              active_agents=sum(row['active_agents'] for row in rows),
                              active_goal_reached=sum(row['active_goal_reached'] for row in rows),
                              mean_final_active_goal_distance=float(np.mean([
                                  row['final_active_goal_distance_mean'] for row in rows])),
                              mean_wall_active_fraction=float(np.mean([
                                  row['wall_active_fraction'] for row in rows])),
                              mean_pair_active_fraction=float(np.mean([
                                  row['pair_active_fraction'] for row in rows])),
                              mean_projection_norm=float(np.mean([
                                  row['mean_projection_norm'] for row in rows])))
    threshold = 11 if cases == 12 else int(np.ceil(.9 * cases))
    pass_gate = all(v['success'] >= threshold and v['collision'] == 0
                    and v['numerical'] == 0 for v in per_mode.values())
    report = dict(schema='gap1_n20_fresh_nonopposing_acceptance_v1',
                  checkpoint=str(checkpoint),checkpoint_sha256=sha,
                  distribution=f'dev fresh seed {fresh_seed}; {cases} cases per mode',
                  success_threshold_per_mode=threshold,pass_gate=pass_gate,
                  per_mode=per_mode,rollouts=groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=12)
    parser.add_argument('--fresh-seed', type=int, default=398201)
    args = parser.parse_args()
    report = run(args.root, args.checkpoint, args.output,
                 cases=args.cases, fresh_seed=args.fresh_seed)
    print(json.dumps(dict(pass_gate=report['pass_gate'],
                          per_mode=report['per_mode']), indent=2))


if __name__ == '__main__':
    main()

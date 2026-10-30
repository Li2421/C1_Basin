"""Reproducible, conservative post-freeze audit of opposing Gap1 traces.

This script reads full simulator evidence.  Its labels are screening labels:
the scientific report must inspect borderline/other episodes individually.
The thresholds below are fixed before inspecting the opposing outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_trace(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as trace:
        meta = json.loads(str(trace['metadata_json']))
        p = np.asarray(trace['positions'], dtype=np.float64)
        g = np.asarray(trace['goals'], dtype=np.float64)
        u_flow = np.asarray(trace['u_flow'], dtype=np.float64)
        u_ref = np.asarray(trace['u_ref'], dtype=np.float64)
        u_safe = np.asarray(trace['u_safe'], dtype=np.float64)
        active_pair = np.asarray(trace['active_pair_count'], dtype=int)
        active_wall = np.asarray(trace['active_wall_count'], dtype=int)
        clearance = np.asarray(trace['swept_clearance'], dtype=float)
    steps, n = len(u_safe), p.shape[1]
    if p.shape != (steps + 1, n, 2) or g.shape != (n, 2):
        raise ValueError(f'incomplete or malformed trace: {path}')
    dt = float(meta['config']['dt'])
    tolerance = float(meta['config']['goal_tolerance'])
    dist = np.linalg.norm(p - g[None], axis=2)
    mean_goal = dist.mean(axis=1)
    left = np.flatnonzero(p[0, :, 0] < 0)
    right = np.flatnonzero(p[0, :, 0] > 0)
    interaction = None
    # Both sides must physically reach the near-gate region and have an
    # opposing pair within 0.7 m while a pair CBF constraint is active.
    for t in range(1, steps + 1):
        if active_pair[t - 1] == 0:
            continue
        li = left[np.abs(p[t, left, 0]) <= 1.5]
        ri = right[np.abs(p[t, right, 0]) <= 1.5]
        if len(li) and len(ri) and np.linalg.norm(
                p[t][li][:, None, :] - p[t][ri][None, :, :], axis=2).min() <= .7:
            interaction = t
            break
    tail = max(0, steps - max(1000, steps // 4))
    crossed = np.zeros(n, dtype=bool)
    first_cross = np.full(n, steps + 1, dtype=int)
    for i in range(n):
        region = p[:, i, 0] >= .3 if i in left else p[:, i, 0] <= -.3
        hits = np.flatnonzero(region)
        if len(hits):
            crossed[i] = True
            first_cross[i] = int(hits[0])
    correction = np.linalg.norm(u_safe - u_ref, axis=2).mean(axis=1)
    desired_speed = np.linalg.norm(u_flow, axis=2).mean(axis=1)
    executed_speed = np.linalg.norm(u_safe, axis=2).mean(axis=1)
    common = dict(
        rollout_id=meta['rollout_id'], n=n, steps=steps,
        termination=meta['termination'], collision=bool(meta['collision']),
        min_swept_clearance=float(clearance.min()) if len(clearance) else None,
        interaction_step=interaction,
        initial_mean_goal_distance=float(mean_goal[0]),
        final_mean_goal_distance=float(mean_goal[-1]),
        final_goal_fraction=float(np.mean(dist[-1] <= tolerance)),
        crossed_gate=int(crossed.sum()),
        late_gate_crossings=int(np.sum((first_cross >= tail) & (first_cross <= steps))),
        tail_goal_progress_per_agent=float(mean_goal[tail] - mean_goal[-1]),
        tail_pair_active_fraction=float(np.mean(active_pair[tail:] > 0)),
        tail_wall_active_fraction=float(np.mean(active_wall[tail:] > 0)),
        tail_mean_flow_speed=float(desired_speed[tail:].mean()),
        tail_mean_executed_speed=float(executed_speed[tail:].mean()),
        tail_mean_correction_per_agent=float(correction[tail:].mean()),
    )
    if interaction is not None:
        approach = np.r_[left[np.abs(p[interaction, left, 0]) <= 1.5],
                         right[np.abs(p[interaction, right, 0]) <= 1.5]]
        approach_progress = dist[0, approach] - dist[interaction, approach]
        common.update(
            approach_agents=int(len(approach)),
            approach_progress_median=float(np.median(approach_progress)),
            pre_interaction_progress_per_agent=float(mean_goal[0] - mean_goal[interaction]),
            pre_interaction_progress_rate=float((mean_goal[0] - mean_goal[interaction]) /
                                                (interaction * dt)),
            post_interaction_progress_rate=float((mean_goal[interaction] - mean_goal[-1]) /
                                                 (max(steps - interaction, 1) * dt)),
            pre_interaction_mean_speed=float(executed_speed[:interaction].mean()),
            post_interaction_mean_speed=float(executed_speed[interaction:].mean()),
            post_interaction_pair_active_fraction=float(np.mean(active_pair[interaction:] > 0)),
            post_interaction_mean_correction_per_agent=float(correction[interaction:].mean()),
        )
    if meta['collision'] or (len(clearance) and clearance.min() <= 0):
        label = 'collision'
    elif meta['termination'] == 'numerical_failure':
        label = 'numerical_failure'
    elif meta['termination'] == 'success':
        label = 'spontaneous_success'
    elif interaction is None:
        label = 'navigation_or_noninteraction_failure'
    elif (steps - tail >= 1000 and
          common['approach_progress_median'] >= .5 and
          common['tail_goal_progress_per_agent'] <= .1 and
          common['late_gate_crossings'] == 0 and
          common['tail_pair_active_fraction'] >= .05):
        label = 'clean_bounded_gridlock_candidate'
    else:
        label = 'mixed_or_unclear'
    common['screening_label'] = label
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    root = args.evaluation
    rows = [audit_trace(p) for p in sorted((root / 'traces').glob('*.npz'))]
    summary = json.loads((root / 'summary.json').read_text())
    if len(rows) != len(summary['rollouts']):
        raise ValueError('audit must cover every rollout in evaluation summary')
    counts = {label:sum(r['screening_label'] == label for r in rows)
              for label in sorted({r['screening_label'] for r in rows})}
    result = {'schema':'gap1_scale_opposing_audit_v1',
              'thresholds':{'interaction_gate_x':1.5,'interaction_pair_distance':.7,
                            'approach_progress_median':.5,'tail_goal_progress_per_agent':.1,
                            'tail_pair_active_fraction':.05,'tail_min_steps':1000},
              'counts':counts,'rollouts':rows}
    (root / 'opposing_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    main()

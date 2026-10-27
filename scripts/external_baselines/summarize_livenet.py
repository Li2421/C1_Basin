#!/usr/bin/env python3
"""Recompute metrics from the authors' committed LiveNet CSV and histories.

Usage: python3 scripts/external_baselines/summarize_livenet.py /path/to/LiveNet
"""
import csv
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
results = {}
for path in sorted((root / 'experiment_results').glob('*.csv')):
    with path.open(newline='') as f:
        rows = [[float(x) for x in row] for row in csv.reader(line for line in f if not line.startswith('#')) if row]
    if not rows:
        continue
    assert all(len(row) == 14 for row in rows), path
    collision = sum(any(r[i] != 0 for i in (3, 5, 7)) for r in rows)
    unfinished = sum(any(r[i] == -1 for i in (0, 1)) for r in rows)
    both = sum(all(r[i] != -1 for i in (0, 1)) and not any(r[i] != 0 for i in (3, 5, 7)) for r in rows)
    times = [max(r[:2]) * 0.2 for r in rows if all(r[i] != -1 for i in (0, 1))]
    unique_outcomes = len({tuple(r[:12]) for r in rows})
    results[path.stem] = {'rows': len(rows), 'collision_rows': collision, 'unfinished_rows': unfinished,
                          'collision_free_both_reached': both, 'mean_makespan_reached_s': statistics.mean(times) if times else None,
                          'unique_non_timing_metric_rows': unique_outcomes}
for name in ('LiveNet_Doorway', 'LiveNet_Intersection'):
    path = root / 'experiment_results' / 'histories' / (name + '.json')
    obj = json.loads(path.read_text())
    steps = obj['iterations']
    goal_indices = []
    for i in range(2):
        goal_indices.append(next((j for j, step in enumerate(steps) if math.dist(step['states'][i][:2], step['goals'][i][:2]) < 0.05 and abs(step['states'][i][3]) < 0.05), None))
    min_dist = min(math.dist(s['states'][0][:2], s['states'][1][:2]) for s in steps)
    obs_min_clearance = []
    for i in range(2):
        end = goal_indices[i] if goal_indices[i] is not None else len(steps)
        obs_min_clearance.append(min(math.dist(s['states'][i][:2], o[:2]) - (0.1 + o[2]) for s in steps[:end] for o in obj['obstacles']))
    results[name]['saved_history'] = {'samples':len(steps),'first_goal_indices':goal_indices,'min_agent_distance_m':min_dist,'min_obstacle_clearance_m':obs_min_clearance}
print(json.dumps(results, indent=2, sort_keys=True))

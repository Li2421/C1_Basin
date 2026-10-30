"""Versioned offline split of timeout outcomes into stalled and moving cases.

The original first-terminal-event result is immutable.  This diagnostic emits a
new, mutually exclusive six-class protocol by splitting only `other_timeout`.
"""
import json
from pathlib import Path

import numpy as np


OUTCOMES = ('success', 'wall_collision', 'agent_collision', 'safe_deadlock',
            'stalled_deadlock', 'other_timeout')
PROTOCOL = 'exclusive_first_event_with_stalled_timeout_v3'


def classify_timeout_trace(trace, outcome, dt, *, window_seconds=2.0,
                           max_speed=0.05, max_goal_error_change=0.02):
    """Classify a terminal trace without changing any non-timeout outcome."""
    if outcome != 'other_timeout':
        return outcome, {'eligible': False}
    steps = int(round(window_seconds / dt))
    speeds = np.asarray(trace['max_speed'], dtype=float)
    errors = np.asarray(trace['goal_errors'], dtype=float)
    if speeds.ndim != 1 or errors.ndim != 2 or errors.shape[1] != 2:
        raise ValueError('Expected max_speed[T] and goal_errors[T,2]')
    if len(speeds) != len(errors) or len(speeds) < steps:
        return outcome, {'eligible': True, 'enough_window': False,
                         'episode_steps': len(speeds)}
    tail_speed = speeds[-steps:]
    error_change = np.abs(errors[-1] - errors[-steps])
    stalled = bool(np.all(tail_speed < max_speed) and
                   np.all(error_change < max_goal_error_change))
    details = dict(eligible=True, enough_window=True, window_steps=steps,
                   tail_max_speed=float(tail_speed.max()),
                   goal_error_change=error_change.tolist(), stalled=stalled)
    return ('stalled_deadlock' if stalled else outcome), details


def reclassify_evaluation(folder, *, window_seconds=2.0, max_speed=0.05,
                          max_goal_error_change=0.02):
    """Reclassify completed paired output and verify a six-class partition."""
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    if not (folder/'complete.json').is_file():
        raise ValueError(f'Incomplete evaluation: {folder}')
    dt = float(config['environment']['dt'])
    report = dict(folder=str(folder.resolve()), protocol=PROTOCOL,
                  detector=dict(window_seconds=window_seconds, max_speed=max_speed,
                                max_goal_error_change=max_goal_error_change,
                                strict_speed_comparison='<',
                                strict_goal_error_change_comparison='<'), arms={})
    for arm in ('mac_only', 'mac_cbf'):
        summaries = json.loads((folder/arm/'summary.json').read_text())['rollouts']
        rows = []
        for summary in summaries:
            rid = int(summary['rollout_id'])
            with np.load(folder/arm/f'rollout_{rid:04d}.npz') as trace:
                label, details = classify_timeout_trace(trace, summary['outcome'], dt,
                    window_seconds=window_seconds, max_speed=max_speed,
                    max_goal_error_change=max_goal_error_change)
            rows.append(dict(rollout_id=rid, original_outcome=summary['outcome'],
                             outcome=label, **details))
        labels = [row['outcome'] for row in rows]
        if any(label not in OUTCOMES for label in labels) or len(labels) != len(summaries):
            raise ValueError('Invalid reclassified outcomes')
        counts = {label: labels.count(label) for label in OUTCOMES}
        if sum(counts.values()) != len(rows):
            raise AssertionError('Outcome counts must partition all rollouts')
        report['arms'][arm] = dict(n_rollouts=len(rows), outcome_counts=counts,
                                   outcome_rates={key:value/len(rows) for key,value in counts.items()},
                                   rows=rows)
    return report


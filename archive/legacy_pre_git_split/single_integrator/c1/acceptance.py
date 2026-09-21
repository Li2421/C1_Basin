"""Explicit endpoint gates; a completed process alone is not efficacy evidence."""
import numpy as np
from single_integrator.c1.paired_statistics import summarize


def assess(records, expected_rids, expected_noises, *, family_size):
    rids, noises = list(expected_rids), list(expected_noises)
    if len(set(rids)) != len(rids) or len(set(noises)) != len(noises):
        raise ValueError('Duplicate entries in the frozen test design')
    expected = {(r,n,m) for r in rids for n in noises for m in ('Safety','C1')}
    actual = [(r['rid'],r['noise_seed'],r['method']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Records do not exactly cover the frozen paired test')
    safety_fields = ('agent_collision_steps','wall_collision_steps','outside_endpoints',
                     'cbf_violations','speed_violations')
    safety = {name:{key:0 for key in safety_fields} for name in ('Safety','C1')}
    for row in records:
        label = row['six_class_outcome']
        success, dead, timeout = map(bool,(row['success'],row['any_deadlock'],row['timeout']))
        if sum((success,dead,timeout)) != 1:
            raise ValueError('Expected mutually exclusive original first-event outcomes')
        if label not in ('success','safe_deadlock','stalled_deadlock','other_timeout'):
            raise ValueError('Collision/unknown outcome cannot enter a safety-certified result')
        if (label=='success') != success or (label=='safe_deadlock') != dead:
            raise ValueError('Six-class label disagrees with first-event outcome')
        if (label in ('stalled_deadlock','other_timeout')) != timeout:
            raise ValueError('Timeout subtype disagrees with first-event outcome')
        for key in safety_fields:
            value = row['safety'][key]
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid safety audit count')
            safety[row['method']][key] += value
    stats = summarize(records, family_size=family_size, endpoint='strict_or_stalled')
    gates = dict(enough_independent_events=stats['event_count_gate_passed'],
        relative_deadlock_reduction_at_least_80_percent=stats['relative_deadlock_reduction'] is not None and stats['relative_deadlock_reduction']>=.8,
        adjusted_deadlock_difference_excludes_zero=stats['rate_reduction_interval_excludes_zero'],
        baseline_deadlocks_become_success_at_least_80_percent=stats['safety_deadlock_to_success_fraction'] is not None and stats['safety_deadlock_to_success_fraction']>=.8,
        total_failures_do_not_increase=stats['c1_failures']<=stats['safety_failures'],
        observed_safety_violations_zero=all(v==0 for policy in safety.values() for v in policy.values()))
    return dict(gates=gates, all_endpoint_gates_passed=all(gates.values()), statistics=stats,
        safety_counts=safety, limitations='Per-comparison empirical evidence only; not population safety or global C1 convergence; other required runs must also be reviewed')

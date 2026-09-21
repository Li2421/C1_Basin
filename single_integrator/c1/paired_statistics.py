"""Paired episode comparisons with initial-state cluster bootstrap.

Noise replicas from one initial state stay together. Safety/C1 pairing and
the baseline denominator are preserved in every resample. This is descriptive
finite-sample inference, not proof of population safety or liveness.
"""
import numpy as np


def summarize(records, *, replicates=10000, seed=20260916, family_size=6, endpoint='strict'):
    if replicates < 1000 or family_size < 1:
        raise ValueError('at least 1000 resamples and positive comparison family required')
    if endpoint not in ('strict', 'strict_or_stalled'):
        raise ValueError('unknown deadlock endpoint')
    paired = {}
    for row in records:
        name = row['method']
        if name not in ('Safety', 'C1'):
            raise ValueError('unexpected policy')
        key = (row['rid'], row['noise_seed'])
        pair = paired.setdefault(key, {})
        if name in pair:
            raise ValueError('duplicate paired outcome')
        if bool(row['success']) and bool(row['any_deadlock']):
            raise ValueError('requires first-event outcomes; recovered-deadlock runs are different')
        if endpoint=='strict_or_stalled' and 'six_class_outcome' not in row:
            raise ValueError('combined endpoint requires original explicit six-class detector labels')
        pair[name] = row
    if not paired or any(set(p) != {'Safety', 'C1'} for p in paired.values()):
        raise ValueError('missing paired outcomes')
    by_start = {}
    noises = {}
    for (rid, noise), pair in sorted(paired.items()):
        a, b = pair['Safety'], pair['C1']
        def dead(row):
            return bool(row['any_deadlock']) or (endpoint=='strict_or_stalled' and row['six_class_outcome']=='stalled_deadlock')
        da, db = dead(a), dead(b)
        sa, sb = bool(a['success']), bool(b['success'])
        values = np.array([da, db, not sa, not sb, da and sb,
                           da and not db, db and not da, not sa and sb, sa and not sb], float)
        by_start.setdefault(rid, []).append(values)
        noises.setdefault(rid, set()).add(noise)
    if any(x != next(iter(noises.values())) for x in noises.values()):
        raise ValueError('noise replicas differ across initial states')
    clusters = np.stack([np.sum(v, axis=0) for v in by_start.values()])
    n = len(paired)
    totals = clusters.sum(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, clusters.shape[1]))
    # Bound memory while retaining the entire initial-state cluster.
    for lo in range(0, replicates, 128):
        hi = min(lo+128, replicates)
        indices = rng.integers(len(clusters), size=(hi-lo, len(clusters)))
        boot[lo:hi] = clusters[indices].sum(axis=1)
    alpha = .05/family_size
    def interval(values):
        finite = values[np.isfinite(values)]
        return list(map(float, np.quantile(finite, [alpha/2, 1-alpha/2]))) if len(finite) else None
    with np.errstate(divide='ignore', invalid='ignore'):
        reduction = (boot[:, 0]-boot[:, 1])/boot[:, 0]
        resolution = boot[:, 4]/boot[:, 0]
    reduction[boot[:, 0] == 0] = np.nan
    resolution[boot[:, 0] == 0] = np.nan
    dead_diff_ci = interval((boot[:, 0]-boot[:, 1])/n)
    event_clusters = int(np.sum(clusters[:, 0] > 0))
    adequate = len(clusters) >= 30 and event_clusters >= 10
    return dict(endpoint=endpoint, episodes_per_policy=n, initial_state_clusters=len(clusters),
        noise_replicas=len(next(iter(noises.values()))),
        safety_deadlocks=int(totals[0]), c1_deadlocks=int(totals[1]),
        safety_failures=int(totals[2]), c1_failures=int(totals[3]),
        deadlock_rate_difference=float((totals[0]-totals[1])/n),
        deadlock_rate_difference_ci=dead_diff_ci,
        failure_rate_difference=float((totals[2]-totals[3])/n),
        failure_rate_difference_ci=interval((boot[:, 2]-boot[:, 3])/n),
        relative_deadlock_reduction=float((totals[0]-totals[1])/totals[0]) if totals[0] else None,
        relative_deadlock_reduction_ci=interval(reduction),
        safety_deadlock_to_success=int(totals[4]),
        safety_deadlock_to_success_fraction=float(totals[4]/totals[0]) if totals[0] else None,
        safety_deadlock_to_success_fraction_ci=interval(resolution),
        deadlock_removed=int(totals[5]), deadlock_introduced=int(totals[6]),
        new_success=int(totals[7]), lost_success=int(totals[8]),
        baseline_deadlock_clusters=event_clusters,
        event_count_gate_passed=adequate,
        zero_baseline_bootstrap_replicates=int(np.sum(boot[:, 0] == 0)),
        bootstrap_replicates=replicates, bootstrap_seed=seed, family_size=family_size,
        per_comparison_confidence=1-alpha,
        uncertainty='Approximate percentile cluster bootstrap; sparse/zero-event intervals do not bound unseen events',
        rate_reduction_interval_excludes_zero=bool(dead_diff_ci[0] > 0),
        supports_reduction_with_event_gate=bool(adequate and dead_diff_ci[0] > 0))

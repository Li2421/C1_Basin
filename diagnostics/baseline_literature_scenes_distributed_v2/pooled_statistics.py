"""Equal-controller point estimates when scene eligibility creates missing cells.

The existing bootstrap already averages types, controllers and state clusters
in that order. Its flat nanmean point was equivalent only for balanced cells.
This wrapper corrects the pooled points, retaining every paired draw and CI.
Per-scene sealed statistics are unchanged; there all controllers share states.
"""
from experiment import *
import numpy as np
from migration_statistics_complete import paired


def equal_point(matrix, types):
    groups = [np.flatnonzero(types == t) for t in sorted(set(types))]
    return np.mean([np.mean([np.nanmean(matrix[i], axis=0) for i in g], axis=0)
                    for g in groups], axis=0)


def pooled_paired(rows, qualification, scene, methods):
    result = paired(rows, qualification, scene, methods)
    if not result:
        return result
    controllers = result['qualified_controllers']
    states = [tuple(s) for s in result['paired_states']]
    eligible = set(controllers)
    lookup = {}
    types = {}
    for r in rows:
        c = r['case']
        ctl = c['controller_payload']['nominal']
        name = ctl['controller']
        if name not in eligible or c['method'] not in methods:
            continue
        if c['state']['scene'] != scene:
            continue
        s = c['state']
        lookup[(name, s['N'], s['index'], c['seed'], c['method'])] = r
        types[name] = ctl['type_code']
    lower = np.full((len(controllers), len(states), len(methods)), np.nan)
    upper = lower.copy()
    for i, name in enumerate(controllers):
        for j, s in enumerate(states):
            if not all((name, *s, seed, method) in lookup
                       for seed in (0, 1) for method in methods):
                continue
            for k, method in enumerate(methods):
                rr = [lookup[(name, *s, seed, method)] for seed in (0, 1)]
                lower[i, j, k] = np.mean([r['success'] for r in rr])
                upper[i, j, k] = np.mean([min(1, r['success'] + r['numerical_failure']) for r in rr])
    types = np.asarray([types[name] for name in controllers])
    for label in [*sorted(set(types)), 'type_equal']:
        mask = np.ones(len(types), bool) if label == 'type_equal' else types == label
        point = equal_point(lower[mask], types[mask])
        up = equal_point(upper[mask], types[mask])
        for row in result['statistics']:
            if row['type'] != label:
                continue
            i = methods.index(row['method'])
            row.update(success_rate=float(point[i]), success_upper=float(up[i]),
                       delta_vs_safety=float(point[i] - point[0]),
                       identified_delta_bounds=[float(point[i] - up[0]), float(up[i] - point[0])])
            for method, comparison in row.get('delta_vs_fixed', {}).items():
                j = methods.index(method)
                comparison.update(delta=float(point[i] - point[j]),
                                  identified_delta_bounds=[float(point[i] - up[j]), float(up[i] - point[j])])
    result['point_weighting'] = 'Equal types, then equal eligible controllers within type, then equal available paired state clusters within controller; seed twins averaged within state. Bootstrap draws already use this same order.'
    return result


def verify():
    unbalanced = np.asarray([[[0.], [np.nan], [np.nan]], [[1.], [1.], [1.]]])
    assert equal_point(unbalanced, np.asarray(['a', 'a']))[0] == .5
    assert np.nanmean(unbalanced) == .75
    hierarchy = np.asarray([[[0.]], [[0.]], [[1.]]])
    assert equal_point(hierarchy, np.asarray(['a', 'a', 'b']))[0] == .5
    balanced = np.asarray([[[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]]])
    assert np.array_equal(equal_point(balanced, np.asarray(['a', 'a'])), balanced.mean((0, 1)))
    write(HERE / 'POOLED_POINT_ESTIMATE_CHECK.json', dict(
        utc=utc(), unbalanced_equal_controller=.5, flat_cell_weighting=.75,
        unequal_type_sizes_equal_type=.5, balanced_unchanged=True,
        bootstrap_draws_and_CIs_unchanged=True,
        no_new_rollouts=True, scope='Four-scene and supported-scene pooled points only'))
    seal('POOLED_POINT_ESTIMATE_CHECK.receipt.json',
         [HERE / 'POOLED_POINT_ESTIMATE_CHECK.json', HERE / 'pooled_statistics.py',
          HERE / 'migration_statistics_complete.py'],
         before_combined_result_calculation=True)


if __name__ == '__main__':
    verify()

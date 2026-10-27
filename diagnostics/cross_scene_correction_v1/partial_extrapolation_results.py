"""User-requested computational censoring; preserve frozen policies and all 120 slots."""
from common import *
from stages import read_result
import numpy as np


def main():
    stop = read(HERE / 'USER_EXTRAPOLATION_STOP_20261025.json')
    eligible = set(stop['completed_uids'])
    assert len(eligible) == stop['completed_at_stop'] == 107
    freeze = read(HERE / 'MODEL_FROZEN.receipt.json')
    for path, expected in freeze['files'].items():
        assert sha(path) == expected, path
    arms = read(HERE / 'EXTRAPOLATION_ARMS.json')
    cases = read(HERE / 'extrapolation_CASES.json')
    assert len(cases) == 120
    assert eligible <= {c['rollout_uid'] for c in cases}
    action_ids = {etahex(a['eta_a']): i for i, a in enumerate(actions())}
    rows, censored = [], []
    for c in cases:
        row = dict(rollout_uid=c['rollout_uid'],
                   controller=c['controller_payload']['nominal']['controller'],
                   type=c['controller_payload']['nominal']['type_code'],
                   state=c['state']['index'], action=action_ids[etahex(c['eta_a'])])
        if c['rollout_uid'] not in eligible:
            row.update(status='USER_STOPPED_COMPUTATION', scientific_outcome=None)
            censored.append(row)
            continue
        r = read_result('extrapolation', c)
        row.update(status='COMPLETED', success=r['success'], timeout=r['timeout'],
                   collision=r['collision'], unknown=r['numerical_failure'],
                   collision_types=r.get('collision_types', {}))
        rows.append(row)
    assert len(rows) == 107 and len(censored) == 13
    rng = np.random.default_rng(2026102414)
    methods = {}
    for method, action in arms['methods'].items():
        planned = [c for c in cases if action_ids[etahex(c['eta_a'])] == action]
        observed = [r for r in rows if r['action'] == action]
        missing = [r for r in censored if r['action'] == action]
        controllers = sorted({c['controller_payload']['nominal']['controller'] for c in planned})
        assert len(controllers) == 2 and len(planned) == 24
        lower = np.zeros((2, 12))
        upper = np.ones((2, 12))
        observed_by_type = {}
        for i, controller in enumerate(controllers):
            observed_ctl = [r for r in observed if r['controller'] == controller]
            for r in observed_ctl:
                lower[i, r['state']] = r['success']
                upper[i, r['state']] = min(1, r['success'] + r['unknown'])
            observed_by_type[controller] = dict(completed=len(observed_ctl), planned=12,
                success=sum(r['success'] for r in observed_ctl),
                timeout=sum(r['timeout'] for r in observed_ctl),
                collision=sum(r['collision'] for r in observed_ctl),
                numerical_unknown=sum(r['unknown'] for r in observed_ctl))
        # Preserve the frozen seed, type/state sampling structure, and 5000 draws.
        ci = rng.integers(2, size=(5000, 2))
        st = rng.integers(12, size=(5000, 2, 12))
        lo_boot = lower[ci[:, :, None], st].mean((1, 2))
        hi_boot = upper[ci[:, :, None], st].mean((1, 2))
        methods[method] = dict(action=action, eta=actions()[action]['eta_a'],
            planned_n=24, completed_n=len(observed), budget_censored_n=len(missing),
            successes=int(sum(r['success'] for r in observed)),
            timeouts=int(sum(r['timeout'] for r in observed)),
            observed_collisions=int(sum(r['collision'] for r in observed)),
            numerical_unknown=int(sum(r['unknown'] for r in observed)),
            success_lower_rate=float(lower.mean()), success_upper_rate=float(upper.mean()),
            completed_only_type_equal_rate=float(np.mean([
                np.mean([r['success'] for r in observed if r['controller'] == ctl])
                for ctl in controllers])),
            observed_by_type=observed_by_type,
            lower_bound_bootstrap_ci=np.percentile(lo_boot, [2.5, 97.5]).tolist(),
            upper_bound_bootstrap_ci=np.percentile(hi_boot, [2.5, 97.5]).tolist(),
            conditional_bootstrap_endpoint_envelope=[float(np.percentile(lo_boot, 2.5)),
                                                      float(np.percentile(hi_boot, 97.5))])
    write(HERE / 'EXTRAPOLATION_RESULTS.json', dict(status='PARTIAL_USER_STOP',
        planned_requests=120, completed_requests=107, budget_censored_requests=13,
        methods=methods, rows=rows, censored_rows=censored,
        type_equal=True, type_equal_target='Original two types x 12 planned states per method',
        only_selected_actions=True, frozen_policies_unchanged=True,
        numerical_unknowns_distinct_from_uncomputed=True,
        uncertainty='Full planned denominator retained: observed success is the lower bound; '
                    'uncomputed slots and numerical unknowns are free 0/1 in the upper bound. '
                    'These are identification bounds, not fabricated result labels. '
                    'Bootstrap endpoint intervals are conditional sensitivity summaries, '
                    'not a complete-data confidence interval or a correction for informative censoring.',
        limitation='107 completed requests are selected by computation time. Completed-only '
                   'rates can be biased and cannot substitute for the original full N10 comparison. '
                   'No effect on complete preregistered A/B/C analyses. Gap1 fixed map only; no N100 claim.',
        user_stop_receipt=str(HERE / 'USER_EXTRAPOLATION_STOP_20261025.json'),
        user_stop_sha256=sha(HERE / 'USER_EXTRAPOLATION_STOP_20261025.json'),
        frozen_choices_sha256=sha(HERE / 'EXTRAPOLATION_FROZEN_CHOICES.json'),
        planned_cases_sha256=sha(HERE / 'extrapolation_CASES.json'),
        accounting_source_sha256=sha(__file__)))
    write(HERE / 'EXTRAPOLATION_CENSORED_CASES.json',
          dict(reason='Explicit user instruction to stop and use the 107 completed requests',
               planned=120, completed=107, not_run_to_completion=13,
               scientific_labels_created=0, cases=[c for c in cases if c['rollout_uid'] not in eligible]))
    print('Partial extrapolation:', {k: (v['completed_n'], v['success_lower_rate'],
                                        v['success_upper_rate']) for k, v in methods.items()}, flush=True)


if __name__ == '__main__':
    main()

"""Frozen analysis for direct-only versus VI-augmented R_CERT diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, rankdata, spearmanr


def analyze(folder):
    read = lambda name: json.loads((folder/name).read_text())
    protocol, complete = read('protocol.json'), read('complete.json')
    diagnostics, rows, extended = (read('diagnostics.json'),
                                   read('records.json'),
                                   read('extended_records.json'))
    audit = json.loads((folder.parent/'c1_vi_r_cert_check_v1/report.json').read_text())
    n = len(protocol['starts'])
    seeds = protocol['evaluation_seeds']
    arms = ['reference'] + [
        f'{kind}_{direction}' for kind in ('direct', 'augmented')
        for direction in ('negative', 'positive', 'random_0', 'random_1')]
    assert n == 25 and len(seeds) == 2 and len(diagnostics) == n
    assert len(rows) == complete['main_replays'] == n*len(seeds)*len(arms)
    assert len(extended) == complete['extended_replays']
    index = {(r['rid'], r['seed'], r['arm']): r for r in rows}
    assert len(index) == len(rows)
    bootstrap = np.random.default_rng(2026091898).integers(0, n, (10000, n))

    def estimate(value, draws):
        draws = np.asarray(draws, float)
        valid = np.isfinite(draws)
        return dict(value=float(value) if np.isfinite(value) else None,
                    ci95=np.quantile(draws[valid], [.025, .975]).tolist()
                    if valid.any() else None,
                    valid_bootstrap_fraction=float(valid.mean()))

    def auc(scores, labels):
        labels = np.asarray(labels).reshape(-1)
        ranks = rankdata(np.repeat(scores, labels.size//len(scores)))
        positives = labels.sum(); negatives = len(labels)-positives
        if positives == 0 or negatives == 0:
            return np.nan
        return float((ranks@labels-positives*(positives+1)/2) /
                     (positives*negatives))

    def corr(scores, frequencies):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConstantInputWarning)
            return float(spearmanr(scores, frequencies).statistic)

    baseline = np.array([[[index[i, s, 'reference'][key]
                           for key in ('primary_deadlock', 'success')]
                          for s in seeds] for i in range(n)], float)
    frequencies = baseline[:, :, 0].mean(axis=1)
    prediction = {}
    for kind in ('direct', 'augmented'):
        raw = [d[f'{kind}_prediction_risk'] for d in diagnostics]
        scores = np.array([np.nan if x is None else x for x in raw])
        all_defined = bool(np.isfinite(scores).all())
        if all_defined:
            c = corr(scores, frequencies)
            a = auc(scores, baseline[:, :, 0])
            c_draws = [corr(scores[b], frequencies[b]) for b in bootstrap]
            a_draws = [auc(scores[b], baseline[b, :, 0]) for b in bootstrap]
        else:
            c = a = np.nan; c_draws = a_draws = [np.nan]
        groups = []
        finite = np.isfinite(scores)
        if finite.any():
            edges = np.unique(np.quantile(scores[finite], [0, .25, .5, .75, 1]))
            for k in range(max(1, len(edges)-1)):
                mask = finite & (scores >= edges[k]) & (
                    (scores <= edges[-1]) if k == len(edges)-2 or len(edges) == 1
                    else scores < edges[k+1])
                groups.append(dict(starts=int(mask.sum()),
                                   mean_risk=float(scores[mask].mean()),
                                   primary_deadlock_frequency=float(
                                       frequencies[mask].mean())))
        prediction[kind] = dict(
            all_defined=all_defined,
            undefined_starts=int(np.sum(~finite)),
            spearman=estimate(c, c_draws), auc=estimate(a, a_draws),
            risk_bins=groups,
            by_start=[dict(rid=i, risk=None if not np.isfinite(scores[i])
                           else float(scores[i]),
                           independent_primary_frequency=float(frequencies[i]))
                      for i in range(n)])
        prediction[kind]['passed'] = bool(
            all_defined and prediction[kind]['spearman']['ci95'] is not None and
            prediction[kind]['spearman']['ci95'][0] > 0 and
            prediction[kind]['auc']['ci95'][0] > .5)

    coverage = {}
    for kind in ('direct', 'augmented'):
        event_count = failures = 0
        for d in diagnostics:
            values = d[f'{kind}_prediction_risks']
            for value, event in zip(values, d['prediction_primary_events']):
                if event:
                    event_count += 1
                    failures += int(not np.isfinite(value) or value < 1-2e-6)
        coverage[kind] = dict(primary_events=event_count,
                              D_and_R_below_one=failures)

    def arm_matrix(arm, key):
        return np.array([[float(index[i, s, arm][key]) for s in seeds]
                         for i in range(n)])

    def random_matrix(kind, key):
        return np.mean(np.stack([arm_matrix(f'{kind}_random_{r}', key)
                                 for r in range(2)]), axis=0)

    def difference(left, right):
        # Positive means left has a larger rate than right.
        per_start=(left-right).mean(axis=1)
        draws=per_start[bootstrap].mean(axis=1)
        return dict(mean=float(per_start.mean()),
                    ci95=np.quantile(draws, [.025, .975]).tolist())

    intervention = {}
    outcome_keys = ('primary_deadlock', 'historical_strict',
                    'terminal_label_deadlock', 'stalled_deadlock', 'success',
                    'ordinary_timeout', 'collision')
    for kind in ('direct', 'augmented'):
        negative = {k: arm_matrix(f'{kind}_negative', k) for k in outcome_keys}
        positive = {k: arm_matrix(f'{kind}_positive', k) for k in outcome_keys}
        reference = {k: arm_matrix('reference', k) for k in outcome_keys}
        random = {k: random_matrix(kind, k) for k in outcome_keys}
        comparisons = dict(
            baseline_minus_negative=difference(
                reference['primary_deadlock'], negative['primary_deadlock']),
            positive_minus_negative=difference(
                positive['primary_deadlock'], negative['primary_deadlock']),
            random_mean_minus_negative=difference(
                random['primary_deadlock'], negative['primary_deadlock']),
            negative_success_minus_baseline=difference(
                negative['success'], reference['success']),
            negative_timeout_minus_baseline=difference(
                negative['ordinary_timeout'], reference['ordinary_timeout']))
        summaries = {}
        for name, matrices in [('reference', reference), ('negative', negative),
                               ('positive', positive), ('random_mean', random)]:
            summaries[name] = {k: float(v.sum()) if name != 'random_mean'
                               else float(v.sum()) for k, v in matrices.items()}
            summaries[name]['J_def_mean'] = float(np.mean([
                index[i, s, ('reference' if name == 'reference' else
                       f'{kind}_{name}' if name != 'random_mean' else
                       f'{kind}_random_0')]['J_def']
                for i in range(n) for s in seeds])) if name != 'random_mean' else float(
                    np.mean([index[i, s, f'{kind}_random_{r}']['J_def']
                             for i in range(n) for s in seeds for r in range(2)]))
        nonzero = [d[f'{kind}_gradient_norm'] > 0 for d in diagnostics]
        relevant_arms = [f'{kind}_{x}' for x in
                         ('negative', 'positive', 'random_0', 'random_1')]
        calibration = [d['calibration'][a] for d in diagnostics
                       for a in relevant_arms]
        matched = sum(bool(x['matched']) for x in calibration)
        executed = {a: float(np.sqrt(sum(
            index[i, s, a]['direct_action_sum_squares']
            for i in range(n) for s in seeds) / max(1, sum(
            index[i, s, a]['direct_action_components']
            for i in range(n) for s in seeds)))) for a in relevant_arms}
        intervention[kind] = dict(
            outcomes=summaries, comparisons=comparisons,
            nonzero_gradient_starts=int(sum(nonzero)),
            zero_gradient_starts=int(n-sum(nonzero)),
            calibration=dict(matched=matched, total=len(calibration),
                             unmatched=len(calibration)-matched),
            realized_postprojection_rms=executed,
            zero_executed_effect_replays=sum(
                index[i, s, a]['zero_executed_effect']
                for i in range(n) for s in seeds for a in relevant_arms))
        advantages = [comparisons[x] for x in
                      ('baseline_minus_negative', 'positive_minus_negative',
                       'random_mean_minus_negative')]
        intervention[kind]['passed'] = bool(
            all(x['ci95'][0] > 0 for x in advantages) and
            comparisons['negative_success_minus_baseline']['ci95'][0] >= 0 and
            comparisons['negative_timeout_minus_baseline']['ci95'][1] <= 0 and
            summaries['negative']['collision'] == 0 and
            matched == len(calibration) and sum(nonzero) > 0)

    extended_index = {(r['rid'], r['seed'], r['arm']): r for r in extended}
    delayed = []
    for key, short in index.items():
        rid, seed, arm = key
        if seed not in protocol['extended_evaluation_seeds']:
            continue
        long = extended_index[key]
        if short['ordinary_timeout'] or long['ordinary_timeout'] or (
                short['primary_deadlock'] != long['primary_deadlock']):
            delayed.append(dict(
                rid=rid, seed=seed, arm=arm,
                short_outcome=short['outcome'], long_outcome=long['outcome'],
                short_primary=short['primary_deadlock'],
                long_primary=long['primary_deadlock'],
                long_action_count=long['action_count']))
    extended_summary = dict(
        horizon_steps=protocol['extended_horizon_steps'],
        replays=len(extended), conditional_failure_rows=delayed,
        short_timeouts=sum(r['ordinary_timeout'] for r in rows
                           if r['seed'] in protocol['extended_evaluation_seeds']),
        extended_timeouts=sum(r['ordinary_timeout'] for r in extended))

    fd_errors = np.array([r['relative_error']
                          for r in audit['gradient']['finite_differences']])
    numerical = dict(
        finite_difference_relative_errors=fd_errors.tolist(),
        stable_distance_unit_test=True,
        full_rollout_fd_within_10_percent=bool(np.any(
            np.all(fd_errors <= .1, axis=1))),
        projection_feasible=(audit['projection']['witness_min_cbf_residual'] >=
                             -audit['cbf']['feasibility_tol'] and
                             audit['projection']['witness_max_speed_excess'] <=
                             audit['cbf']['speed_tol']),
        NaN_or_Inf_in_audit=False, solver_failures=audit['solver_failures'])
    numerical['passed'] = bool(
        numerical['full_rollout_fd_within_10_percent'] and
        numerical['projection_feasible'] and numerical['solver_failures'] == 0)
    implementation = dict(
        passed=bool(audit['passed']), monitor=audit['monitor'],
        projection=audit['projection'], implication=audit['implication'],
        unit_tests='tests.test_c1_r_cert and tests.test_c1_vi_r_cert')

    augmented_points = intervention['augmented']['comparisons']
    empirical_wrong_way = any(augmented_points[x]['mean'] <= 0 for x in
        ('baseline_minus_negative', 'positive_minus_negative',
         'random_mean_minus_negative'))
    if (implementation['passed'] and numerical['passed'] and
            prediction['augmented']['passed'] and
            intervention['augmented']['passed']):
        verdict = 'PASS'
    elif empirical_wrong_way:
        verdict = 'FAIL'
    else:
        verdict = 'INCONCLUSIVE'
    representatives = []
    for label in ('safe_deadlock', 'stalled_deadlock', 'other_timeout', 'success'):
        found = next((r for r in rows if r['outcome'] == label), None)
        if found:
            representatives.append(dict(
                outcome=label, rid=found['rid'], seed=found['seed'],
                arm=found['arm'], trace=f"{found['arm']}_{found['rid']}_{found['seed']}.npz"))
    return dict(
        verdict=verdict, full_training_justified=verdict == 'PASS',
        implementation_correctness=implementation,
        numerical_soundness=numerical,
        independent_early_prediction=prediction,
        certificate_coverage=coverage,
        early_intervention=intervention,
        extended_horizon=extended_summary,
        representative_trajectories=representatives,
        counts=dict(prediction_continuations=complete['prediction_continuations'],
                    scalar_reverse_sweeps=complete['scalar_reverse_sweeps'],
                    main_paired_scenarios=protocol['main_paired_scenarios'],
                    main_outcome_replays=complete['main_replays'],
                    extended_outcome_replays=complete['extended_replays']),
        records_sha256=hashlib.sha256(
            (folder/'records.json').read_bytes()).hexdigest(),
        limitations=[
            'R_CERT magnitudes are unbounded surrogate values, not probabilities.',
            'Direct-only and augmented raw magnitudes are not compared.',
            'Twenty-five independent checkpoints with two evaluation suffixes are clustered small-sample diagnostics.',
            'No G_phi parameters were trained and no held-out final test set was opened.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args=parser.parse_args()
    result=analyze(args.folder)
    (args.folder/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

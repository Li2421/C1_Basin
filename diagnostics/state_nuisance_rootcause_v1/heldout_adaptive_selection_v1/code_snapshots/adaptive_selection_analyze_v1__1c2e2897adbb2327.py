"""Held-out adaptive selection audit; reads frozen predictors and real rollouts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'heldout_adaptive_selection_v1'
PRIOR = HERE / 'real_state_expansion'
N = 256
ETA = 16
SEEDS = 8
BOOTSTRAP = 5000
BOOTSTRAP_ROOT = 202610071449


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def half(outcomes, lo, hi):
    values = outcomes[:, :, lo:hi]
    n = hi - lo
    success = values[..., 0].sum(-1).astype(int)
    unknown = values[..., 1].sum(-1).astype(int)
    failure = n - success - unknown
    assert np.all(failure >= 0)
    return success, failure, unknown


def bootstrap_mean(values, seed):
    values = np.asarray(values, float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    return np.quantile(values[draws].mean(1), [.025, .975]).tolist()


def bootstrap_ratio(numerator, denominator, seed):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(numerator), size=(BOOTSTRAP, len(numerator)))
    num = numerator[draws].sum(1)
    den = denominator[draws].sum(1)
    vals = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    vals = vals[np.isfinite(vals)]
    return np.quantile(vals, [.025, .975]).tolist() if len(vals) else None


def selected_values(success, failure, unknown, choices):
    """Rows are optimizer fits; columns are independent physical states."""
    rows = np.arange(success.shape[0])[None, :]
    s = success[rows, choices]
    f = failure[rows, choices]
    u = unknown[rows, choices]
    assert np.all(s + f + u == SEEDS)
    lower, upper = s / SEEDS, (s + u) / SEEDS
    valid = np.divide(s, s + f, out=np.full(s.shape, np.nan, float), where=s + f > 0)
    return lower, upper, valid, u


def ranking_agreement(scores, eval_lower, eval_upper):
    a, b = np.triu_indices(ETA, 1)
    direction = np.where(eval_lower[:, a] > eval_upper[:, b], 1,
                         np.where(eval_lower[:, b] > eval_upper[:, a], -1, 0))
    valid = direction != 0
    predicted = np.sign(scores[:, :, a] - scores[:, :, b])
    correct = np.where(predicted == direction[None, :, :], 1.,
                       np.where(predicted == 0, .5, 0.))
    denominator = int(valid.sum())
    assert denominator > 0
    return dict(decisive_eval_state_eta_pair_cases=denominator,
                by_fit=((correct * valid[None, :, :]).sum((1, 2)) / denominator).tolist(),
                mean=float((correct * valid[None, :, :]).sum((1, 2)).mean() / denominator))


def selector_metrics(scores, choices, index, eval_s, eval_f, eval_u, eval_lower, eval_upper):
    ss, ff, uu = eval_s[index], eval_f[index], eval_u[index]
    lower, upper, valid_rate, selected_unknown = selected_values(ss, ff, uu, choices[:, index])
    oracle = eval_lower[index].max(1)
    eval_best = np.argmax(eval_lower[index], axis=1)
    selected = choices[:, index]
    agreement = ranking_agreement(scores[:, index, :], eval_lower[index], eval_upper[index])
    return dict(states=int(len(index)), top1_rollout_success_lower=float(lower.mean()),
                top1_rollout_success_upper=float(upper.mean()),
                top1_success_lower_by_fit=lower.mean(1).tolist(),
                mean_ground_truth_q_selected_valid=float(np.nanmean(valid_rate)),
                oracle_regret_finite_eval_lower=float((oracle[None, :] - lower).mean()),
                oracle_regret_by_fit=(oracle[None, :] - lower).mean(1).tolist(),
                matches_eval_best_eta=float((selected == eval_best[None, :]).mean()),
                matches_eval_best_by_fit=(selected == eval_best[None, :]).mean(1).tolist(),
                pairwise_preference_ranking_agreement=agreement,
                selected_numerical_unknown=int(selected_unknown.sum()))


def score_delta(eval_s, index, adaptive_choices, comparator_choice):
    rows = np.arange(len(index))[None, :]
    selected = eval_s[index][rows, adaptive_choices[:, index]] / SEEDS
    if isinstance(comparator_choice, (int, np.integer)):
        fixed = eval_s[index, comparator_choice][None, :] / SEEDS
    else:
        fixed = eval_s[index][rows, comparator_choice[:, index]] / SEEDS
    difference = selected - fixed
    per_state = difference.mean(0)
    return dict(mean=float(per_state.mean()), by_fit=difference.mean(1).tolist(),
                paired_physical_state_bootstrap_95=bootstrap_mean(
                    per_state, BOOTSTRAP_ROOT + int(index.sum()) + len(index)),
                positive_equal_negative_states=[
                    [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())]
                    for d in difference],
                state_count=len(index))


def all_fixed_falsification(eval_s, eval_u, index, adaptive_choices):
    ss = eval_s[index]
    uu = eval_u[index]
    rows = np.arange(len(index))[None, :]
    adaptive_per_state = (ss[rows, adaptive_choices[:, index]] / SEEDS).mean(0)
    fixed = ss / SEEDS
    table = []
    for eta in range(ETA):
        delta = adaptive_per_state - fixed[:, eta]
        table.append(dict(eta_index=eta, success_lower=float(fixed[:, eta].mean()),
                          success_upper=float(((ss[:, eta] + uu[:, eta]) / SEEDS).mean()),
                          adaptive_minus_this_fixed=float(delta.mean()),
                          paired_physical_state_bootstrap_95=bootstrap_mean(
                              delta, BOOTSTRAP_ROOT + 1000 + eta + len(index))))
    hindsight = int(np.argmax(fixed.mean(0)))
    rng = np.random.default_rng(BOOTSTRAP_ROOT + 9191 + len(index))
    draws = rng.integers(0, len(index), size=(BOOTSTRAP, len(index)))
    contrast = adaptive_per_state[draws].mean(1) - fixed[draws].mean(1).max(1)
    return dict(all_16_policies=table, hindsight_eval_best_fixed_eta_descriptive=hindsight,
                adaptive_minus_hindsight_best_fixed_point=float(adaptive_per_state.mean() - fixed.mean(0).max()),
                simultaneous_max_over_16_fixed_state_bootstrap_95=np.quantile(contrast, [.025, .975]).tolist(),
                stronger_than_every_fixed_at_95=bool(np.quantile(contrast, .025) > 0))


def selection_behavior(choices, preference, index):
    chosen = choices[:, index]
    by_fit = []
    for fit in chosen:
        counts = np.bincount(fit, minlength=ETA)
        rates = counts / len(fit)
        nonzero = rates[rates > 0]
        entropy = -float((nonzero * np.log2(nonzero)).sum())
        changes = 1 - float((counts * (counts - 1)).sum() /
                            (len(fit) * (len(fit) - 1))) if len(fit) > 1 else None
        by_fit.append(dict(eta_choice_counts=counts.tolist(), entropy_bits=entropy,
                           different_eta_on_two_random_states=changes))
    groups = {}
    for eta in range(ETA):
        ix = index[preference[index] == eta]
        if len(ix):
            groups[str(eta)] = dict(states=int(len(ix)),
                                    choice_counts_by_fit=[np.bincount(c[ix], minlength=ETA).tolist()
                                                          for c in choices])
    return dict(by_fit=by_fit, discovery_preference_groups=groups)


def reversal_audit(discovery_lower, discovery_upper, eval_lower, eval_upper,
                   score_dict, choice_dict, eval_success, fixed_eta, fixed_best, gap):
    a, b = np.triu_indices(ETA, 1)
    forward = discovery_lower[:, a] - discovery_upper[:, b] >= gap
    reverse = discovery_lower[:, b] - discovery_upper[:, a] >= gap
    assert not np.any(forward & reverse)
    directions = np.where(forward, 1, np.where(reverse, -1, 0))
    pair_mask = (forward.sum(0) > 0) & (reverse.sum(0) > 0)
    selected = directions[:, pair_mask]
    pair_a, pair_b = a[pair_mask], b[pair_mask]
    state_mask = (selected != 0).any(1)
    if not pair_mask.any():
        return dict(strong_reversal_pairs=0, strong_reversal_states=0,
                    strong_reversal_state_indices=[], strong_pair_state_cases=0,
                    pair_rows=[], models={})
    eval_a = eval_lower[:, pair_a] - eval_upper[:, pair_b] > 0
    eval_b = eval_lower[:, pair_b] - eval_upper[:, pair_a] > 0
    eval_direction = np.where(eval_a, 1, np.where(eval_b, -1, 0))
    decisive = (selected != 0) & (eval_direction != 0)
    confirmed = decisive & (selected == eval_direction)
    contradicted = decisive & (selected == -eval_direction)
    rows = []
    model_pair_signs = {}
    for name, scores in score_dict.items():
        model_pair_signs[name] = np.sign(scores[:, :, pair_a] - scores[:, :, pair_b])
    adaptive_gain_vs_eta = (
        eval_success[np.arange(N)[None, :], choice_dict['qstate512']].mean(0)
        - eval_success[:, fixed_eta]) / SEEDS
    adaptive_gain_vs_fixed = (
        eval_success[np.arange(N)[None, :], choice_dict['qstate512']].mean(0)
        - eval_success[:, fixed_best]) / SEEDS
    for p, (eta_a, eta_b) in enumerate(zip(pair_a, pair_b)):
        ia = np.flatnonzero(selected[:, p] == 1)
        ib = np.flatnonzero(selected[:, p] == -1)
        assert len(ia) and len(ib)
        rows.append(dict(eta_a=int(eta_a), eta_b=int(eta_b), group_A_states=len(ia),
                         group_B_states=len(ib),
                         eval_decisive_A=int(decisive[ia, p].sum()),
                         eval_decisive_B=int(decisive[ib, p].sum()),
                         adaptive_gain_vs_qeta_A=float(adaptive_gain_vs_eta[ia].mean()),
                         adaptive_gain_vs_qeta_B=float(adaptive_gain_vs_eta[ib].mean()),
                         adaptive_gain_vs_fixed_A=float(adaptive_gain_vs_fixed[ia].mean()),
                         adaptive_gain_vs_fixed_B=float(adaptive_gain_vs_fixed[ib].mean()),
                         qstate512_correct_disc_A=float((model_pair_signs['qstate512'][:, ia, p] == 1).mean()),
                         qstate512_correct_disc_B=float((model_pair_signs['qstate512'][:, ib, p] == -1).mean())))
    # Representative pair comes from discovery support alone; never from model
    # accuracy, evaluation outcomes, or policy gains.
    representative = max(rows, key=lambda r: (min(r['group_A_states'], r['group_B_states']),
                                              r['group_A_states'] + r['group_B_states'],
                                              -r['eta_a'], -r['eta_b']))
    models = {}
    switch_contributions = {}
    for name, signs in model_pair_signs.items():
        correct_discovery = (signs == selected[None, :, :]) & (selected[None, :, :] != 0)
        correct_eval = (signs == eval_direction[None, :, :]) & decisive[None, :, :]
        neutral_discovery = (signs == 0) & (selected[None, :, :] != 0)
        neutral_eval = (signs == 0) & decisive[None, :, :]
        correct_eval_weighted = correct_eval.astype(float) + .5 * neutral_eval.astype(float)
        case_weights = decisive.sum(1)
        score_by_state = correct_eval_weighted.sum(2).mean(0)
        switch_contributions[name] = score_by_state
        total = int(decisive.sum())
        assert total > 0
        per_pair_correct_A = ((signs == 1) * (selected[None, :, :] == 1)).sum(1)
        per_pair_correct_B = ((signs == -1) * (selected[None, :, :] == -1)).sum(1)
        count_A = (selected == 1).sum(0)
        count_B = (selected == -1).sum(0)
        both_majority = (per_pair_correct_A > count_A[None, :] / 2) & (per_pair_correct_B > count_B[None, :] / 2)
        models[name] = dict(
            discovery_direction_accuracy_by_fit=(
                (correct_discovery.astype(float) + .5 * neutral_discovery.astype(float)).sum((1, 2))
                / (selected != 0).sum()).tolist(),
            independent_eval_switch_accuracy_by_fit=(correct_eval_weighted.sum((1, 2)) / total).tolist(),
            independent_eval_switch_accuracy=float(score_by_state.sum() / case_weights.sum()),
            independent_eval_switch_accuracy_state_bootstrap_95=bootstrap_ratio(
                score_by_state, case_weights, BOOTSTRAP_ROOT + 1212),
            pair_fraction_majority_correct_on_both_discovery_groups_by_fit=both_majority.mean(1).tolist())
    gain_A = (selected == 1).sum(1)
    gain_B = (selected == -1).sum(1)
    reversal_gains = {}
    for label, values in [('qeta', adaptive_gain_vs_eta), ('best_fixed', adaptive_gain_vs_fixed)]:
        reversal_gains[label] = {}
        for side, weights in [('A', gain_A), ('B', gain_B)]:
            numerator = values * weights
            reversal_gains[label][side] = dict(
                mean=float(numerator.sum() / weights.sum()),
                paired_physical_state_bootstrap_95=bootstrap_ratio(
                    numerator, weights, BOOTSTRAP_ROOT + 2222 + (0 if label == 'qeta' else 7)
                    + (0 if side == 'A' else 1)),
                case_count=int(weights.sum()),
                distinct_states=int((weights > 0).sum()))
    switch_comparisons = {}
    for label in ('qeta', 'best_fixed', 'qstate32'):
        delta = switch_contributions['qstate512'] - switch_contributions[label]
        switch_comparisons[f'qstate512_minus_{label}'] = dict(
            mean=float(delta.sum() / decisive.sum()),
            paired_physical_state_bootstrap_95=bootstrap_ratio(
                delta, decisive.sum(1), BOOTSTRAP_ROOT + 3333 +
                ('qeta', 'best_fixed', 'qstate32').index(label)))
    return dict(strong_reversal_pairs=int(pair_mask.sum()),
                strong_reversal_states=int(state_mask.sum()),
                strong_reversal_state_indices=np.flatnonzero(state_mask).tolist(),
                strong_pair_state_cases=int((selected != 0).sum()),
                independent_eval_decisive_cases=int(decisive.sum()),
                independent_eval_confirmed_cases=int(confirmed.sum()),
                independent_eval_contradicted_cases=int(contradicted.sum()),
                independent_eval_ambiguous_cases=int(((selected != 0) & (eval_direction == 0)).sum()),
                independent_eval_confirmation_rate=float(confirmed.sum() / (selected != 0).sum()),
                representative_pair_selected_only_by_discovery_support=representative,
                pair_rows=rows, models=models,
                paired_switch_accuracy_differences=switch_comparisons,
                adaptive_gain_by_reversal_side=reversal_gains,
                pairs_with_positive_qstate512_gain_on_both_sides_vs_qeta=int(sum(
                    r['adaptive_gain_vs_qeta_A'] > 0 and r['adaptive_gain_vs_qeta_B'] > 0 for r in rows)),
                pairs_with_positive_qstate512_gain_on_both_sides_vs_fixed=int(sum(
                    r['adaptive_gain_vs_fixed_A'] > 0 and r['adaptive_gain_vs_fixed_B'] > 0 for r in rows)),
                adaptive_gain_vs_qeta_by_discovery_direction=dict(
                    A=float(np.average(np.broadcast_to(adaptive_gain_vs_eta[:, None], selected.shape), weights=selected == 1)),
                    B=float(np.average(np.broadcast_to(adaptive_gain_vs_eta[:, None], selected.shape), weights=selected == -1))),
                adaptive_gain_vs_fixed_by_discovery_direction=dict(
                    A=float(np.average(np.broadcast_to(adaptive_gain_vs_fixed[:, None], selected.shape), weights=selected == 1)),
                    B=float(np.average(np.broadcast_to(adaptive_gain_vs_fixed[:, None], selected.shape), weights=selected == -1))))


def main():
    plan = read(OUT / 'analysis_plan.json')
    protocol = read(OUT / 'protocol.json')
    implementation = read(OUT / 'implementation_freeze.json')
    assert implementation['analyzer_sha256'] == sha(__file__)
    assert implementation['analysis_plan_sha256'] == sha(OUT / 'analysis_plan.json')
    collection = read(OUT / 'collection_audit.json')
    prediction_audit = read(OUT / 'prediction_audit.json')
    assert plan['physical_state_count'] == N
    assert plan['bootstrap_replicates'] == BOOTSTRAP
    assert sha(OUT / 'analysis_plan.json') == protocol['analysis_plan_sha256']
    assert sha(OUT / 'fresh_rollouts.npz') == collection['dataset_sha256']
    assert sha(OUT / 'frozen_predictions.npz') == prediction_audit['prediction_file_sha256']
    assert collection['all_cases_match_authoritative_DB']
    data = np.load(OUT / 'fresh_rollouts.npz')
    predictions = np.load(OUT / 'frozen_predictions.npz')
    outcomes = data['outcomes']
    assert outcomes.shape == (N, ETA, 16, 3)
    assert np.all(data['present'])
    assert np.array_equal(data['eta'], predictions['eta'])
    assert np.array_equal(data['state_uids'], predictions['state_uids'])
    assert len(set(data['state_uids'])) == N
    discovery_s, discovery_f, discovery_u = half(outcomes, 0, 8)
    eval_s, eval_f, eval_u = half(outcomes, 8, 16)
    discovery_lower = discovery_s / SEEDS
    discovery_upper = (discovery_s + discovery_u) / SEEDS
    eval_lower = eval_s / SEEDS
    eval_upper = (eval_s + eval_u) / SEEDS
    aggregate_lower = discovery_s.sum(0) / (N * SEEDS)
    aggregate_upper = (discovery_s + discovery_u).sum(0) / (N * SEEDS)
    fixed_best = int(np.argmax(aggregate_lower))
    qeta_logits = predictions['eta_only_n512_logits']
    eta_only = int(np.argmax(qeta_logits))
    preference = np.argmax(discovery_lower, axis=1)
    tie_count = (discovery_lower == discovery_lower.max(1, keepdims=True)).sum(1)
    sensitivity = {}
    for threshold in plan['adaptive_demand_sensitivity_gaps'] + [plan['adaptive_demand_primary_gap']]:
        mask = discovery_lower.max(1) - discovery_upper[:, fixed_best] >= threshold
        sensitivity[str(threshold)] = dict(count=int(mask.sum()), state_indices=np.flatnonzero(mask).tolist())
    adaptive = discovery_lower.max(1) - discovery_upper[:, fixed_best] >= plan['adaptive_demand_primary_gap']

    scores = dict(qeta=np.broadcast_to(qeta_logits, (1, N, ETA)),
                  best_fixed=np.broadcast_to(aggregate_lower, (1, N, ETA)),
                  qstate512=predictions['physical23_eta_n512_logits'],
                  qstate32=predictions['physical23_eta_n32_logits'])
    choices = {name: value.argmax(-1) for name, value in scores.items()}
    assert np.all(choices['qeta'] == eta_only)
    assert np.all(choices['best_fixed'] == fixed_best)
    assert scores['qstate512'].shape == scores['qstate32'].shape == (3, N, ETA)
    reversal = reversal_audit(discovery_lower, discovery_upper, eval_lower, eval_upper,
                              scores, choices, eval_s, eta_only, fixed_best,
                              plan['strong_pair_reversal_gap'])
    reversal_indices = np.asarray(reversal.get('strong_reversal_state_indices', []), dtype=int)
    cohorts = {'all': np.arange(N), 'adaptive_demand': np.flatnonzero(adaptive),
               'strong_reversal': reversal_indices}
    cohort_results = {}
    for name, index in cohorts.items():
        if not len(index):
            cohort_results[name] = dict(states=0, state_indices=[],
                                        status='No naturally qualifying states under frozen discovery rule')
            continue
        measures = {selector: selector_metrics(scores[selector], choices[selector], index,
                                               eval_s, eval_f, eval_u, eval_lower, eval_upper)
                    for selector in scores}
        comparisons = dict(qstate512_minus_qeta=score_delta(eval_s, index, choices['qstate512'], eta_only),
                           qstate512_minus_discovery_best_fixed=score_delta(
                               eval_s, index, choices['qstate512'], fixed_best),
                           qstate512_minus_qstate32=score_delta(
                               eval_s, index, choices['qstate512'], choices['qstate32']))
        cohort_results[name] = dict(states=len(index), state_indices=index.tolist(),
                                    selectors=measures, comparisons=comparisons,
                                    all_fixed=all_fixed_falsification(eval_s, eval_u, index,
                                                                     choices['qstate512']))

    a, b = np.triu_indices(ETA, 1)
    strong_fwd = discovery_lower[:, a] - discovery_upper[:, b] >= plan['strong_pair_reversal_gap']
    strong_rev = discovery_lower[:, b] - discovery_upper[:, a] >= plan['strong_pair_reversal_gap']
    pairs_with_both = (strong_fwd.sum(0) > 0) & (strong_rev.sum(0) > 0)
    all_pair_rows = [dict(eta_a=int(a[p]), eta_b=int(b[p]),
                          strong_A_states=int(strong_fwd[:, p].sum()),
                          strong_B_states=int(strong_rev[:, p].sum()),
                          natural_reversal=bool(pairs_with_both[p]))
                     for p in range(len(a))]
    natural = dict(global_best_fixed_discovery_eta=fixed_best,
                   qeta_train_selected_eta=eta_only,
                   aggregate_discovery_q_lower=aggregate_lower.tolist(),
                   aggregate_discovery_q_upper=aggregate_upper.tolist(),
                   preferred_eta_by_state=preference.tolist(),
                   preferred_eta_frequency=np.bincount(preference, minlength=ETA).tolist(),
                   preferred_eta_tie_count_frequency=np.bincount(tie_count, minlength=ETA + 1).tolist(),
                   preferred_differs_from_fixed_count=int((preference != fixed_best).sum()),
                   preferred_differs_from_fixed_distinct_etas=np.flatnonzero(
                       np.bincount(preference[preference != fixed_best], minlength=ETA)).tolist(),
                   adaptive_demand_preferred_eta_frequency=np.bincount(
                       preference[adaptive], minlength=ETA).tolist(),
                   adaptive_demand_distinct_preferred_eta_count=int(
                       np.unique(preference[adaptive]).size),
                   adaptive_demand_sensitivity=sensitivity,
                   strong_preference_pairs_with_reversal=int(pairs_with_both.sum()),
                   strong_pairwise_direction_cases_by_state=(strong_fwd | strong_rev).sum(1).tolist(),
                   strong_pairwise_reversal_state_cases=int(((strong_fwd | strong_rev)[:, pairs_with_both]).sum()),
                   discovery_numerical_unknown=int(discovery_u.sum()),
                   evaluation_numerical_unknown=int(eval_u.sum()))
    selection = {name: selection_behavior(choices[name], preference, np.arange(N))
                 for name in ('qeta', 'best_fixed', 'qstate512', 'qstate32')}

    # Every state/eta/seed remains available in the NPZ and raw JSON. Compact
    # CSVs make the label and prediction correspondence easy to inspect.
    with (OUT / 'state_eta_results.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['state_index', 'state_uid', 'draw_seed', 'eta_index',
                         'eta1', 'eta2', 'eta3', 'disc_success', 'disc_failure',
                         'disc_unknown', 'eval_success', 'eval_failure', 'eval_unknown',
                         'disc_q_lower', 'disc_q_upper', 'eval_q_lower', 'eval_q_upper',
                         'qeta_logit', 'qstate512_seed17', 'qstate512_seed23',
                         'qstate512_seed41', 'qstate32_seed17',
                         'qstate32_seed23', 'qstate32_seed41'])
        for i in range(N):
            for j in range(ETA):
                writer.writerow([i, data['state_uids'][i], data['state_draw_seeds'][i], j,
                                 *data['eta'][j], discovery_s[i, j], discovery_f[i, j],
                                 discovery_u[i, j], eval_s[i, j], eval_f[i, j],
                                 eval_u[i, j], discovery_lower[i, j], discovery_upper[i, j],
                                 eval_lower[i, j], eval_upper[i, j], qeta_logits[j],
                                 *scores['qstate512'][:, i, j], *scores['qstate32'][:, i, j]])
    with (OUT / 'seed_outcomes.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['state_index', 'state_uid', 'draw_seed', 'eta_index',
                         'seed_index', 'split', 'success', 'numerical_unknown', 'collision'])
        for i in range(N):
            for j in range(ETA):
                for k in range(16):
                    writer.writerow([i, data['state_uids'][i], data['state_draw_seeds'][i],
                                     j, k, 'discovery' if k < 8 else 'evaluation',
                                     *outcomes[i, j, k]])
    with (OUT / 'all_strong_reversal_pairs.csv').open('w') as file:
        if reversal['pair_rows']:
            writer = csv.DictWriter(file, fieldnames=list(reversal['pair_rows'][0]))
            writer.writeheader()
            writer.writerows(reversal['pair_rows'])
    with (OUT / 'all_120_eta_pairs.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_pair_rows[0]))
        writer.writeheader()
        writer.writerows(all_pair_rows)

    state_gain = (eval_s[np.arange(N)[None, :], choices['qstate512']].mean(0)
                  - eval_s[:, eta_only]) / SEEDS
    per_state_regret = eval_lower.max(1) - (
        eval_s[np.arange(N)[None, :], choices['qstate512']] / SEEDS).mean(0)
    failure_order = sorted(range(N), key=lambda i: (state_gain[i], -per_state_regret[i], i))
    failures = [dict(state_index=i, state_uid=str(data['state_uids'][i]),
                     discovery_preferred_eta=int(preference[i]),
                     qeta_choice=eta_only, discovery_fixed_choice=fixed_best,
                     qstate512_choices=choices['qstate512'][:, i].tolist(),
                     qstate32_choices=choices['qstate32'][:, i].tolist(),
                     qstate512_minus_qeta_eval_success=float(state_gain[i]),
                     qstate512_eval_oracle_regret=float(per_state_regret[i]))
                for i in failure_order[:12]]

    result = dict(schema='heldout_adaptive_selection_results_v1',
                  scope='256 new globally sampled Ring true-t0 physical states; fixed FF execution and 16 frozen eta',
                  definitions=plan, sampling_audit=dict(
                      state_count=N, draw_seed_start=int(data['state_draw_seeds'][0]),
                      draw_seed_end=int(data['state_draw_seeds'][-1]),
                      no_overlap_with_previous_608=True,
                      no_outcome_based_sampling_or_rejection=True,
                      generator_sha256=protocol['generator_sha256']),
                  dataset_audit=collection, prediction_audit=prediction_audit,
                  natural_eta_preference=natural,
                  cohorts=cohort_results, reversal=reversal,
                  selection_behavior=selection, worst_failure_cases=failures,
                  interpretation_limits=[
                      'TEST discovery defines groups and fixed policy before opening its independent evaluation half; subgroup analysis is not a prospectively trained policy.',
                      'Eight evaluation seeds estimate success probabilities with Monte Carlo noise; finite-seed oracle is optimistic.',
                      'Physical state, not state-eta pair or rollout, is the bootstrap unit.',
                      'No new critic training, controller change, Flow change, safety change, eta-pool change, or targeted state sampling.'])
    write(OUT / 'audit_results.json', result)
    concise = {cohort: dict(states=row['states'],
                            qeta=row['selectors']['qeta']['top1_rollout_success_lower'],
                            fixed=row['selectors']['best_fixed']['top1_rollout_success_lower'],
                            qstate512=row['selectors']['qstate512']['top1_rollout_success_lower'],
                            delta_vs_qeta=row['comparisons']['qstate512_minus_qeta'],
                            delta_vs_fixed=row['comparisons']['qstate512_minus_discovery_best_fixed'],
                            delta_vs_hindsight_best_fixed=row['all_fixed']['adaptive_minus_hindsight_best_fixed_point'],
                            simultaneous_fixed_CI=row['all_fixed']['simultaneous_max_over_16_fixed_state_bootstrap_95'])
               for cohort, row in cohort_results.items() if row['states']}
    print(json.dumps(dict(natural=natural, cohorts=concise,
                          strong_reversal_pairs=reversal['strong_reversal_pairs'],
                          switch_accuracy=(reversal['models']['qstate512']['independent_eval_switch_accuracy']
                                           if reversal['models'] else None)),
                     indent=2))


if __name__ == '__main__':
    main()

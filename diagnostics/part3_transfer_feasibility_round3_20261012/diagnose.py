"""Read-only secondary diagnostics from frozen outer predictions and archived labels."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'controller_q_transfer_round3_v1'
R2 = HERE.parent / 'part3_necessity_representation_round2_20261012'


def boot(a):
    a = np.asarray(a, float)
    draws = np.random.default_rng(20261012).integers(0, len(a), (20000, len(a)))
    return [float(v) for v in np.quantile(a[draws].mean(axis=1), [.025, .975])]


def main():
    result = json.loads((HERE / 'results.json').read_text())
    rows = result['per_checkpoint']
    with np.load(P1 / 'P1_LABELS.npz') as z:
        y, u = z['Y'], z['U']
    with np.load(HERE / 'outer_predictions.npz') as z:
        pred, arms = z['predictions'], z['arms'].tolist()
    assert pred.shape == (6, 14, 24, 16)
    qlo = y.mean(-1)
    qhi = (y + u).mean(-1)
    qmid = (qlo + qhi) / 2
    sortedq = np.sort(qmid, axis=-1)
    gap = sortedq[..., -1] - sortedq[..., -2]
    half_a = (y[..., :2] + .5 * u[..., :2]).mean(-1)
    half_b = (y[..., 2:] + .5 * u[..., 2:]).mean(-1)
    agreement = np.argmax(half_a, axis=-1) == np.argmax(half_b, axis=-1)
    def residual(q):
        return q - q.mean(axis=-1, keepdims=True) - q.mean(axis=-2, keepdims=True) + q.mean(axis=(-2, -1), keepdims=True)
    reproducible_residual_energy = float(np.mean(residual(half_a) * residual(half_b)))
    residual_by_checkpoint = [{
        'held': rows[i]['held'],
        'half_cross_product': float(np.mean(residual(half_a[i]) * residual(half_b[i]))),
        'R3_prediction_target_Q4_cross_product': float(np.mean(residual(pred[arms.index('R3'), i]) * residual(qmid[i]))),
        'R5_prediction_target_Q4_cross_product': float(np.mean(residual(pred[arms.index('R5'), i]) * residual(qmid[i]))),
    } for i in range(14)]
    residual_predictive = {}
    for ai, arm in enumerate(arms):
        pr = residual(pred[ai])
        residual_predictive[arm] = {
            'predicted_residual_energy': float(np.mean(pr**2)),
            'prediction_target_half_A_cross_product': float(np.mean(pr * residual(half_a))),
            'prediction_target_half_B_cross_product': float(np.mean(pr * residual(half_b))),
            'prediction_target_Q4_residual_MSE': float(np.mean((pr - residual(qmid))**2))}
    fixed_stable = []
    for held in range(14):
        src = [i for i in range(14) if i != held]
        fixed_stable.append({'held': rows[held]['held'],
                             'lower': int(np.argmax(qlo[src].mean(axis=(0, 1)))),
                             'mid': rows[held]['source_fixed_eta'],
                             'upper': int(np.argmax(qhi[src].mean(axis=(0, 1))))})
    pairwise = {}
    for base, target in [('R3', 'R5'), ('R3', 'R3_near'), ('R0', 'R5'), ('R3', 'R3_R4')]:
        a = np.array([r['arms'][base]['q_mse_mid'] - r['arms'][target]['q_mse_mid'] for r in rows])
        v = np.array([r['arms'][target]['top1_Q_interval'][0] - r['arms'][base]['top1_Q_interval'][0] for r in rows])
        pairwise[f'{target}_minus_{base}'] = {
            'MSE_reduction_mean': float(a.mean()), 'MSE_reduction_checkpoint_bootstrap_95': boot(a),
            'MSE_reduction_positive_checkpoint_count': int((a > 0).sum()),
            'selected_Q_lower_gain_mean': float(v.mean()), 'selected_Q_lower_gain_checkpoint_bootstrap_95': boot(v),
            'selected_Q_lower_gain_positive_checkpoint_count': int((v > 0).sum()),
            'selected_Q_lower_gain_negative_checkpoint_count': int((v < 0).sum())}
    old = json.loads((R2 / 'representation_results.json').read_text())['loco_exploratory']
    nearest = {arm: {'mean_Q_MSE': old[arm]['mean_Q_MSE'],
                     'mean_selected_minus_source_fixed': old[arm]['mean_selected_minus_source_fixed']}
               for arm in ('R3', 'R4_action', 'R5')}
    selected = {arm: Counter(int(j) for r in rows for j in r['arms'][arm]['selected_eta']).most_common()
                for arm in arms}
    out = {'scope': 'Secondary descriptive diagnostics after frozen source-only outer LOCO; no gate effect.',
           'old_nearest_source': nearest, 'pairwise': pairwise,
           'mean_source_fixed_Q_interval': np.mean([r['source_fixed_Q_interval'] for r in rows], axis=0).tolist(),
           'mean_empirical_Q4_oracle_interval': np.mean([r['target_Q4_oracle_interval'] for r in rows], axis=0).tolist(),
           'empirical_Q4_oracle_minus_source_fixed_lower_mid': float(qlo.max(axis=-1).mean() - np.mean([r['source_fixed_Q_interval'][0] for r in rows])),
           'Q4_top_two_mid_gap_median': float(np.median(gap)),
           'Q4_top_two_mid_gap_le_0_25_fraction': float(np.mean(gap <= .25)),
           'Q2_half_argmax_agreement_fraction': float(agreement.mean()),
           'Q2_half_reproducible_state_eta_residual_energy': reproducible_residual_energy,
           'state_eta_residual_by_checkpoint': residual_by_checkpoint,
           'state_eta_residual_prediction': residual_predictive,
           'target_crossfit_headroom_lower_positive_checkpoint_count': int(sum(r['target_half_crossfit_oracle_minus_source_fixed'][0] > 0 for r in rows)),
           'target_crossfit_headroom_upper_negative_checkpoint_count': int(sum(r['target_half_crossfit_oracle_minus_source_fixed'][1] < 0 for r in rows)),
           'source_fixed_index_under_unknown_assignments': fixed_stable,
           'selected_eta_counts': {arm: [[j, count] for j, count in selected[arm]] for arm in arms},
           'decomposition_mean': {arm: {term: float(np.mean([r['arms'][arm]['mse_decomposition'][term] for r in rows]))
                                       for term in rows[0]['arms'][arm]['mse_decomposition']} for arm in arms},
           'chosen_config_counts': {arm: {'metric': dict(Counter(r['chosen'][arm]['metric'] for r in rows)),
                                          'inverse_temperature': dict(Counter(str(r['chosen'][arm]['inverse_temperature']) for r in rows)),
                                          'match_weight': dict(Counter(str(r['chosen'][arm]['match_weight']) for r in rows))}
                                    for arm in arms if arm != 'R0'}}
    (HERE / 'diagnostics.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: out[k] for k in ('pairwise', 'Q4_top_two_mid_gap_median',
        'Q4_top_two_mid_gap_le_0_25_fraction', 'Q2_half_argmax_agreement_fraction')}, indent=2))


if __name__ == '__main__':
    main()

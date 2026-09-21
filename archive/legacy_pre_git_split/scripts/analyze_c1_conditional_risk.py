"""Independent-noise association on a frozen development experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def weighted_auc(scores, deadlocks, repeats):
    differences = scores[:, None]-scores[None, :]
    weights = deadlocks[:, None]*(repeats-deadlocks)[None, :]
    if weights.sum() == 0:
        return None
    return float(np.sum(weights*((differences > 0)+.5*(differences == 0)))/weights.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    folder = args.folder
    protocol = json.loads((folder/'protocol.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    records = json.loads((folder/'records.json').read_text())
    assert complete['rollouts'] == len(records) == protocol['total_rollouts'] == 400
    assert not set(protocol['prediction_seeds']) & set(protocol['outcome_seeds'])
    predictions, outcomes, prediction_events = [], [], []
    for rid in range(40):
        rows = {r['seed']: r for r in records if r['rid'] == rid}
        assert len(rows) == 10
        predictions.append([rows[s]['risk'] for s in protocol['prediction_seeds']])
        prediction_events.append([rows[s]['either_deadlock'] for s in protocol['prediction_seeds']])
        outcomes.append([rows[s]['either_deadlock'] for s in protocol['outcome_seeds']])
    predictions, outcomes = np.asarray(predictions), np.asarray(outcomes, dtype=int)
    deadlocks = outcomes.sum(axis=1)
    prediction_events = np.asarray(prediction_events, dtype=float)
    summaries = {}
    for count in [1, 4]:
        scores = predictions[:, :count].mean(axis=1)
        rho = correlation(scores, deadlocks)
        auc = weighted_auc(scores, deadlocks, 6)
        rng = np.random.default_rng(2026091803)
        boot_rho, boot_auc = [], []
        for _ in range(2000):
            idx = rng.integers(0, 40, 40)
            r = correlation(scores[idx], deadlocks[idx])
            a = weighted_auc(scores[idx], deadlocks[idx], 6)
            if r is not None: boot_rho.append(r)
            if a is not None: boot_auc.append(a)
        null = [correlation(scores, rng.permutation(deadlocks)) for _ in range(5000)] if rho is not None else []
        zeros = scores == 0
        summaries[str(count)] = dict(spearman=rho, auc=auc,
            spearman_ci95=np.quantile(boot_rho, [.025, .975]).tolist() if boot_rho else None,
            auc_ci95=np.quantile(boot_auc, [.025, .975]).tolist() if boot_auc else None,
            spearman_permutation_two_sided_p=(1+sum(abs(r) >= abs(rho) for r in null))/5001 if null else None,
            zero_score_starts=int(zeros.sum()), zero_score_outcome_deadlocks=int(deadlocks[zeros].sum()),
            zero_score_outcome_count=int(6*zeros.sum()),
            unique_scores=int(len(np.unique(scores))),
            per_start_scores=scores.tolist())
        event_score = prediction_events[:, :count].mean(axis=1)
        differences = []
        rng = np.random.default_rng(2026091804)
        for _ in range(2000):
            idx = rng.integers(0, 40, 40)
            risk_auc = weighted_auc(scores[idx], deadlocks[idx], 6)
            event_auc = weighted_auc(event_score[idx], deadlocks[idx], 6)
            if risk_auc is not None and event_auc is not None:
                differences.append(risk_auc-event_auc)
        summaries[str(count)]['predicted_event_frequency_reference'] = dict(
            purpose='nondifferentiable prediction reference only, not a new controller',
            auc=weighted_auc(event_score, deadlocks, 6),
            spearman=correlation(event_score, deadlocks),
            risk_minus_event_auc_paired_ci95=np.quantile(differences, [.025, .975]).tolist()
                if differences else None)
    evaluation = [r for r in records if r['role'] == 'outcome']
    safety_keys = ('agent_collision_steps', 'wall_collision_steps', 'outside_endpoints',
                   'cbf_violations', 'speed_violations')
    safety_totals = {key: sum(r['safety'][key] for r in records) for key in safety_keys}
    if any(safety_totals.values()):
        raise ValueError('Safety failure: cannot publish successful analysis')
    result = dict(scope='Safety-only development prediction; no C1 efficacy claim',
        records_sha256=hashlib.sha256((folder/'records.json').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
        independent_starts=40, outcome_rollouts=240,
        safety_totals_all_400=safety_totals,
        outcome_counts={k:sum(r['outcome']==k for r in evaluation) for k in
                        ['success', 'safe_deadlock', 'stalled_deadlock', 'other_timeout']},
        baseline_difficulty_passed=bool(deadlocks.sum()/240 >= .1 and np.sum(deadlocks > 0) >= 10),
        deadlock_frequencies=(deadlocks/6).tolist(), scores_by_prediction_budget=summaries,
        limitations=['Only 40 starts and six outcome repeats; conditional probabilities are noisy',
                     'Model noise only; no model-mismatch or sensor-noise robustness evidence',
                     'Full future prediction horizon; no 20-second-window validation',
                     'Association cannot prove gradients improve controls or shared-policy training',
                     'Two prediction budgets reported; no unadjusted confirmatory significance claim'])
    (folder/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['deadlock_frequencies', 'scores_by_prediction_budget']}, indent=2))
    for k,v in summaries.items():
        print(k, json.dumps({a:b for a,b in v.items() if a != 'per_start_scores'}))


if __name__ == '__main__':
    main()

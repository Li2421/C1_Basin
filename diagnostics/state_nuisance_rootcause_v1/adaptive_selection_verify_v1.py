"""Independent final identity, split, and primary-metric checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'heldout_adaptive_selection_v1'
PRIOR = HERE / 'real_state_expansion'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / 'audit_results.json').read_text())
    frozen = json.loads((PRIOR / 'models_frozen.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    plan = json.loads((OUT / 'analysis_plan.json').read_text())
    sample = json.loads((OUT / 'sampling_audit.json').read_text())
    collection = json.loads((OUT / 'collection_audit.json').read_text())
    predictions = json.loads((OUT / 'prediction_audit.json').read_text())
    implementation = json.loads((OUT / 'implementation_freeze.json').read_text())
    amendments = json.loads((OUT / 'implementation_amendments.json').read_text())
    assert sample['all_states_regenerated_exactly'] and sample['no_overlap_with_earlier_608']
    assert sha(OUT / 'analysis_plan.json') == protocol['analysis_plan_sha256']
    assert sha(OUT / 'protocol.json') == collection['protocol_sha256']
    assert sha(OUT / 'frozen_predictions.npz') == predictions['prediction_file_sha256']
    assert predictions['inference_backend'] == 'gpu'
    assert predictions['old_fresh_test_logit_parity_max_abs_error'] == 0
    assert sha(HERE / 'adaptive_selection_analyze_v1.py') == implementation['analyzer_sha256']
    assert amendments[-1]['new_sha256'] == implementation['analyzer_sha256']
    assert amendments[-1]['results_not_opened'] and amendments[-1]['rollouts_still_running']
    assert sha(HERE / 'adaptive_selection_predict_v1.py') == implementation['predictor_sha256']
    assert sha(HERE / 'adaptive_selection_audit_v1.py') == implementation['rollout_producer_sha256']
    assert sha(OUT / 'fresh_rollouts.npz') == collection['dataset_sha256']
    assert sha(PRIOR / 'models_frozen.json') == protocol['frozen_models_sha256']
    for name in ('physical23_eta_n512', 'physical23_eta_n32'):
        for model in frozen['selections'][name]['models']:
            assert sha(model['path']) == model['sha256']

    data = np.load(OUT / 'fresh_rollouts.npz')
    pred = np.load(OUT / 'frozen_predictions.npz')
    outcomes = data['outcomes']
    assert outcomes.shape == (256, 16, 16, 3)
    assert np.all(data['present'])
    assert len(set(data['state_uids'])) == 256
    assert np.array_equal(data['state_uids'], pred['state_uids'])
    assert np.array_equal(data['eta'], pred['eta'])
    assert collection['attempted'] == 256 * 16 * 16
    success = outcomes[..., 0]
    unknown = outcomes[..., 1]
    assert int(success.sum()) == collection['successes']
    assert int(unknown.sum()) == collection['numerical_unknown']
    assert int((1 - success - unknown).sum()) == collection['failures']
    assert plan['discovery_seed_indices'] == list(range(8))
    assert plan['evaluation_seed_indices'] == list(range(8, 16))

    discovery_success = success[:, :, :8].sum(-1)
    discovery_unknown = unknown[:, :, :8].sum(-1)
    evaluation_success = success[:, :, 8:].sum(-1)
    discovery_fixed = int(np.argmax(discovery_success.sum(0)))
    qeta = int(np.argmax(pred['eta_only_n512_logits']))
    assert discovery_fixed == result['natural_eta_preference']['global_best_fixed_discovery_eta']
    assert qeta == result['natural_eta_preference']['qeta_train_selected_eta']
    discovery_lower = discovery_success / 8
    discovery_upper = (discovery_success + discovery_unknown) / 8
    demand = np.flatnonzero(discovery_lower.max(1) - discovery_upper[:, discovery_fixed]
                            >= plan['adaptive_demand_primary_gap'])
    assert demand.tolist() == result['cohorts']['adaptive_demand']['state_indices']
    a, b = np.triu_indices(16, 1)
    ab = discovery_lower[:, a] - discovery_upper[:, b] >= plan['strong_pair_reversal_gap']
    ba = discovery_lower[:, b] - discovery_upper[:, a] >= plan['strong_pair_reversal_gap']
    pairs = (ab.sum(0) > 0) & (ba.sum(0) > 0)
    reversal = np.flatnonzero((ab[:, pairs] | ba[:, pairs]).any(1))
    assert reversal.tolist() == result['cohorts']['strong_reversal']['state_indices']
    assert int(pairs.sum()) == result['reversal']['strong_reversal_pairs']

    policy = pred['physical23_eta_n512_logits'].argmax(-1)
    adaptive_per_state = evaluation_success[np.arange(256)[None, :], policy].mean(0) / 8
    qeta_per_state = evaluation_success[:, qeta] / 8
    fixed_per_state = evaluation_success[:, discovery_fixed] / 8
    for cohort, idx in [('all', np.arange(256)), ('adaptive_demand', demand),
                        ('strong_reversal', reversal)]:
        if not len(idx):
            continue
        row = result['cohorts'][cohort]
        np.testing.assert_allclose(adaptive_per_state[idx].mean(),
                                   row['selectors']['qstate512']['top1_rollout_success_lower'],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(qeta_per_state[idx].mean(),
                                   row['selectors']['qeta']['top1_rollout_success_lower'],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(fixed_per_state[idx].mean(),
                                   row['selectors']['best_fixed']['top1_rollout_success_lower'],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose((adaptive_per_state[idx] - qeta_per_state[idx]).mean(),
                                   row['comparisons']['qstate512_minus_qeta']['mean'],
                                   rtol=0, atol=1e-12)
        for eta in range(16):
            np.testing.assert_allclose((evaluation_success[idx, eta] / 8).mean(),
                                       row['all_fixed']['all_16_policies'][eta]['success_lower'],
                                       rtol=0, atol=1e-12)

    with (OUT / 'state_eta_results.csv').open() as file:
        state_eta_rows = sum(1 for _ in csv.DictReader(file))
    with (OUT / 'seed_outcomes.csv').open() as file:
        seed_rows = sum(1 for _ in csv.DictReader(file))
    assert state_eta_rows == 256 * 16
    assert seed_rows == 256 * 16 * 16
    raw_paths = list((OUT / 'raw').glob('*.json'))
    assert len(raw_paths) == seed_rows
    first_raw_ns = min(path.stat().st_mtime_ns for path in raw_paths)
    assert (OUT / 'protocol.json').stat().st_mtime_ns < first_raw_ns
    assert (OUT / 'analysis_plan.json').stat().st_mtime_ns < first_raw_ns
    assert not list(OUT.rglob('*.mp4'))
    report = dict(verified=True, states=256, etas=16, seeds_per_pair=16,
                  state_eta_rows=state_eta_rows, seed_rows=seed_rows,
                  all_labels_match_authoritative_DB=collection['all_cases_match_authoritative_DB'],
                  all_checkpoints_unchanged=True, exact_prior_GPU_logit_parity=True,
                  states_and_rules_frozen_before_first_rollout=True,
                  physical_state_bootstrap_unit=True,
                  all_fixed_policies_compared=True, no_videos=True)
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

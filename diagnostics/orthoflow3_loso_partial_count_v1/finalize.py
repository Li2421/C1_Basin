"""Freeze the partial-count experiment report; never selects using target scores."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import OUT, OLD, ROOT, FOLDS, VARIANTS, baseline, csvout, dump, load, sha


def readcsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def main():
    results = readcsv(OUT / 'loso_results.csv')
    by = {(r['fold'], r['method']): r for r in results}
    audit = load(OUT / 'build_audit.json')
    assert load(OUT / 'source_failure_learning_complete.json')['models_changed'] is False
    assert not load(OUT / 'outcome_version_audit.json')['updates']
    protocol = load(OUT / 'protocol.json')
    assert sha(ROOT / 'diagnostics/orthoflow3_unified_representation_v1/representation.py') == protocol['architecture_sha256']
    for fold in FOLDS:
        assert sha(baseline(fold) / 'final_decision.json') == audit['old_artifact_hashes'][fold]
        assert load(OUT / fold / 'normalization.json') == load(baseline(fold) / 'normalization.json')

    summary = []
    stability = []
    checkpoints = []
    for fold in FOLDS:
        for method in ('old_shared', 'old_source_selected_eta', 'full_continuation_shared',
                       'full_continuation_eta_only', 'partial_count_shared', 'partial_count_eta_only',
                       'negative_binary_shared', 'negative_binary_eta_only'):
            summary.append(by[fold, method])
        for variant in VARIANTS:
            frozen = load(OUT / fold / variant / 'models_frozen.json')
            manifest = load(OUT / fold / variant / 'dataset_manifest.json')
            assert FOLDS[fold] not in manifest['sources']
            assert manifest['target_labels_used'] is False
            for kind in ('shared', 'eta_only'):
                rr = [by[fold, f'{variant}_{kind}_seed{s}'] for s in (17, 23, 41)]
                n = np.array([int(r['B15']) for r in rr])
                u = np.array([int(r['unresolved']) for r in rr])
                stability.append({'fold': fold, 'variant': variant, 'model': kind,
                                  'seed_order': '17;23;41', 'B15_counts': ';'.join(map(str, n)),
                                  'unresolved_counts': ';'.join(map(str, u)),
                                  'B15_mean': float(n.mean()), 'B15_std_sample': float(n.std(ddof=1)),
                                  'B15_median': float(np.median(n)), 'source_VAL_selected_seed': frozen[kind]['selected']['seed']})
                for run in frozen[kind]['runs']:
                    assert sha(run['checkpoint']) == run['sha256']
                    assert run['target_labels_used'] is False
                    checkpoints.append({k: run[k] for k in ('fold', 'variant', 'kind', 'seed', 'best_step',
                                                          'steps', 'seconds', 'validation_nll', 'checkpoint', 'sha256')})
    csvout('primary_summary.csv', summary)
    csvout('seed_stability.csv', stability)
    csvout('checkpoint_inventory.csv', checkpoints)

    refs = readcsv(OLD / 'joint_capacity_diagnostic.csv')
    refs = [{**r, 'provenance': str(OLD / 'joint_capacity_diagnostic.csv'),
             'scope': 'TARGET_SUPERVISED_REFERENCE_NOT_ZERO_SHOT'} for r in refs
            if r['method'] in (f'single_{r["fold"]}_shared', 'joint_enriched_shared', 'joint_shared')]
    csvout('target_supervised_references.csv', refs)

    cache = []
    for fold in FOLDS:
        s = load(OUT / fold / 'cache_preflight.json')['summary']
        cache.append({'fold': fold, **s, 'newly_executed': 0,
                      'unresolved_policy': 'retain interval; no numerical outcome imputation or rerun'})
    csvout('cache_usage.csv', cache)
    totals = {k: sum(r[k] for r in cache) for k in ('total_requested', 'exact_reusable',
                                                  'partial_reusable', 'aggregate_reusable', 'genuinely_missing')}
    assert totals['total_requested'] == 39936
    assert totals['exact_reusable'] + totals['partial_reusable'] == 39745
    assert totals['genuinely_missing'] == 191
    dump('budget_audit.json', {**totals, 'new_rollouts': 0, 'DB_opened_read_only': True,
                              'new_DB_records': 0, 'postflight_new_record_requirement': 'NOT_APPLICABLE_NO_NEW_ROLLOUT',
                              'GPU_shards_used_per_job': 1, 'max_authorized_shards': 6,
                              'successful_jobs': [1728, 1731, 1750],
                              'failed_audit_job': {'job': 1748, 'cause': 'missing sklearn; replaced AUC with scipy rank statistic',
                                                   'training_or_outcomes_changed': False}})

    # Preserve semantics metadata separately without changing the frozen pair-table bytes.
    native = pq.read_table(ROOT / 'datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet',
                           columns=['state_uid', 'eta_uid', 'controller_uid', 'label_semantics_version']).to_pylist()
    native = {(r['state_uid'], r['eta_uid'], r['controller_uid']): r['label_semantics_version'] for r in native}
    pairs = pq.read_table(OUT / 'all_pairs.parquet').to_pylist()
    controllers = load(OUT / 'controller_profiles.json')
    provenance = []
    for r in pairs:
        version = native.get((r['state_uid'], r['eta_uid'], r['controller_uid']), 'OLD_FROZEN_SEED_VERIFIED_PROFILE')
        if r['partial_record'] and r['scenario'] == 'ring_exchange':
            assert version == 'ring_current_safety_v2'
        provenance.append({'state_uid': r['state_uid'], 'eta_uid': r['eta_uid'], 'controller_uid': r['controller_uid'],
                           'label_semantics_version': version, 'partial_record': r['partial_record'],
                           'n_observed': r['n'], 'n_seed_identities': len(r['seed_keys']),
                           'training_target_columns': 's,f; never legacy q or imputed Q16',
                           'controller_compatibility': controllers[r['controller_uid']]['compatibility_quality']})
        assert len(r['seed_keys']) == len(set(r['seed_keys'])) == r['n'] == r['s'] + r['f']
    assert len({(r['state_uid'], r['eta_uid'], r['controller_uid']) for r in pairs}) == len(pairs)
    csvout('seed_semantics_provenance.csv', provenance)
    dump('integrity_audit.json', {'canonical_pair_duplicates': 0, 'seed_duplicate_count': 0,
                                'target_label_changes': 0, 'target_state_overlap': 0,
                                'source_family_train_val_overlap': 0, 'normalization_unchanged': True,
                                'architecture_unchanged': True, 'generator_unchanged': True,
                                'old_result_hashes_unchanged': True,
                                'legacy_state_identity_quality': load(OUT / 'state_identity_audit.json'),
                                'scope': 'critic-only source-label zero-shot; historically established schema and target proposal generators are not claimed zero-shot'})

    ring_source = {r['method']: r for r in readcsv(OUT / 'source_failure_learning.csv')
                   if r['held_out_scene'] == 'ring' and r['source_scene'] == 'four_way_intersection'}
    decision = {
        'task': 'ORTHOFLOW3_LOSO_PARTIAL_COUNT_V1', 'status': 'COMPLETE',
        'negative_supervision_as_LOSO_cause': 'NOT_SUFFICIENT',
        'verdict': 'NEGATIVE_SUPERVISION_FIX_INSUFFICIENT',
        'source_supervision_bias_confirmed_and_repaired': True,
        'stable_state_aware_advantage_over_eta_only': False,
        'ring_old_B15': 0, 'ring_partial_count_B15': 0, 'ring_states': 60,
        'ring_partial_shared_seeds_17_23_41': [0, 0, 6],
        'ring_negative_binary_shared_seeds_17_23_41': [0, 0, 18],
        'ring_partial_eta_only_B15': 27,
        'ring_p_above_90_Q_at_most_half': {'old': 60, 'partial_count': 60},
        'ring_p_above_95_B15_precision': 0,
        'selected_models': {f: load(OUT / f / 'partial_count/models_frozen.json')['shared']['selected'] for f in FOLDS},
        'fold_metrics': [by[f, 'partial_count_shared'] for f in FOLDS],
        'remaining_bottleneck': 'SOURCE_PHYSICAL_CONTROLLER_REGIME_SUPPORT_AND_HARMFUL_STATE_CONDITIONED_EXTRAPOLATION',
        'causal_identification_limit': 'Support vs omitted controller-response conditioning vs learned extrapolation remain UNDERRESOLVED; this experiment does not prove invariance or representation impossibility.',
        'source_failure_learning': ring_source,
        'representation': 'KEEP', 'generator_changed': False, 'new_rollouts': 0,
        'no_further_test_guided_model_search': True,
        'source_side_additional_experiment': 'NOT_RUN: primary transfer gain not reliable; do not tune on frozen target panels',
    }
    dump('final_decision.json', decision)

    def score(f, m):
        r = by[f, m]
        return r['B15'] + (f'+{r["unresolved"]} unknown' if int(r['unresolved']) else '')

    lines = [
        'Success Basin C1: early-stop supervision repair and strict LOSO replication',
        '',
        'VERDICT: NOT_SUFFICIENT / NEGATIVE_SUPERVISION_FIX_INSUFFICIENT.',
        'Discarding early-stop negatives was a real source-supervision bias. Retaining them fixes source failure learning but does not establish cross-scene state-eta generalization. No primary fold shows a stable state-aware advantage over its matched source-only eta-only model.',
        '',
        'FROZEN K16 RESULTS (checkpoint and seed chosen only on source VAL)',
        'Target | N | old shared | same old pairs/count-weighted | partial shared | partial eta-only | negative-binary shared | oracle',
    ]
    for f in FOLDS:
        r = by[f, 'partial_count_shared']
        lines.append(' | '.join([f, r['states']] + [score(f, m) for m in
                     ('old_shared', 'full_continuation_shared', 'partial_count_shared', 'partial_count_eta_only', 'negative_binary_shared')] + [r['oracle_B15']]))
    lines += [
        '',
        'Four-Way partial shared has 21 confirmed B15 and 3 unresolved, not three confirmed failures. Its oracle gap is 0..3; eta-only has 24 confirmed B15. Numerical unresolved records remain interval-censored and were never turned into binary successes/failures.',
        'Available-B15 selection lower bounds: Toy75%, DB54.2%, Four87.5% (upper100%), Ring0%. All four candidate-set oracles cover every state. Matched target-supervised same-input references remain Toy45/48, DB24/24, Four22/24+2 unknown, Ring47/60. The stronger historical Ring-only classifier53/60 and enriched joint58/60 remain target-supervised references, not zero-shot successes.',
        '',
        'SEED STABILITY (17,23,41; no TEST selection)',
    ]
    for r in stability:
        if r['variant'] == 'partial_count':
            lines.append(f"{r['fold']} {r['model']}: {r['B15_counts']} B15; {r['unresolved_counts']} unknown; mean={r['B15_mean']:.2f}, sampleSD={r['B15_std_sample']:.2f}; source-selected seed={r['source_VAL_selected_seed']}")
    lines += [
        'Negative-binary Ring shared is 0/0/18 across seeds; selected18 equals its matched eta-only18 (9 rescues and 9 breaks). It is not stable shared-state transfer. Do not replace the preregistered primary model by a target-best diagnostic variant.',
        '',
        'PAIRED RESULTS / UNCERTAINTY',
        'Toy partial vs old:9 rescues/3 breaks, net+6, exact paired p=.146; vs matched eta-only:3/9, net-6. The modest old-baseline gain does not establish state benefit.',
        'DB partial vs old:2/9, net-7; vs same-weight full-data-only control:0/9, p=.00391. Adding source partial data hurts this held-out regime.',
        'Ring partial vs matched eta-only:0/27, p=1.49e-8. Four comparisons retain unknown status; no success/failure imputation. Exact paired tests and state-bootstrap intervals are descriptive, unadjusted across comparisons; no positive claim relies on them. Q16 is a finite-seed estimate, not the latent exact probability.',
        '',
        'DID THE SOURCE MODEL LEARN THE FAILURES?',
        f"Yes. In Ring's source Four-Way VAL (3330 certified nonB15 pairs), negative mean prediction {float(ring_source['old_full']['mean_p_on_nonB15']):.4f} -> {float(ring_source['partial_count']['mean_p_on_nonB15']):.4f}; count-NLL {float(ring_source['old_full']['count_likelihood_per_observed_trial']):.4f} -> {float(ring_source['partial_count']['count_likelihood_per_observed_trial']):.4f}; B15 AUROC {float(ring_source['old_full']['B15_AUROC']):.4f} -> {float(ring_source['partial_count']['B15_AUROC']):.4f}.",
        'This directly contradicts the explanation that the repaired likelihood simply failed to ingest or fit the newly retained source failures. Source-VAL improvement is not counted as transfer success.',
        '',
        'UPPER-TAIL PATHOLOGY',
        'Ring partial selected mean predicted p=.999664; compatible empirical Q16 interval averaged over states=.01875..0333333. All60 selections have p>.95 and are nonB15; B15 precision in that tail is0. Count with p>.9 and Q upper bound<=.5 remains60/60 (old60). At p>.95 old59 -> new60. No improvement.',
        'The failure is not uniform across folds: Toy partial is underconfident (selected p mean=.360, Q=.824); DB p=.164 versus Q=.776. Thus simply correcting global calibration cannot explain/fix the cross-scene rankings. Complete tails and per-seed predictions are retained.',
        '',
        'SUPERVISION REPAIR / RETENTION',
        '47486 unique canonical state-eta-controller pairs, including42840 added verified early-stop records (33090 stop at second failure;9750 stop at15 successes). No unrun continuation has been imputed.',
        'TRAIN: Toy48 states/2876pairs/29780 observed trials unchanged; DB69/1518/16627; Four64/16985/83842; Ring64/16865/92041. Detailed TRAIN/VAL positive/negative/count/eta statistics:dataset_audit.csv.',
        'TRAIN nonB15 retention:DB142/823 ->823/823; Four12/13356 ->13276/13356 (99.40%); Ring23/12997 ->12733/12997 (97.97%). Residual exclusions include numerical/unresolved stopping, not fabricated low-Q labels. Full-Q16 low-Q evidence remains scarce: repaired Four/Ring still have0 fully measured Q16<=.5 pairs in these training pools. Their short records supply observed failures and a nonB15 certificate, NOT exact Q16.',
        'OLD_FULL_ONLY means the frozen old completed-budget builder, not literally every pair having16 trials: old Toy includes planned Q4/Q8 and stronger Q64. Old pair membership/outcomes/normalization/results were retained. The extra full_continuation control isolates changing W1 to exposure weighting on the same original pair pool.',
        '',
        'LIKELIHOOD AND SPLITS',
        'Primary loss is -s log p-f log(1-p), using only observed successes/failures. Per source, uniform pairs are sampled and their total likelihood is divided by the fixed source TRAIN mean exposure; equal-size scene minibatches make scenes equally weighted in expectation. Within-scene contribution is continuation-equivalent (n=4 has one quarter the exposure of n=16). All retained historical trials remain; no trial cap or invented completion.',
        'Known first-stopping-time logic was validated against seed identities. Exhaustive stopped-path tests show zero expected likelihood gradient at the generating Bernoulli p, while naive equal-pair stopped empirical rates are biased. This assumes conditional exchangeable Bernoulli seed outcomes; it does not remove adaptive eta acquisition or physical-support bias.',
        'Binary diagnostic uses -log P(nonB15|p) with P(B15|p)=p^16+16p^15(1-p), not false Q=0. Each negative event gets one exposure; partial positives are excluded in that diagnostic. Its information content/weight differs from the main likelihood, so it is not a pure one-factor comparison of negative presence alone.',
        'Shared physical model and eta-only network unchanged; seeds17/23/41; source-only AdamW and early stopping. Each held-out scene is excluded from training, sampler statistics, normalization and model selection. No source TRAIN/VAL family overlap; no confirmation state UID overlap. Legacy state identity qualities are disclosed in state_identity_audit.json rather than relabelled CONTENT_EXACT.',
        'Strictness scope: critic-only source-label LOSO under the inherited frozen schema. The historical schema and target candidate-generators have prior scene exposure. These repeated frozen benchmarks are not new untouched confirmation sets, and this is not whole-pipeline zero-shot.',
        '',
        'ROOT-CAUSE UPDATE',
        'Confirmed: label ascertainment badly distorted source supervision, and this repair learns the source failure regions. Not established: it was the main cause of LOSO failure. Removing it does not rescue Ring and can worsen DB.',
        'Most specific remaining mechanism: state-conditioned extrapolation outside source physical/controller regimes. The earlier support audit found absent Ring curvature and disjoint physical neighborhoods; source-local fitting now succeeds while held-out Ring remains confidently wrong. These findings favor support/learned-extrapolation limits, but do not causally separate missing controller-response conditioning from harmful inductive extrapolation. Representation/invariance failure remains UNDERRESOLVED, not proven.',
        'KEEP the audited representation and generator unchanged. Do not launch target-guided architecture/ranking-loss sweeps. No additional source-side intervention is claimed here: the primary transfer improvement criterion was not met. A future matched source physical/controller-response support intervention would distinguish support from missing conditioning; it is not executed or counted as evidence.',
        '',
        'COST / REPRODUCTION',
        'NEW ROLLOUT=0. DB read-only.39936 target seed slots queried;37472 exact-complete-pair seed records and2273 reusable partial-pair seed records =39745 valid observations;191 unresolved/missing slots retained, not rerun. No journals/merger writes are required because no new outcomes were generated.',
        '72 new small-model runs (3 variants x2models x3seeds x4folds); frozen old baseline reused. Jobs1728 training,1731 frozen evaluation,1750 source audit each used1GPU shard/2CPU. User authorized up to6 shards, not a utilization requirement. Audit1748 failed only for unavailable sklearn; scipy rank-based AUC replacement did not change models/labels.',
        'Reproduction: cwd=/home/zhihan/research/Basin_C1; Python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python; PYTHONPATH=cwd. data.py prepare is freeze-guarded; train.py --fold {toy,db,four,ring} --variant {full_continuation,partial_count,negative_binary}; source training is blocked after frozen target prediction exists. evaluate.py evaluate reuses frozen predictions and DB evidence; source_audit.py computes the matched source diagnostic; finalize.py writes this report. Slurm wrappers are included.',
        'Key artifacts: protocol.json, all_pairs.parquet, partial_count_semantics.csv, seed_semantics_provenance.csv, fold dataset_manifest/normalization/models_frozen files, checkpoint_inventory.csv, primary_summary.csv, seed_stability.csv, paired_comparisons.csv, source_failure_learning.csv, upper_tail_calibration.csv, target_supervised_references.csv, budget_audit.json, integrity_audit.json, artifact_hashes.json.',
    ]
    (OUT / 'final_report.txt').write_text('\n'.join(lines) + '\n')
    csvout('experiment_ledger.csv', [
        {'stage': 'dataset_repair', 'status': 'COMPLETE', 'job': '', 'new_rollouts': 0, 'note': 'seed-verified observed counts; frozen protocol'},
        {'stage': 'source_training', 'status': 'COMPLETE', 'job': '1728', 'new_rollouts': 0, 'note': '72runs; 3seeds; source-only selection'},
        {'stage': 'frozen_target_evaluation', 'status': 'COMPLETE', 'job': '1731', 'new_rollouts': 0, 'note': 'same K16; unchanged outcomes'},
        {'stage': 'source_failure_audit', 'status': 'FAILED_THEN_RECOVERED', 'job': '1748;1750', 'new_rollouts': 0, 'note': 'missing dependency only; no model/label change'},
        {'stage': 'final_adjudication', 'status': 'COMPLETE', 'job': '', 'new_rollouts': 0, 'note': 'NOT_SUFFICIENT; no test-guided continuation'},
    ])
    dump('working_state.json', {'stage': 'complete', 'new_rollouts': 0,
                               'final_verdict': decision['verdict'], 'pending_jobs': [], 'generator_changed': False})
    hashed = [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.parquet', '.npz', '.msgpack', '.py', '.txt', '.sbatch')
              and p.name != 'artifact_hashes.json' and '__pycache__' not in str(p)]
    dump('artifact_hashes.json', {str(p.relative_to(OUT)): sha(p) for p in sorted(hashed)})
    print(json.dumps({'status': 'complete', 'verdict': decision['verdict'], 'new_rollouts': 0,
                      'checkpoints_verified': len(checkpoints), 'artifacts_hashed': len(hashed)}))


if __name__ == '__main__':
    main()

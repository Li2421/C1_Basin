from core import *
import csv
from datetime import datetime


def main():
    required=['PREREGISTRATION.receipt.json','HISTORY_AUDIT.receipt.json','RHR_SHORTLIST.receipt.json','CANDIDATES.receipt.json','CONSERVATIVE_TRAINING_CONFIGS.receipt.json','NEW_CONTROLLER_CONFIGS.receipt.json','TRAINING_FINAL_PREFLIGHT_BUDGET.receipt.json','CONTROLLER_FEATURES.receipt.json','FEATURE_TABLE.receipt.json','SELECTABLE_SPACE.receipt.json','DEVELOPMENT_RESULTS.receipt.json','LEAVE_ONE_FAMILY.receipt.json','SCALING_RESULTS.receipt.json','FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json','CONFIRMATORY_RESULTS.receipt.json','AXIS_TABLE.receipt.json','ZOO_MANIFEST.receipt.json','DESCRIPTIVE_ANALYSIS.receipt.json','FIGURES.receipt.json']
    verified=0
    for name in required:
        receipt=read(HERE/name)
        if name=='PREREGISTRATION.receipt.json':
            assert sha(HERE/'PREREGISTRATION.md')==receipt['sha256']
            assert sha(ROOT/'docs/prompt_crosstype2_mixed_zoo_selector_20261018.md')==receipt['prompt_sha256']
            verified+=2
        else:
            for p,h in receipt['files'].items():assert sha(p)==h,(name,p);verified+=1
    chosen=read(HERE/'NEW_EVAL_CHOICES.receipt.json')['utc'];frozen=read(HERE/'FROZEN_PIPELINE.receipt.json')['utc'];submission=read(HERE/'new_evaluation_run_SUBMISSION.json')['utc']
    assert datetime.fromisoformat(frozen)<datetime.fromisoformat(chosen)<datetime.fromisoformat(submission)
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:
        labels=z['labels'];assert labels.shape==(60,24,17,2) and not np.any(labels==-2)
        names=z['names'];types=z['types'];assert len(set(names))==60 and len(set(types))==16
    with (HERE/'CONTROLLER_STATE_DETAILS.csv').open() as f:detail_rows=sum(1 for _ in csv.DictReader(f))
    assert detail_rows==60*24*16
    with (HERE/'CONTROLLER_STATE_CROSSFIT_REGRET.csv').open() as stream:crossfit_rows=list(csv.DictReader(stream))
    assert len(crossfit_rows)==60*24
    confirmation=read(HERE/'CONFIRMATORY_RESULTS.json')['results']
    for kind in set(r['type'] for r in crossfit_rows):
        rows=[r for r in crossfit_rows if r['type']==kind]
        expected=confirmation[kind]['type_equal']['methods']['pipeline']['state_crossfit_regret_pp']
        assert np.isclose(np.mean([float(r['crossfit_regret_pp']) for r in rows]),expected,atol=1e-10,rtol=0)
    assert len(read(HERE/'FROZEN_PIPELINE.json')['outer'])==14
    development=read(HERE/'DEVELOPMENT_RESULTS.json')['reports']
    full_source_rows=[r for r in read(HERE/'SCALING_RESULTS.json')['rows'] if r['m']==13]
    assert len(full_source_rows)==14
    for row in full_source_rows:
        reference=development[row['held_type']]['type_equal']['methods']['pipeline']['success_lower_pct']
        assert np.isclose(row['success_pct'],reference,atol=1e-10,rtol=0),('m13_outer_fold_inconsistency',row['held_type'])
    assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'] and read(HERE/'FROZEN_COMPONENT_REPRODUCTION.json')['passed']
    assert read(HERE/'CORRECTED_ASTAR_TESTS.json')['passed'] and read(HERE/'MIXED_CONTROLLER_TESTS.json')['passed']
    assert read(HERE/'TRAINING_ENV_GATE.json')['passed'] and read(HERE/'CONSERVATIVE_REWARD_TESTS.json')['passed']
    assert read(HERE/'FROZEN_INFERENCE_TESTS.json')['passed'] and read(HERE/'STATISTICS_TESTS.json')['passed']
    budget=read(HERE/'TRAINING_FINAL_PREFLIGHT_BUDGET.json');assert budget['planned_missing']==25760
    for phase in ('training_legacy','training_mixed','training_ppo','new_evaluation','locked_few_labels'):
        assert not read(HERE/f'{phase}_POSTFLIGHT_UNKNOWN.json')['absent_rows']
    check=dict(utc=utc(),passed=True,verified_receipt_file_links=verified,new_state_label_shape=list(labels.shape),controller_state_candidate_detail_rows=detail_rows,outer_type_models=14,new_evaluation_types=16,choice_before_submission=True,full_menu_two_seed_coverage=True,full_source_scaling_matches_all14_original_outer_folds=True,optional_implicit_gap_preserved=True,unknown_reruns=0)
    write(HERE/'DELIVERY_CHECKS.json',check)
    seal('DELIVERY_CHECKS.receipt.json',[HERE/'DELIVERY_CHECKS.json',HERE/'delivery_checks.py'])
    print('Delivery checks passed:',json.dumps(check))


if __name__=='__main__':main()

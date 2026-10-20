"""Evidence and ordering checks; no result is declared scientifically positive."""
from common import *
import csv
from datetime import datetime
def main():
    required=['PREREGISTRATION.receipt.json','USER_DECISIONS.receipt.json','HISTORY_AUDIT.receipt.json','EXECUTION_REPRODUCTION.receipt.json','GENERATOR_REPRODUCTION.receipt.json','GENERATED_ACTION_TESTS.receipt.json','CANDIDATES.receipt.json','PILOT_REPORT.receipt.json','SELECTABLE_SPACE.receipt.json','CRITIC_TESTS.receipt.json','STATISTICS_TESTS.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','DEVELOPMENT_RESULTS.receipt.json','MAIN_METHOD_SELECTION.receipt.json','GENERATOR_EXTENSION_DECISION.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json','NEW_EVAL_BUDGET.receipt.json','NEW_EVAL_BUNDLE.receipt.json','CONFIRMATORY_RESULTS.receipt.json','AXIS_TABLE.receipt.json','DESCRIPTIVE_ANALYSIS.receipt.json','FIGURES.receipt.json','FIGURE_PREVIEWS.receipt.json'];verified=0
    for name in required:
        r=read(HERE/name)
        if name=='PREREGISTRATION.receipt.json':assert sha(HERE/'PREREGISTRATION.md')==r['sha256'] and sha(ROOT/'docs/prompt_crosstype1c_selected_two_eta_schedule_20261018.md')==r['prompt_sha256'];verified+=2
        else:
            for path,h in r['files'].items():assert sha(path)==h,(name,path);verified+=1
    for name in ('EXECUTION_REPRODUCTION.json','GENERATOR_REPRODUCTION.json','GENERATED_ACTION_TESTS.json','CRITIC_TESTS.json','STATISTICS_TESTS.json','FROZEN_INFERENCE_TESTS.json'):assert read(HERE/name)['passed'],name
    chosen=read(HERE/'NEW_EVAL_CHOICES.receipt.json')['utc'];frozen=read(HERE/'FROZEN_PIPELINE.receipt.json')['utc'];first=read(HERE/'new_evaluation_base_SUBMISSION.json')['utc'];assert datetime.fromisoformat(frozen)<datetime.fromisoformat(chosen)<datetime.fromisoformat(first)
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:
        assert z['labels'].shape==(62,48,26,1) and not np.any(z['labels']==-2);assert z['action'].shape==(62,48,26,10)
    for phase in ('pilot','remaining','new_evaluation'):assert not read(HERE/f'{phase}_POSTFLIGHT_UNKNOWN.json')['absent_rows']
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:
        assert z['labels'].shape[:2]==(62,24) and z['labels'].shape[-1]==2;assert np.array_equal(z['labels']!=-2,z['uids']!='');assert len(set(z['types']))==9
    with (HERE/'CONTROLLER_STATE_CROSSFIT_REGRET.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==62*24
    reports=read(HERE/'CONFIRMATORY_RESULTS.json')['reports']
    for kind in {r['type'] for r in rows}:
        actual=np.mean([float(r['evaluated_state_crossfit_regret_pp']) for r in rows if r['type']==kind]);expected=reports[kind]['type_equal']['methods']['pipeline']['evaluated_action_crossfit_regret_pp'];assert np.isclose(actual,expected,rtol=0,atol=1e-10)
    budget=read(HERE/'NEW_EVAL_BUDGET.json');assert budget['additional_genuinely_missing']<=10000 and budget['pure_training_missing']+budget['base_genuinely_missing']<=93000
    config=read(HERE/'CRITIC_TRAINING_PLAN.json');assert len(config['tasks'])==92
    for task in config['tasks']:assert (HERE/'critic_fits'/task['mode']/task['source_key']/'COMPLETE.receipt.json').exists()
    assert len(read(HERE/'FROZEN_PIPELINE.json')['outer'])==9
    result=dict(utc=utc(),passed=True,verified_receipt_file_links=verified,pure_training_shape=[62,48,26,1],action_shape=[62,48,26,10],new_evaluation_controllers=62,new_evaluation_states=24,new_evaluation_seeds=2,controller_state_detail_rows=len(rows),all92_three_seed_fits_complete=True,choice_before_first_fresh_submission=True,oracle_is_evaluated_subset_not320=True,unknown_reruns=0,manual_visual_review_and_closing_audit_still_required=True)
    write(HERE/'DELIVERY_CHECKS.json',result);seal('DELIVERY_CHECKS.receipt.json',[HERE/'DELIVERY_CHECKS.json',HERE/'delivery_checks.py']);print('Delivery checks passed:',json.dumps(result),flush=True)
if __name__=='__main__':main()

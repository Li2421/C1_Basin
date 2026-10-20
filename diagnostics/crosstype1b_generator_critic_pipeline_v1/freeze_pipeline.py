"""Seal the best source-nested development version before any new rollout."""
from common import *


def main():
    assert not (HERE/'FROZEN_PIPELINE.json').exists()
    assert (HERE/'DEVELOPMENT_REVIEW.json').exists(),'Inspect development evidence before freezing'
    review=read(HERE/'DEVELOPMENT_REVIEW.json');assert review['freeze_current_version'] is True
    receipts=['DEVELOPMENT_RESULTS_V2B.receipt.json','DEVELOPMENT_CHOICES_V2B.receipt.json','CANDIDATES.receipt.json',
        'CRITIC_TRAINING_PLAN.receipt.json','NESTED_SELECTION_TESTS.receipt.json','STATISTICS_TESTS.receipt.json',
        'CRITIC_MODEL_TESTS.receipt.json','CRITIC_TRAINING_TESTS.receipt.json','USER_TRAINING_ORACLE_DECISION.receipt.json',
        'USER_CPU_CAPACITY_DECISION.receipt.json','SHRINKAGE_SELECTION_TESTS.receipt.json','ITERATION_V2_DESIGN.receipt.json',
        'ACTION_ONLY_TESTS.receipt.json','MATCHED_SELECTION_TESTS.receipt.json','ITERATION_V2_MATCHED_ABLATIONS_DESIGN.receipt.json',
        'FROZEN_INFERENCE_TESTS.receipt.json','PREPROCESSING_DTYPE_DIAGNOSIS.receipt.json']
    for receipt in receipts:
        for path,digest in read(HERE/receipt)['files'].items():assert sha(path)==digest,path
    assert not (HERE/'new_evaluation_run_SUBMISSION.json').exists()
    selected=read(HERE/'SELECTED_MODELS_V2B.json');paths=[]
    records=[m for group in selected['outer'].values() for m in group.values()]
    records.extend(m['model'] for m in selected['all_types'].values())
    assert len(selected['outer'])==9
    for held,group in selected['outer'].items():
        assert all(held not in m['source_types'] and len(m['source_types'])==8 for m in group.values())
    def leaves(model):
        return leaves(model['full'])+leaves(model['no_controller']) if model.get('kind')=='probability_shrinkage' else [model]
    for model in [leaf for record in records for leaf in leaves(record)]:
        for kind in ('checkpoint','transform','fit_receipt'):
            p=Path(model[kind+'_path']);assert sha(p)==model[kind+'_sha256'];paths.append(p)
    files=['critic_model.py','train_critic.py','select_models.py','new_evaluation_choices.py',
        'generated_execution.py','design_execution.py','statistics_pipeline.py','development_statistics.py',
        'freeze_pipeline.py','DEVELOPMENT_REVIEW.json','SELECTED_MODELS_V2B.json','PREREGISTRATION.md',
        'select_models_v2.py','ITERATION_V2_DESIGN.json','development_statistics_v2.py',
        'select_matched_v2b.py','action_only_model.py','train_action_only.py','development_statistics_v2b.py',
        'PREREGISTRATION_CLARIFICATIONS.json','CRITIC_PRECISION_DESIGN.json']
    record=dict(utc=utc(),experiment=EXP,version='v2b_source_nested_positive_controller_shrinkage_matched_ablations',outer=selected['outer'],
        all_types=selected['all_types'],seeds=[17,23,41],x64=False,critic_matmul_precision='highest',
        inputs=dict(state='h100 and c33 from fixed reference Flow; state-only normalization fitted on 48 training states',
            controller='24 outcome-free mechanism features; source-type-only imputation/normalization',
            action='early eta3, late eta3, fixed/progress/step500 onehot3, safety flag1',
            normalization_parameter_dtypes='h/c mean and scale float32; controller fill,mean,scale float64; normalized network inputs float32'),
        candidates=dict(path=str(HERE/'CANDIDATES.json'),sha256=sha(HERE/'CANDIDATES.json'),count=13,
            generator='Immutable seed41 step1900, transformed mean plus four deterministic SHA samples',
            source_critic_warm_start='Immutable seed23 step2700, trained historically on original native Flow; source nested fine tuning does not erase this pretraining scope'),
        selection=dict(primary='argmax three-seed mean, ties prefer progress schedule',
            conservative='argmax(mean-standard deviation), fallback if best score is below progress mean',
            new_evaluation='Each controller uses its held-type model; all-nine-type model is descriptive reference'),
        development_review=review,training_oracle_budget_decision=read(HERE/'USER_TRAINING_ORACLE_DECISION.json'),new_evaluation_outcomes_observed=False,
        code_sha256={name:sha(HERE/name) for name in files})
    target=HERE/'FROZEN_PIPELINE.json';write(target,record)
    seal('FROZEN_PIPELINE.receipt.json',[target,*[HERE/name for name in files],*[HERE/name for name in receipts],*paths],
        before_new_evaluation_choices=True,before_new_evaluation_rollout_submission=True)
    print('Nine held-type models and all-type reference frozen',flush=True)


if __name__=='__main__':main()

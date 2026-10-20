"""Seal models, inputs, primary fallback decision and statistical code."""
from common import *
def main():
    assert read(HERE/'FROZEN_INFERENCE_TESTS.json')['passed'] and read(HERE/'STATISTICS_TESTS.json')['passed'];assert not (HERE/'FROZEN_PIPELINE.receipt.json').exists()
    selected=read(HERE/'SELECTED_MODELS.json');decision=read(HERE/'MAIN_METHOD_SELECTION.json');states=read(HERE/'NEW_EVAL_STATES.json')
    with rollout_db.connect(True) as db:
        assert all(db.execute('SELECT 1 FROM rollout WHERE state_uid=? LIMIT 1',(s['state_uid'],)).fetchone() is None for s in states),'No fresh closed loop before freeze'
    assert not (HERE/'new_evaluation').exists()
    frozen=dict(utc=utc(),outer=selected['outer'],all_types=selected['all_types'],primary_method=decision['primary_method'],primary_decision=decision,initializer='scratch_no_human_eta_pretraining',state_dimension=100,environment_dimension=33,controller_dimension=24,action_definition='early eta3,late eta3,late-minus-early3,switch/H',action_dimension=10,generator_proposals=8,full_indices=320,unique_actions=288,times=[200,300,400,500,600],eta13_eta10_never_candidates=True,probability_rule='Three paired seeds17/23/41; source-only positive full-branch shrinkage; ensemble mean argmax',tie_rule='Lowest canonical candidate ID',conservative='Mean minus one ensemble standard deviation in same primary candidate pool',normalization_dtypes=dict(h='float32',c='float32',controller='float64'),x64=False,matmul_precision='highest',optional_schedule_generator_enabled=False)
    write(HERE/'FROZEN_PIPELINE.json',frozen)
    paths=[HERE/n for n in ('FROZEN_PIPELINE.json','CANDIDATES.receipt.json','MAIN_METHOD_SELECTION.receipt.json','DEVELOPMENT_RESULTS.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','PREREGISTRATION.receipt.json','IMPLEMENTATION_CLARIFICATIONS.receipt.json','GENERATOR_EXTENSION_DECISION.receipt.json','TRAINING_BUNDLE.npz','CONTROLLER_FEATURES.npz','STATE_INPUTS.npz','CANDIDATES.json')]
    paths.extend(HERE/n for n in ('GENERATOR_REPRODUCTION.receipt.json','EXECUTION_REPRODUCTION.receipt.json','GENERATED_ACTION_TESTS.receipt.json','USER_DECISIONS.receipt.json','HISTORY_AUDIT.receipt.json','STATISTICS_TESTS.receipt.json','legacy_conditioning.py','frozen_networks.py'))
    paths.extend(HERE/n for n in ('RANKING_NUMERIC_DESIGN.receipt.json','right_hand.py','trigger.py','UNIT_REPLAY_DOCUMENTATION_V2.receipt.json'))
    paths.extend(HERE/n for n in ('critic_model.py','legacy_critic_model.py','train_critic.py','legacy_train_critic.py','select_models.py','inference.py','candidates.py','new_choices.py','new_design.py','statistics_pipeline.py','confirmatory_statistics.py','assemble_evaluation.py','development_statistics.py','freeze_pipeline.py','core.py','common.py','generated_execution.py','rollout_runtime.py','zoo_runtime.py','zoo_models.py','controllers.py','historical_controllers.py','legacy_controllers.py','learned.py','gaussian.py','ippo.py','progress.py','worker.py'))
    def collect(model):
        if model.get('kind')=='probability_shrinkage':collect(model['full']);collect(model['no_controller'])
        else:
            for key in ('checkpoint','transform','fit_receipt'):
                p=Path(model[key+'_path']);assert sha(p)==model[key+'_sha256'];paths.append(p)
    for group in selected['outer'].values():
        for m in group.values():collect(m)
    collect(selected['all_types']['full']);collect(selected['all_types']['no_controller'])
    seal('FROZEN_PIPELINE.receipt.json',sorted(set(paths)),no_fresh_closed_loop_before_freeze=True)
    print('Frozen primary',frozen['primary_method'],'and all9 outer/all-type model weights',flush=True)
if __name__=='__main__':main()

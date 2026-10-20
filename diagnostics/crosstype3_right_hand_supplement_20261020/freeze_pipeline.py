"""Seal every model and rule before any fresh-state closed loop."""
from common import *
def main():
    for name in ('FROZEN_INFERENCE_TESTS','STATISTICS_TESTS','CRITIC_TESTS'):assert read(HERE/f'{name}.json')['passed']
    assert (HERE/'CONTROLLER_FEATURES.receipt.json').exists() and (HERE/'LEAVE_ONE_FAMILY_RESULTS.receipt.json').exists()
    assert not (HERE/'FROZEN_PIPELINE.receipt.json').exists();selected=read(HERE/'SELECTED_MODELS.json');decision=read(HERE/'MAIN_METHOD_SELECTION.json');states=read(HERE/'NEW_EVAL_STATES.json')
    with rollout_db.connect(True) as db:assert all(db.execute('SELECT 1 FROM rollout WHERE state_uid=? LIMIT 1',(s['state_uid'],)).fetchone() is None for s in states)
    assert not (HERE/'new_evaluation').exists()
    frozen=dict(utc=utc(),outer=selected['outer'],all_types=selected['all_types'],primary_method=decision['primary_method'],primary_decision=decision,initializer='scratch_no_human_eta_pretraining',state_dimension=100,environment_dimension=33,controller_dimension=24,action_definition='early eta3,late eta3,late-minus-early3,switch/H',action_dimension=10,generator_proposals=8,full_indices=320,unique_actions=288,times=[200,300,400,500,600],eta13_eta10_never_candidates=True,continuing_right_hand_rules_excluded=True,probability_rule='Three paired seeds17/23/41; source-only positive full-branch shrinkage; ensemble mean argmax',tie_rule='Lowest canonical candidate ID',conservative='Mean minus one ensemble standard deviation in same primary candidate pool',normalization_dtypes=dict(h='float32',c='float32',controller='float64'),x64=False,matmul_precision='highest',training_types=16,confirmed_unseen_types=['transformer_bc','gradient_mpc','conflict_based_search'],pre_freeze_exceptions=str(HERE/'USER_DESIGN_DECISIONS.receipt.json'))
    assert len(frozen['outer'])==16;write(HERE/'FROZEN_PIPELINE.json',frozen)
    paths=[HERE/'FROZEN_PIPELINE.json']+list(HERE.glob('*.py'))+[HERE/n for n in ('CANDIDATES.receipt.json','MAIN_METHOD_SELECTION.receipt.json','DEVELOPMENT_RESULTS.receipt.json','LEAVE_ONE_FAMILY_RESULTS.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','PREREGISTRATION.receipt.json','USER_DESIGN_DECISIONS.receipt.json','TRAINING_BUNDLE.npz','CONTROLLER_FEATURES.npz','STATE_INPUTS.npz','CANDIDATES.json','GENERATOR_REPRODUCTION.receipt.json','EXECUTION_REPRODUCTION.receipt.json','HISTORY_AUDIT.receipt.json','STATISTICS_TESTS.receipt.json','CRITIC_TESTS.receipt.json','UNSEEN_CAPABILITY.receipt.json','astar_kernel.so','cbs_kernel.so')]
    def collect(model):
        if model.get('kind')=='probability_shrinkage':collect(model['full']);collect(model['no_controller'])
        else:
            for key in ('checkpoint','transform','fit_receipt'):
                p=Path(model[key+'_path']);assert sha(p)==model[key+'_sha256'];paths.append(p)
    for group in selected['outer'].values():
        for model in group.values():collect(model)
    collect(selected['all_types']['full']);collect(selected['all_types']['no_controller'])
    seal('FROZEN_PIPELINE.receipt.json',sorted(set(paths)),no_fresh_closed_loop_before_freeze=True,blinded_pre_freeze_capability_and_mechanism_exception_user_approved=True)
    print('Frozen16 outer models and final16-type model; primary',frozen['primary_method'],flush=True)
if __name__=='__main__':main()

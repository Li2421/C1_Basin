"""Seal all 19 leave-type-out models and final model before fresh choices."""
from common import *

def main():
    assert read(HERE/'CRITIC_TESTS.json')['passed'] and read(HERE/'FROZEN_INFERENCE_TESTS.json')['passed'];assert (HERE/'DEVELOPMENT_RESULTS.receipt.json').exists();assert (HERE/'NEW_EVAL_INPUTS.receipt.json').exists();selected=read(HERE/'SELECTED_MODELS.json');assert len(selected['outer'])==19;states=read(HERE/'NEW_EVAL_STATES.json')
    with rollout_db.connect(True) as db:
        assert all(not db.execute('SELECT 1 FROM rollout WHERE state_uid=? LIMIT 1',(s['state_uid'],)).fetchone() for s in states)
    frozen=dict(utc=utc(),evidence='DEVELOPMENT',outer=selected['outer'],all_types=selected['all_types'],primary_method='学习双 η 含不修正·可选按进度',primary_role='本轮主方法 = 学习双 η 含不修正·可选按进度',training_types=19,training_controllers=122,state_dimension=100,environment_dimension=33,controller_dimension=24,action_dimension=13,generator_proposals=8,zero_index=8,full_indices=486,unique_actions=441,times=[200,300,400,500,600,'progress'],tie_rule='Highest three-seed ensemble mean then lowest canonical candidate ID',normalization_dtypes=dict(h='float32',c='float32',controller='float64'),x64=False,matmul_precision='highest',no_unseen_confirmatory_claim=True)
    write(HERE/'FROZEN_PIPELINE.json',frozen);paths=[HERE/'FROZEN_PIPELINE.json',*HERE.glob('*.py'),*[HERE/n for n in ('CANDIDATES.receipt.json','DEVELOPMENT_RESULTS.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','PREREGISTRATION.receipt.json','TRAINING_BUNDLE.npz','CONTROLLER_FEATURES.npz','STATE_INPUTS.npz','NEW_EVAL_INPUTS.receipt.json','NEW_EVAL_INPUTS.npz','NEW_CANDIDATES.json','EXECUTION_REPRODUCTION.receipt.json','EXECUTION_EQUIVALENCE.receipt.json','HISTORY_AUDIT.receipt.json','CRITIC_TESTS.receipt.json')]]
    def collect(model):
        if model.get('kind')=='probability_shrinkage':collect(model['full']);collect(model['no_controller'])
        else:
            for key in ('checkpoint','transform','fit_receipt'):
                p=Path(model[key+'_path']);assert sha(p)==model[key+'_sha256'];paths.append(p)
    for held,group in selected['outer'].items():
        for model in group.values():assert held not in model['source_types'];collect(model)
    collect(selected['all_types']['full']);collect(selected['all_types']['no_controller'])
    seal('FROZEN_PIPELINE.receipt.json',sorted(set(paths)),no_fresh_closed_loop_before_freeze=True,all19_types_development=True)
if __name__=='__main__':main()

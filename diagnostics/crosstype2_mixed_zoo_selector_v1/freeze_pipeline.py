from common import *


def leaves(model):
    return leaves(model['full'])+leaves(model['no_controller']) if model.get('kind')=='probability_shrinkage' else [model]


def main():
    assert not (HERE/'FROZEN_PIPELINE.json').exists()
    receipts=['DEVELOPMENT_CHOICES.receipt.json','DEVELOPMENT_RESULTS.receipt.json','LEAVE_ONE_FAMILY.receipt.json','SCALING_RESULTS.receipt.json','CANDIDATES.receipt.json','TRAINING_BUNDLE.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','ARCHITECTURE_FIXED.receipt.json','USER_DESIGN_DECISIONS.receipt.json','RESOURCE_AUTHORIZATION_24_CPU.receipt.json','RESOURCE_AUTHORIZATION_8_GPU.receipt.json','ENDPOINT4_SCOPE_DESIGN.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json']
    for r in receipts:
        for path,digest in read(HERE/r)['files'].items():assert sha(path)==digest,path
    selected=read(HERE/'SELECTED_MODELS.json');types=sorted(selected['outer']);assert len(types)==14
    paths=[]
    groups=[*selected['outer'].values(),{k:v['model'] for k,v in selected['all_types'].items()},{k:v['model'] for k,v in selected['all_initializations'].items()}]
    for held,group in selected['outer'].items():assert all(held not in m['source_types'] for m in group.values())
    for group in groups:
        for model in group.values():
            for leaf in leaves(model):
                for kind in ('checkpoint','transform','fit_receipt'):
                    p=Path(leaf[kind+'_path']);assert sha(p)==leaf[kind+'_sha256'];paths.append(p)
    code=['critic_model.py','train_critic.py','select_models.py','select_mixed.py','single_source_selection.py','inference.py','legacy_inference.py','legacy_critic_model.py','legacy_train_critic.py','new_choices.py','evaluation_design.py','zoo_design.py','mixed_execution.py','mixed_runtime.py','mixed_controllers.py','generated_execution.py','statistics_pipeline.py','baselines.py','assemble_evaluation.py','confirmatory_statistics.py','freeze_pipeline.py']
    from zoo_design import DEPENDENCIES,profiles
    code=sorted(set(code)|set(DEPENDENCIES))
    write(HERE/'FROZEN_ZOO_BASE_PROTOCOL.json',profiles())
    inputs=['STATE_INPUTS.npz','CONTROLLER_FEATURES.npz','CANDIDATES.json','TRAINING_STATES.json','NEW_EVAL_STATES.json','BASE_PROTOCOL.json','TRAINING_BUNDLE.npz','RHR_SHORTLIST.json','NEW_CONTROLLER_CONFIGS.json','IPPO_TRAINING_CONFIGS.json','FROZEN_ZOO_BASE_PROTOCOL.json','DEVELOPMENT_CAPABILITY.json','PPO_CAPABILITY.json','LOCKED_CAPABILITY.json']
    record=dict(utc=utc(),experiment=EXP,version='mixed_zoo_source_nested_probability_shrinkage',outer=selected['outer'],all_types=selected['all_types'],all_initializations=selected['all_initializations'],source_types=types,inputs=dict(state_dimensions=100,context_dimensions=33,controller_dimensions=34,action_dimensions=16,normalization='Source-only; restore h/c mean/scale float32, C fill/mean/scale float64; network float32 highest'),seeds=[17,23,41],candidates=dict(path=str(HERE/'CANDIDATES.json'),sha256=sha(HERE/'CANDIDATES.json'),count=16,nominal_excluded=True),selection=dict(primary='three-seed mean argmax, ties progress then menu order',conservative='mean minus standard deviation; fallback to progress if below its predicted mean',development='held-type outer model',locked='all-development model'),development_review=read(HERE/'DEVELOPMENT_REVIEW.json'),code_sha256={n:sha(HERE/n) for n in code},new_evaluation_outcomes_read=False,pretraining_scope='Frozen generator/original critic historically trained on native Flow; warm starts do not erase that history')
    write(HERE/'FROZEN_PIPELINE.json',record)
    seal('FROZEN_PIPELINE.receipt.json',[HERE/'FROZEN_PIPELINE.json',HERE/'SELECTED_MODELS.json',HERE/'PREREGISTRATION.md',*[HERE/r for r in receipts],*[HERE/n for n in code],*[HERE/n for n in inputs],*set(paths)],before_new_evaluation_submission=True)
    print('Fourteen outer models and all-development models frozen; new-evaluation labels still unseen')


if __name__=='__main__':main()

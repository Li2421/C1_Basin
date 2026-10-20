"""Only a completed logged development iteration may become the frozen pipeline."""
from common import *
from development import load,TYPES
from selector import Regression,tune,ARMS
from features import NAMES
from selector_candidates_v2 import fit_selected


def main(bundle,iteration):
    prereg();assert not (HERE/'FROZEN_PIPELINE.json').exists()
    for path,digest in read(HERE/'DEVELOPMENT_FINAL_SELECTION.receipt.json')['files'].items():assert sha(path)==digest
    assert read(HERE/'DEVELOPMENT_FINAL_SELECTION.json')['selected']==iteration
    for path,digest in read(HERE/'iterations'/iteration/'ITERATION.receipt.json')['files'].items():assert sha(path)==digest
    result=read(HERE/'iterations'/iteration/'RESULTS.json')
    assert result.get('feature_schema','mechanism_v1')=='mechanism_v1'
    assert sha(bundle)==result['input_sha256']
    data=load(bundle);types=data['types'];rates=(data['labels']==1).mean((1,3))
    strategy=result.get('strategy','ridge')
    model,inner=fit_selected(data['features'],rates,types,strategy)
    comparator,pca_inner=fit_selected(data['r5'],rates,types,strategy,pca_dimensions=2)
    write(HERE/'FINAL_INNER_SELECTION.json',dict(mechanism=inner,r5_pca2=pca_inner))
    record=dict(schema='crosstype1_learned_zoo_selector_v1',feature_schema='mechanism_v1',utc=utc(),iteration=iteration,strategy=strategy,
        feature_names=list(NAMES),feature_definition_path=str(HERE/'features.py'),feature_definition_sha256=sha(HERE/'features.py'),
        arms=list(ARMS),model=model.to_dict(),r5_comparator=comparator.to_dict(),
        development_types=sorted(set(map(str,types))),
        missing_development_types=sorted(set(TYPES)-set(types)),
        source_controllers=list(map(str,data['names'])),development_targets=result['development_targets'],
        source_bundle_path=str(bundle),source_bundle_sha256=sha(bundle),locked_outcomes_read=False,
        fallback='All mechanism features unavailable -> S500; tied highest score -> S500; candidate margin as recorded in model',
        training_objective='Type-equal source-controller prediction of per-arm lower success minus S500',
        hyperparameter_selection='Inner leave-one-source-type; minimum type-equal regret, deterministic tie rule recorded in FINAL_INNER_SELECTION',
        scale_curve_required_before_freeze=True)
    assert (HERE/'SCALE_CURVE_COMPLETE.receipt.json').exists() or (HERE/'SCALE_CURVE_GAP.receipt.json').exists(), 'Complete scale-curve stage or an explicit gap receipt first'
    scale_receipt=HERE/('SCALE_CURVE_COMPLETE.receipt.json' if (HERE/'SCALE_CURVE_COMPLETE.receipt.json').exists() else 'SCALE_CURVE_GAP.receipt.json')
    for path,digest in read(scale_receipt).get('files',{}).items():assert sha(path)==digest
    if (HERE/'SCALE_CURVE_RESULTS.json').exists():
        scale=read(HERE/'SCALE_CURVE_RESULTS.json')
        assert scale['source_bundle_sha256']==sha(bundle) and scale['pipeline_structure']==strategy and scale['iteration']==iteration
    write(HERE/'FROZEN_PIPELINE.json',record)
    paths=[HERE/name for name in ('FROZEN_PIPELINE.json','FINAL_INNER_SELECTION.json','features.py','selector.py','selector_candidates.py','selector_candidates_v2.py','r5.py',
           'initial_actions.py','zoo_runtime.py','zoo_models.py','ippo.py','rollout_runtime.py','probe_prefix.py','prefix_reuse.py',
           'choose_locked.py','seal_comparators.py','frozen_standard.py','frozen_expanded.py','analysis_statistics.py','confirmatory_analysis.py',
           'transition_summary.py','LOCKED_ENDPOINT_DEFINITIONS.json','TRAJECTORY_FIGURE_DESIGN.json','plot_locked_episodes.py',
           'DEVELOPMENT_FINAL_SELECTION.json','DEVELOPMENT_FINAL_SELECTION.receipt.json','SCALE_CURVE_COMPLETE.receipt.json',
           'DEVELOPMENT_SELECTION_VALUE.json','freeze_pipeline.py','assemble.py','advance_locked.py')]
    seal('FROZEN_PIPELINE.receipt.json',paths,locked_eval_or_tuning_labels_read=False,
         development_targets_met=result['development_targets']['passed'])
    print('Pipeline frozen; development targets met:',result['development_targets']['passed'])

if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])

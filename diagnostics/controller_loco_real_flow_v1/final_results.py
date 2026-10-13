"""Aggregate all executed stages while preserving frozen Stage A metrics."""
import json
import hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())

def main(classification=None):
    frozen=BASE/'STAGE_A_RESULTS_FROZEN.json'
    result=read(frozen)
    protocol=read(BASE/'protocol.json');extra=read(BASE/'stage_b/protocol.json')
    result['configuration']=dict(environment=protocol['environments']['ring_exchange'],
        original_controller_order=protocol['controller_order'],extra_source_controller_order=extra['controller_order'],
        execution_profiles={**protocol['controllers']['ring_exchange'],**extra['controllers']['ring_exchange']},
        eta_candidates=protocol['eta'],eta_indexing='zero-based 0..15; eta13 is index13',
        train_states=128,stage_a_test_states=64,stage_c_unique_evaluated_physical_states=32,
        train_eta_per_state=8,train_seeds=[0,1,2,3],test_discovery_seeds=[0,1,2,3],test_endpoint_seeds=[4,5,6,7],
        fit_seeds=[17,23,41],architecture='physical23 + padded controller descriptor; 343 -> 64 SiLU -> 64 SiLU -> 16 logits',
        optimizer='AdamW lr0.0008 wd0.0001 global-norm clip5',steps=12000,batch_size=128,
        checkpoint_rule='source validation NLL averaged across 3 fit seeds',jax_enable_x64=False,
        feature_definitions_file=str(BASE/'features.py'),normalization='source training only',
        main_bootstrap_draws=5000,bootstrap_group='shared physical state across controllers')
    stages={}
    for stage,folder in [('stage_a',BASE),('stage_b',BASE/'stage_b'),('stage_c',BASE/'stage_c')]:
        file=frozen if stage=='stage_a' else folder/'results.json'
        if file.exists():
            stages[stage]=dict(status='COMPLETED',results_file=str(file),postflight=read(folder/'POSTFLIGHT.json'),
                               metrics=read(file))
        else:stages[stage]=dict(status='NOT EXECUTED')
    result['stage_a_results_frozen_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest()
    result['conditional_stages']={k:v for k,v in stages.items() if k!='stage_a'}
    result['study_summary']=dict(status='COMPLETE',classification=classification,
         stages={k:{'status':v['status'],'results_file':v.get('results_file'),'postflight':v.get('postflight')} for k,v in stages.items()},
         total_new_rollouts=sum(v['postflight']['raw'] for v in stages.values() if v['status']=='COMPLETED'),
         total_unknown=sum(v['postflight']['unknown'] for v in stages.values() if v['status']=='COMPLETED'),
         compatible_p1_validation_label_slots_reused=21504,
         compatible_p1_valid_labels_reused=21232,compatible_p1_unknown_labels_preserved=272,
         historical_label_reuse=read(BASE/'HISTORICAL_LABEL_REUSE_COUNTS.json'),
         stage_a_fit_seed_count=105,stage_b_fit_seed_count=30,
         stage_a_primary=result['primary'],
         stage_b_primary=stages.get('stage_b',{}).get('metrics',{}).get('primary'),
         stage_c_confirmation=stages.get('stage_c',{}).get('metrics'),
         controller_prior_decomposition=read(BASE/'controller_prior/results.json'),
         source_regime_coverage=read(BASE/'SOURCE_REGIME_COVERAGE.json'),
         training_provenance_audit=str(BASE/'TRAINING_PROVENANCE_AUDIT.json'),
         opened_state_policy='Stage A64 and Stage C32 physical states are DEVELOPMENT for future model selection; unused C32 states have no labels.',
         reports={'report':str(BASE/'REPORT.md'),'handoff':str(BASE/'HANDOFF.md')},
         meaning='Top-level rows/primary retain Stage A for compatibility; study_summary and conditional_stages contain the complete study.')
    confirmation=stages.get('stage_c',{}).get('metrics',{})
    prior=read(BASE/'controller_prior/results.json')
    extra_state_gain=any(r['ci95_simultaneous'][0]>0 for r in prior.get('stage_c',{}).get('contrasts',[]))
    result['hypotheses']={
        'controller_changes_q_level':{'verdict':'SUPPORTED','scope':'descriptive, fixed Ring FF H700 K16 controller panel'},
        'controller_changes_optimal_eta':{'verdict':'SUPPORTED','evidence':'2/7 conservative necessity gates passed'},
        'controller_only_changes_q_level_with_invariant_eta_ranking':{'verdict':'REJECTED','scope':'fixed Ring panel, necessity gate and controller×eta interaction'},
        'preregistered_independent_confirmation_success':{'verdict':'SUPPORTED' if confirmation.get('confirmed_arms') else 'REJECTED','scope':'operational joint decision criterion, not a claim that population effect equals zero'},
        'static_representation_heldout_selection_transfer':{'verdict':'SUPPORTED' if confirmation.get('confirmed_arms') else 'INCONCLUSIVE','confirmed_arms':confirmation.get('confirmed_arms',[])},
        'source_controller_coverage_is_main_root_cause':{'verdict':'INCONCLUSIVE','reason':'Stage B-A paired R3/combined selector gains are not significant; R0 decreased and source sample budget/distribution changed'},
        'state_selector_improves_over_learned_controller_prior':{'verdict':'SUPPORTED' if extra_state_gain else 'INCONCLUSIVE','scope':'exploratory, frozen prior rule and independent Stage C state panel'},
        'continuous_success_basin_generalization':{'verdict':'INCONCLUSIVE','reason':'K16 discrete candidates cannot identify a continuous success basin'},
        'cross_geometry_or_execution_operator_transfer':{'verdict':'INCONCLUSIVE','reason':'geometry FF safety layer and horizon fixed'}}
    (BASE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result['study_summary']

if __name__=='__main__':main()

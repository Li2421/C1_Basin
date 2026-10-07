"""Record the adjudication without promoting any model or opening target labels."""
import csv
import hashlib
from itertools import combinations
import numpy as np
from scipy.special import expit
from .audit import OUT,SRC,read,write,load,evaluate


def main():
    d,x,rows,states,indices,tr,va,seeds,s,n,s4,n4=load()
    decode=read(OUT/'controller_decode.json')
    pc=np.asarray(decode['VAL_controller_posteriors'])
    prior=(s4[:,tr].sum(1)+.5)/(n4[:,tr].sum(1)+1.)
    hard=prior[pc.argmax(-1)]
    hard_metrics=evaluate(hard,s[:,va],n[:,va],va)
    write('hard_controller_decode_diagnostic.json',{
        'posthoc_diagnostic_only':True,'uses_frozen_classifier_argmax_no_threshold_tuning':True,
        'not_a_deployable_unseen_controller_model':True,
        'metrics':hard_metrics})
    ss=s[:,va];ff=n[:,va]-ss;lo=ss/16;hi=(16-ff)/16
    models={}
    for seed in (17,23,41):
        z=np.load(SRC/f'models/physical_context/seed{seed}/validation_predictions.npz')['correct'][:,indices[va]]
        models[str(seed)]=z
    strong=[]
    for a,b in combinations(range(len(va)),2):
        for c in range(3):
            for i,j in combinations(range(16),2):
                sa=1 if lo[c,a,i]-hi[c,a,j]>=.25 else -1 if hi[c,a,i]-lo[c,a,j]<=-.25 else 0
                sb=1 if lo[c,b,i]-hi[c,b,j]>=.25 else -1 if hi[c,b,i]-lo[c,b,j]<=-.25 else 0
                if sa*sb>=0:continue
                bi=(ss[c,a,i]>=15,ss[c,b,i]>=15)
                bj=(ss[c,a,j]>=15,ss[c,b,j]>=15)
                robust_reversal=(bi[0] and not bj[0] and bj[1] and not bi[1]) or (bj[0] and not bi[0] and bi[1] and not bj[1])
                strong.append({'controller':c,'states':[int(va[a]),int(va[b])],'eta_indices':[i,j],
                    'requires_different_action_for_B15_in_this_two_candidate_subset':bool(robust_reversal),
                    'neural_correct':{seed:bool(np.sign(z[c,a,i]-z[c,a,j])==sa and np.sign(z[c,b,i]-z[c,b,j])==sb) for seed,z in models.items()}})
    relevant=[r for r in strong if r['requires_different_action_for_B15_in_this_two_candidate_subset']]
    write('decision_relevant_reversals.json',{'all_strong_reversals':len(strong),
        'two_candidate_B15_reversals':len(relevant),
        'neural_correct_by_seed':{seed:sum(r['neural_correct'][seed] for r in relevant) for seed in models},
        'warning':'Overlapping cases across six families, not independent samples; two-candidate diagnostic does not change frozen K16 deployment pool.',
        'rows':strong})
    probes=read(OUT/'fixed_eta_predictability.json')
    micro=read(OUT/'micro_summary.json')
    decision={
        'scope':'Source-only cause adjudication, 24 TRAIN families and 6 already-inspected source VAL families; no unseen-controller or LOSO claim.',
        'claim_more_information_must_significantly_beat_eta_only':{
            'verdict':'FALSE_WITHOUT_ADDITIONAL_ASSUMPTIONS',
            'probability_estimation':'Under one fixed population and unrestricted Bayes-optimal predictors, extra information cannot increase optimal expected log loss; strict improvement requires conditional outcome information.',
            'decision':'With identical finite candidates, optimal expected utility cannot decrease when additional information may be ignored. Strict improvement requires changes in the optimal decision, not merely probability changes.',
            'statistical_significance':'Neither strict improvement nor statistical significance is guaranteed. These inequalities do not guarantee finite-data or source-to-target performance.'},
        'selection_headroom':read(OUT/'selection_headroom.json'),
        'ranked_findings':[
            {'finding':'CONTROLLER_ETA_PRIOR_DOMINATES_THIS_K16_PANEL',
             'evidence':'Best global eta covers 13/17 eligible cases; controller-specific fixed eta covers 16/17; source-trained controller/eta table also 16/17. At most one extra case requires state adaptation beyond this controller prior.',
             'confidence':'HIGH_FOR_THIS_SMALL_PANEL_ONLY'},
            {'finding':'STATE_RESIDUAL_NOT_LEARNED_OR_GENERALIZED_FROM_CURRENT_DATA',
             'evidence':{'prior_NLL':probes['priors']['NLL'],
                 'fixed_eta_kernel_NLL':{k:v['validation']['NLL'] for k,v in probes['models'].items()},
                 'fixed_eta_kernel_state_permutation_p':{k:v['state_block_permutation_null']['one_sided_empirical_p'] for k,v in probes['models'].items()},
                 'neural_VAL_NLL_best_to_final':[[r['best_VAL_NLL'],r['final_VAL_NLL']] for r in read(OUT/'trajectory_audit.json')],
                 'state_heterogeneity_columns_BY005':17},
             'interpretation':'Real state variation exists, but it is not reliably predicted on source held-out families. More training steps worsen validation. Small state support/noisy labels and modeling limitations remain confounded.',
             'confidence':'HIGH_FOR_FAILURE_OF_TESTED_MODELS; UNDERRESOLVED_FOR_UNIQUE_CAUSE'},
            {'finding':'CURRENT_CONTEXT_CONTAINS_CONTROLLER_INFORMATION_BUT_IS_NOT_PROVEN_SUFFICIENT_FOR_STATE_INTERACTION',
             'evidence':{'nominal_context_controller_decode_accuracy':decode['VAL_controller_classification_accuracy'],
                 'soft_decode_Q_NLL':decode['correct_context']['NLL'],
                 'soft_decode_B15':decode['correct_context']['selected_B15'],
                 'hard_decode_diagnostic_B15':hard_metrics['selected_B15'],
                 'H20_vs_task_horizon_seconds':[1,35],
                 'micro_initial_vs_late_progress_mps':[micro['median_initial_progress_mps'],micro['median_late_progress_mps']]},
             'against_strong_insufficiency_claim':'No exact or near-exact augmented-input aliases found in this matrix. Better descriptor content is a hypothesis, not established irreducible missing information.',
             'confidence':'INFORMATION_PRESENT; SUFFICIENCY_UNDERRESOLVED'}],
        'explanations_not_supported_by_tests':[
            'Context disconnected, wrong checkpoint, or wrong action path as explanation of this repaired run',
            'All state variation is merely Q4 sampling noise',
            'An uncommitted noisy Flow reference alone explains the failed state learning',
            'Context probe Monte Carlo noise destroys every useful physical feature',
            'Extending training steps alone fixes source validation',
            'All observed state ranking reversals imply a large K16 top-1 advantage must exist'],
        'remaining_uncertainty':'Cannot yet separate inadequate physical-state support, insufficient future-response summaries, and inability of tested models to learn the residual. Cannot conclude the augmented input is information-insufficient or that cross-scene generalization is impossible.',
        'minimum_next_discriminating_experiment':{
            'design':'Freeze a source-only blocked study with independent families. Compare current H20 against a context capturing the onset of progress loss/final projection response, on exactly the same labels and model. Cross with a broader TRAIN family set; keep a controller-eta prior control.',
            'interpretation':'Context-only gain favors missing response information; family-coverage gain favors support limitation; neither gain leaves learning/representation unresolved.',
            'evaluation':'Predefine interaction-relevant two-candidate subsets from a separate discovery split; confirm full K16 separately. Keep controller-family and scene holdouts untouched.',
            'not_executed':'Requires a separately frozen data/compute plan; this adjudication did not alter or promote the pipeline.'},
        'resource_accounting':{'new_full_continuations':0,'new_success_labels':0,'global_DB_writes':0,
            'derived_microprobe_trajectories':48,'derived_microprobe_steps':3840,'maximum_GPU_shards':3,
            'kernel_null_controls':495,'generator_modified':False,'target_labels_opened':False},
        'validation':{'seed_records_match_frozen_aggregate':True,'canonical_h_permutation_check':True,
            'source_family_overlap':0,'all_context_cache_means_match_frozen_matrix':True},
        'reproduce':['python -m diagnostics.orthoflow3_state_context_adjudication_v1.audit',
                     'python -m diagnostics.orthoflow3_state_context_adjudication_v1.context_audit',
                     'python -m diagnostics.orthoflow3_state_context_adjudication_v1.controller_decode',
                     'python -m diagnostics.orthoflow3_state_context_adjudication_v1.finalize']}
    write('final_decision.json',decision)
    write('working_state.json',{'phase':'bounded_cause_adjudication_complete',
        'larger_generalization_goal':'NOT_ACHIEVED', 'active_jobs':[], 'new_full_continuations':0,
        'remaining_question':'Predictable state residual under stronger independent family support vs later controller response information.'})
    write('hypothesis_status.json',{'guaranteed_significant_gain':'REJECTED',
        'context_not_connected':'REJECTED_FOR_REPAIRED_RUN', 'context_contains_information':'SUPPORTED',
        'state_effect_absent':'NOT_SUPPORTED', 'state_effect_predictable_from_current_h_C':'NOT_ESTABLISHED',
        'augmented_representation_irreducibly_insufficient':'UNDERRESOLVED',
        'source_support_or_learning_limits':'SUPPORTED_AS_COMBINED_EXPLANATION'})
    with (OUT/'experiment_ledger.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['experiment','data','new_full_continuations','result'])
        for name,summary in probes['models'].items():
            w.writerow([name,'frozen source matrix',0,json_string(summary['validation'])])
        w.writerow(['remove_uncommitted_flow','frozen source matrix',0,json_string(read(OUT/'uncommitted_flow_ablation.json')['metrics'])])
        w.writerow(['controller_decoder','frozen source matrix',0,json_string(decode['correct_context'])])
        w.writerow(['micro_response','48 source-only truncated physical probes',0,'3840 derived steps; no task labels'])
    write('artifact_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.py','.json') and p.name!='artifact_hashes.json'})
    print({'decision':decision['ranked_findings'][0]['finding'],'hard_decode':hard_metrics,'decision_relevant_reversals':len(relevant)})


def json_string(x):
    import json
    return json.dumps(x,sort_keys=True)


if __name__=='__main__':main()

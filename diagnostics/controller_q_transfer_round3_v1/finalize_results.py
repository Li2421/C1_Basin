"""Assemble the final compact machine-readable Round 3 ledger from immutable evidence."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text())


def main():
    result=read('results.json')
    post=read('P2PRIME_POSTFLIGHT.json')
    label=read('P2PRIME_LABEL_AUDIT.json')
    p2=read('P2PRIME_RESULTS.json')
    p3=read('P3_RESULTS.json')
    source=read('P2PRIME_SOURCE_COUNT_ANALYSIS.json')
    train=read('P2PRIME_TRAINING.json')
    p3train=read('P3_TRAINING.json')
    p3feat=read('P3_FEATURE_RECEIPT.json')
    assert post['new_raw']==post['total']==67584 and post['missing']==post['quarantined']==0
    assert post['valid']+post['unknown']==67584 and label['observed_valid']==post['valid']
    assert not p2['primary_confirmed'] and p3['status']=='COMPLETED EXPLORATORY DIAGNOSTIC'
    workers=[read('p2prime/'+p.name) for p in sorted((HERE/'p2prime').glob('worker_*.json'))]
    assert len(workers)==27 and sum(x['new'] for x in workers)==67584
    cpu_hours=sum(x['episode_cpu_seconds'] for x in workers)/3600
    rows=[]
    for z in p2['rows']:
        rows.append({key:z[key] for key in ('source_controller_count','arm','held_controller','selected_success',
                   'selected_success_observed','selected_success_upper','selected_unknown','success_by_fit','baselines',
                   'gains','gain_ci95','unknown_robust_familywise_ci','matched_minus_mismatched',
                   'unknown_robust_mismatch_familywise_ci','unknown_robust_all16_fixed_familywise_ci',
                   'finite_seed_oracle_regret','q_mse','state_eta_rank_spearman','state_eta_rank_evaluable_states')})
    result['stages']['P2prime']={
        'status':'COMPLETED; NO PREREGISTERED RELIABLE STATE-CONDITIONED TRANSFER',
        'branch':'P1 gate failed; fixed native v10 Flow, held TF/FT placement',
        'training_controllers':['FF','TT','rotate_negative','isotropic','rotate_positive'],
        'held_controllers':['TF','FT'],'new_state_counts':{'source_train_each':256,'source_validation_each':32,'held_test_each':64},
        'eta_candidates':16,'seed_budget':{'source_train':4,'source_validation':4,'held_test':8,
                         'held_baseline_selection':[0,1,2,3],'held_evaluation':[4,5,6,7]},
        'new_rollouts':67584,'valid':post['valid'],'numerical_unknown':post['unknown'],
        'missing':post['missing'],'quarantined':post['quarantined'],'duplicate_journal_rows':1792,
        'episode_cpu_hours':cpu_hours,'main_worker_new':sum(x['new'] for x in workers if x['index']<22),
        'assistance_worker_new':sum(x['new'] for x in workers if x['index']>=22),
        'training_fits':train['new_fits'],'training_elapsed_seconds':train['elapsed_seconds'],
        'primary_confirmed':p2['primary_confirmed'],'source_prior_transfer_confirmed':p2['source_prior_transfer_confirmed'],
        'qualifications':p2['qualifications'],'source_best_fixed_indices':p2['source_best_fixed_indices'],
        'pooled_best_fixed_eta':p2['pooled_best_fixed_eta'],'nearest_fixed_eta':p2['nearest_fixed_eta'],
        'target_discovery_best_fixed_eta':p2['target_discovery_best_fixed_eta'],
        'target_hindsight_best_fixed_eta':p2['target_hindsight_best_fixed_eta'],
        'selection_rows':rows,'controller_eta_pattern':p2['controller_eta_pattern'],
        'source_count_ablation':source['rows'],
        'r3_aliasing_diagnostic':'P2PRIME_R3_ALIASING_DIAGNOSTIC.json',
        'raw_path':'p2prime/raw/','labels_path':'P2PRIME_LABELS.npz','predictions_checkpoints_path':'p2prime_fits/',
        'postflight':'P2PRIME_POSTFLIGHT.json','full_metrics':'P2PRIME_RESULTS.json',
        'unknown_pairwise_fix':'P2PRIME_PAIRWISE_UNKNOWN_FIX.md',
        'merge_registration_fix':'P2PRIME_MERGE_REGISTRATION_FIX.md',
        'rollout_repeated_after_merge_error':0,
    }
    result['stages']['P3']={
        'status':'COMPLETED EXPLORATORY DIAGNOSTIC AFTER P2PRIME FAILURE',
        'new_rollouts':0,'new_fits':p3train['new_fits'],
        'training_elapsed_seconds':p3train['elapsed_seconds'],
        'static_feature_elapsed_seconds':p3feat['elapsed_seconds'],
        'physical_env_steps_for_static_features':0,
        'rows':p3['rows'],'full_metrics':'P3_RESULTS.json',
        'features':'p3_features/','predictions_checkpoints':'p2prime_fits/C*_R4_t0/ and C*_R4_openloop/',
        'confirmatory':False,
    }
    result['total_new_rollouts']=19968+67584
    result['total_new_rollout_cpu_hours']=result['stages']['P1']['episode_cpu_hours']+cpu_hours
    result['historical_reuse_accounting']={
        'P0_labels_read':106496,'P1_exact_cache_hits':1536,
        'P1_cache_hits_overlap_P0_historical_v10_dev_states':True,
        'sum_is_not_distinct_historical_label_count':True,
        'P2prime_cache_hits':0,
    }
    result['status']='COMPLETE — P2PRIME FAILED PRIMARY SELECTION GATE; P3 DIAGNOSTIC COMPLETED'
    result['interpretation']=('P1 confirms real Flow controller-dependent Q and K16 eta preference on opened development states, '
         'but fails the 8-controller qualification gate. P2 original real-Flow holdout was not executed. '
         'Independent fixed-Flow held-placement P2prime did not confirm reliable state-conditioned eta selection transfer '
         'against pooled/nearest/R0/mismatched/all-K16-fixed references; R4 shows descriptor sensitivity, and R3 has static aliasing. '
         'P3 open-loop static probe improves some Q-pattern metrics but not closed-loop selection. '
         'True held-real-Flow generalization and continuous success basin remain inconclusive.')
    result['report']='REPORT.md';result['handoff']='HANDOFF.md'
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'new_rollouts':result['total_new_rollouts'],'cpu_hours':result['total_new_rollout_cpu_hours'],
                      'P2prime':result['stages']['P2prime']['status'],'P3':result['stages']['P3']['status']}))


if __name__=='__main__':main()

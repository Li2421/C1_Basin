"""Materialize compact final tables and immutable lineage after one evaluation."""
import csv,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .build_source import ROOT,OUT,load,dump,sha

def main():
    result=load(OUT/'final_decision.json');frozen=load(OUT/'models_frozen.json');audit=load(OUT/'post_evaluation_audit.json')
    sx=dict(np.load(OUT/'source_entities.npz'));tx=dict(np.load(OUT/'target_entities.npz'))
    trainix=sorted({r['state_index'] for r in pq.read_table(OUT/'source_pairs.parquet').to_pylist() if r['split']=='train'})
    audit['source_policy_axis_applicability']=np.unique(sx['agents'][trainix,:,10][sx['agent_mask'][trainix]>0]).tolist()
    audit['target_policy_axis_applicability']=np.unique(tx['agents'][:,:,10][tx['agent_mask']>0]).tolist()
    audit['cache_numerical_unknown_slots']=sum(sum(t['unknown_seeds']) for t in load(OUT/'target_truth.json'))
    dump('post_evaluation_audit.json',audit)
    rows=[]
    for name,v in result['models'].items():
        rows.append({'method':name,'B15':v['b15'],'states':v['states'],'unresolved_B15':v['unresolved'],
          'oracle_gap':v['oracle_b15']-v['b15'],'selection_rate_given_B15':v['b15']/v['oracle_b15'],
          'mean_Q_lower':v['mean_selected_Q_lower'],'mean_Q_upper':v['mean_selected_Q_upper'],
          'exact_Q_states':v['selected_exact_states'],'regret_lower':v['mean_regret_lower'],'regret_upper':v['mean_regret_upper'],
          'severe_false_positive':v['severe_false_positive'],'collisions':v['collisions'],
          **({} if name=='ring_only_reference' else v['probability_metrics'])})
    with (OUT/'model_comparison.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    history=[]
    for kind in ('shared','eta_only'):
        for r in frozen[kind]['runs']:
            h=load(OUT/kind/f'seed{r["seed"]}/history.json');best=next(t for t in h if t['step']==r['best_step'])
            for sc,v in best['by_scene'].items():history.append({'model':kind,'seed':r['seed'],'source_scene':sc,'step':r['best_step'],**v})
    with (OUT/'source_validation_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    # B15 probability scores are valid for selection, not Q-calibration metrics.
    result['models']['ring_only_reference']['probability_metrics']=None
    result['models']['ring_only_reference']['score_semantics']='probability of logical B15; not single-continuation Q'
    result['implementation_audit']=audit
    result['generator_phase']='NOT_STARTED_CRITIC_GATE_FAILED'
    result['other_LOSO_folds']='NOT_STARTED_FIRST_FOLD_FAILED'
    result['failure_layer']='critic_zero_shot; source_support_and_state_conditioned_extrapolation_confound'
    result['representation_information_sufficiency']='UNDERRESOLVED'
    result['strict_target_naive_representation_design']=False
    result['source_model_training_and_selection_target_label_free']=True
    dump('final_decision.json',result)
    cache=load(OUT/'cache_preflight.json')['summary']
    dump('runtime_statistics.json',{'new_rollouts':0,'training_gpu_shards_peak':1,'training_cpus':2,
       'training_seconds_excluding_initialization_compilation':sum(r['seconds'] for k in ('shared','eta_only') for r in frozen[k]['runs']),
       'cache':cache,'cache_unknowns_not_executed':True,'no_jobs_left':True})
    paths=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.py','.npz','.parquet','.csv','.md','.sbatch') and p.name!='manifest.json']
    dump('manifest.json',{'files':{p.name:sha(p) for p in paths},'model_freeze_sha256':sha(OUT/'models_frozen.json'),
          'prediction_freeze_sha256':sha(OUT/'target_predictions.json'),'new_rollouts':0,'source_model_mutation_after_target':False})

if __name__=='__main__':main()

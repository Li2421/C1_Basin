"""Evidence integrity and independent formula checks for the scientific audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
import pandas as pd
from analyze import HERE,PREV,PARENT,CACHE,read,save,sha,residual_rank,bootstrap_rank

def main():
    checks={}
    prov=read(HERE/'provenance.json');checks['registered_inputs']=all(prov['checks'].values())
    cfg=read(PARENT/'screening_manifest.json')
    root=__import__('pathlib').Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
    checks['all_runner_registered_code_hashes']=all(sha(root/k)==v for k,v in cfg['controller_code_hashes'].items())
    checks['complete_contexts']=prov['contexts']==21504 and prov['hashes_checked']==7680
    checks['no_new_rollouts_or_actions']=prov['new_rollouts']==0 and prov['new_action_evaluations']==0
    checks['baseline_correction_decomposition']=prov['decomposition_residual_max']<1e-14
    df=pd.read_csv(HERE/'context_metrics.csv.gz',low_memory=False)
    checks['four_placements_per_context']=bool((df.groupby(['task_id','array_prefix']).size()==4).all())
    checks['all_reliable_have_three_finite_singular_values']=bool(df[df.reliable][['sigma1','sigma2','sigma3']].notna().all().all())
    checks['input_basis_nonzero_handling']=bool((df.loc[~df.basis1_nonzero,'basis1_gain'].isna()).all())
    checks['all_initial_panels_complete']=bool((df[df.phase=='initial'].groupby(['scene','chain']).size()==6144).all())
    geom=read(HERE/'geometry.json')
    checks['rank_counts']=all(sum(v['rank_resolved_counts'].values())==v['reliable'] for v in geom.values())
    cells=pd.read_csv(HERE/'basin_action_family_pairs.csv')
    checks['all_six_cell_contrasts']=len(cells)==48*16*6
    checks['group_counts_complete']=all(sum(v['counts'].values())==384 for v in read(HERE/'basin_groups.json').values())
    checks['raw_fd_reconstruction']=True
    tasks=[t for t in read(CACHE/'tasks.json')['tasks'] if t['kind']=='initial' and t['seed']==0]
    for t in [tasks[0],tasks[-1]]:
        with np.load(CACHE/'results'/f"{t['id']}.npz") as z:
            out=z['context0_exec_all'];ix=z['context0_center4_fd_eta_indices']
            for k,h in enumerate([.01,.001,.0001]):
                jac=(out[:,ix[k,:,0]]-out[:,ix[k,:,1]]).transpose(0,2,1)/(2*h)
                checks['raw_fd_reconstruction'] &= bool(np.max(np.abs(jac-z['context0_center4_J_eta_exec'][:,k]))<1e-12)
    checks['fast_bootstrap_matches_full_regression']=True
    for sc in cells.scene.unique():
        g=cells[(cells.scene==sc)&(cells.contrast=='TT-FF')].reset_index(drop=True)
        codes=pd.factorize(g.state_uid)[0]
        for feat in ['log_effective_ratio','subspace_distance']:
            values=g[[feat,'qdiff_low','input_norm','eta_norm']].to_numpy(float)
            for fe in [False,True]:
                fast=bootstrap_rank(values,codes,g.eta_index.to_numpy(int) if fe else None)
                slow=residual_rank(g,feat,'qdiff_low',fe)
                checks['fast_bootstrap_matches_full_regression'] &= abs(fast-slow)<1e-10
    seq=read(HERE/'sequence_provenance.json')
    checks['full_sequence_initial_action_replay']=seq['max_initial_probe_replay_error']==0
    checks['full_sequence_plant_equation']=seq['max_plant_residual']<1e-14
    checks['full_sequence_fixed_noise']=seq['all_paired_noise_identical'] and seq['paired_noise_checks']==1920
    checks['full_sequence_hashes']=seq['unique_rollouts']==960 and seq['hashes_checked']==1920
    counter=pd.DataFrame(read(HERE/'initial_family_counterexamples.json'))
    q=counter[(counter.scene=='toy_give_way')&(counter.contrast=='TT-FT')&(counter.max_bank_action_distance<1e-8)]
    checks['equal_initial_family_counterexample']=len(q)==8 and q.flipped_cells.sum()==28
    refs=['path_integral_checks.json','candidate_linearization.json','state_chart_control.json','fixed_noise_state_dependence.json',
          'recurrence_checks.json','scale_probe_summary.json','validated_counterexample_operators.json','boundary_counterexamples.json',
          'other_scenario_probes.json','runtime_inspection.json','verification.json','final_report.txt']
    save('previous_evidence_hashes.json',{str(PREV/f):sha(PREV/f) for f in refs})
    path=read(PREV/'path_integral_checks.json')
    checks['previous_path_integral_evidence']=len(path)==8 and max(v['quadrature_errors']['33']['relative_error'] for v in path)<.011
    save('verification.json',dict(passed=all(checks.values()),checks=checks,n_checks=len(checks),script_sha256=sha(__file__)))
    assert all(checks.values()),checks
    print('PASS',len(checks),'scientific integrity checks')
if __name__=='__main__':main()

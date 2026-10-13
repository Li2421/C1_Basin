"""Complete the single sampling intervention, including every held-out fold."""
import csv,json
from .study import HERE, SEEDS, FOLDS, read, write, sha


def dump(name, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def run():
    m=read(HERE/'matrix_results.json');i=read(HERE/'interaction_results.json');n=read(HERE/'native_results.json')
    q=[r for sc in ('toy_give_way','ring_exchange') for r in read(HERE/f'q16_{sc}_results.json')['rows']]
    fits=[read(HERE/'fits'/fold/arm/f'seed{s}/complete.json') for fold in FOLDS for arm in ('eta_only','union_native_slots') for s in SEEDS]
    assert len(fits)==30 and read(HERE/'source_gate.json')['passed']
    table=[]
    for r in m['rows']:
        if r['reference']!='flow' or not (r['panel']=='all_held' or r['fold']=='joint' and r['panel']=='validation'):continue
        key={k:r[k] for k in ('fold','scene','panel','reference','condition')}
        c={x['comparison']:x for x in m['comparisons'] if all(x[k]==v for k,v in key.items())}
        mechanism=next(x for x in m['mechanisms'] if all(x[k]==v for k,v in key.items()))
        triple=next(x for x in i['rows'] if all(x[k]==r[k] for k in ('fold','scene','reference')) and x['first']==r['condition'] and x['second']=='zero' and x['interaction']=='state_controller_eta')
        table.append(dict(**key,N=r['N'],controllers=r['controllers'],success=r['selected_success_bounds'],oracle=r['oracle_success_bounds'],
            regret=r['oracle_regret_bounds'],Q_MSE=r['Q_MSE_midpoint_sensitivity'],state_effect_corr=r['state_effect_corr_midpoint_sensitivity'],
            state_eta_corr=r['state_eta_corr_midpoint_sensitivity'],mismatch=c['matched_minus_same_scene_wrong_state'],
            versus_eta=c['selection_minus_eta_only'],versus_fixed=c['selection_minus_posthoc_best_fixed_per_controller'],
            versus_uniform=c.get('selection_improvement_over_uniform_native_slots'),interaction_skill=triple,
            reversal=mechanism['reversal'],entropy_bits=r['selection_entropy_bits'],selected_collisions=r['selected_collision'],
            selected_unknown=r['selected_numerical_unknown'],selected_timeout=r['selected_timeout'],selected_deadlock=r['selected_deadlock']))
    dump('matrix_summary.csv',table);dump('native_summary.csv',n['rows']);dump('q16_summary.csv',q)
    dump('native_paired.csv',n['paired_new_minus_uniform'])
    # All individual cases and selections already saved by the reused engines, not selected favorable examples.
    write(HERE/'artifact_manifest.json',dict(
        protocols=['protocol.json','evaluation_protocol.json','native_evaluation_protocol.json','interaction_protocol.json'],
        states_labels='dataset.npz and original per-seed files referenced in dataset_audit.json; native raw outcomes in ../controller_entity_coverage_check_v1/dataset.npz',
        matrix_raw='matrix_review/raw_labels.npz',matrix_cases='matrix_review/development_predictions.npz and development_results.json',
        native_predictions='native_predictions_and_controls.npz',
        q16_predictions='q16_*_predictions_and_controls.npz',bootstrap='*_bootstrap.npz and matrix_review/development_bootstrap.npz',
        normalization='Each fit normalization.json is source TRAIN-only',all_states_development=True))
    lines=['STAGE22: RESTORE THE EXISTING NATIVE20% SAMPLER',
        'Result: REPAIRED_LOCAL_NATIVE_LEARNING_BUT_NOT_CONTROLLER_CONDITIONED_LOSO. Research remains active.',
        'One intervention, no weight sweep: native sampling5.263% ->20% within native-containing source scenes; other controllers share80%. Same cached97,280 outcomes, current physical/API-slot+compact64 input, architecture, optimizer,6000steps and source-only checkpoint rule. Fair eta-only retrained with identical draws/labels.',
        '30 new fits,30 old fits reused. All four held folds completed after the prospectively frozen SOURCE gate passed. No new physical rollouts or static controller queries. Three new native TEST inferences; old native predictions reused.',
        'All evaluated states are opened DEVELOPMENT. Joint sourceVAL rows are descriptive; native64 and opened Q16 have no training/validation state overlap. No untouched positive confirmation or formal K16/B15 four-scene benchmark is claimed.',
        'Statistics:20000 physical-state paired bootstraps, not rollout resampling. Mismatch donors and recipients jointly resampled;200 saved derangements. Numerical outcomes unidentified -> success intervals and censor-aware CIs, not silently failures. All fixed candidates included; best fixed reselected inside bootstrap.',
        'Matrix K4 Q2/Q4; native Ring64 K16 Q16; independent analytic Q16 panels32 physicalstates with2 unseen controllers. No fabricated same-controller Safety baseline. Raw labels and every failure/state retained.',
        'Matrix engine aliases are documented in ANALYSIS_CORRECTION.json: no new state-erasure/encounter intervention occurred in this stage. Old donor-fixed bootstrap corrected equally for cachedold/new models.','',
        'PRIMARY NATIVE64: arm Q_MSE state_eta_corr success B15 matched-minus-mismatch95CI versus-reselected-best-fixed95CI']
    for r in n['rows']:
        if r['panel']=='test_all':lines.append(f"{r['arm']} {r['Q_MSE_midpoint']:.6f} {r['state_eta_corr']} {r['success_bounds']} {r['B15_bounds']}/64 {r['matched_minus_mismatch']['censor_CI95']} {r['minus_bootstrap_reselected_per_controller_fixed']['censor_CI95']}")
    lines.extend(['Native64 paired new minus uniform: '+json.dumps(n['paired_new_minus_uniform'][0]),'',
        'FOUR-SCENE FLOW-REFERENCE DEVELOPMENT: fold arm success oracle mismatch95CI vs-fixed95CI SxCxeta-skill95CI'])
    for r in table:
        lines.append(f"{r['fold']} {r['scene']} N={r['N']} {r['condition']} {r['success']} {r['oracle']} {r['mismatch']['censor_CI95']} {r['versus_fixed']['censor_CI95']} {r['interaction_skill']['censor_CI95']}")
    lines.extend(['','Independent32-state Q16 unseen analytic6/7: scene fold arm success state_eta_corr mismatch95CI'])
    for r in q:lines.append(f"{r['scene']} {r['variant']} {r['arm']} {r['success_bounds']} {r['state_eta_corr']} {r['matched_minus_mismatch']['censor_CI95']}")
    lines.extend(['','Causal interpretation:',
        'The sampler restoration improves independent native Ring Q, interaction and selected success, beyond the source gate. Native success rises92.09% ->95.41%; paired+3.32pp95CI[+0.68,+6.74]. Local underexposure/training allocation was consequential and repairable.',
        'All four held-scene state-mismatch CIs still include0; all four Flow-reference SxCxeta skill CIs are negative. Large DB success gains over eta-only are global/controller-prior effects and are worse than a fixed eta. They do not establish state-conditioned transfer.',
        'This rules out native-data dilution as a sufficient explanation or repair. It does not exclude joint causes or prove current descriptors sufficient. Do not tune more sampling weights on these results.',
        'The remaining input-sufficiency versus learnable-composition/support distinction is unresolved. No theoretical impossibility claim and no complete root-cause claim.',
        'Safety: all current matrix/Q16/native selected collisions are0. Unknown/numerical, timeout and deadlock counts remain separately in original results and summary tables. Zero observed collisions is not a universal safety guarantee.'])
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_LOCAL_REPAIR_NOT_LOSO',new_fits=30,reused_fits=30,new_rollouts=0,new_static_queries=0,
        all_four_folds_complete=True,native_test_states=64,root_cause_closed=False,untouched_confirmation=False,report_sha256=sha(HERE/'REPORT.txt')))
    print('Stage22 complete; native repair independently supported, all four LOSO falsification failures retained.')


if __name__=='__main__':run()

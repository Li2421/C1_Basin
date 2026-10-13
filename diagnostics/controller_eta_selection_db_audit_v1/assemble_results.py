"""Build the compact, machine-readable cross-stage result and provenance index."""
from pathlib import Path
import json,hashlib,datetime,subprocess,sys,sqlite3
H=Path(__file__).resolve().parent
DB=H.parents[1]/'shared_rollout_db/rollout.sqlite'
def get(name):return json.loads((H/name).read_text())
def hashfile(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
    inv=get('inventory.json');ext=get('EXTRACTION_AUDIT.json');idx=get('PANEL_INDEX.json')
    panel=get('E0_PANEL_RESULTS.json');dec=get('E0_DECISION.json');legacy=get('LEGACY_13_AUDIT.json')
    reuse=get('REUSE_AUDIT.json');e3=get('E3_COVERAGE_GAP.json');e4=get('E4_REGIME_RESULTS.json');e5=get('E5_DENSE_ETA_AUDIT.json')
    ctl=get('CONTROLLER_COMPARISONS.json');src=get('SOURCE_FILES.json')
    decisions={x['panel_uid']:x for x in dec['panel_decisions']}
    panel_rows=[]
    for p in panel:
        d=decisions[p['panel_uid']]
        panel_rows.append({k:p[k] for k in ['panel_uid','scene','controller_uid','n_states','n_eta','selected_rollouts','heldout_fixed_success','heldout_state_specific_success','heldout_gain','heldout_gain_ci95','robust_fixed','robust_state_specific','robust_gain','robust_gain_ci95','fixed_failure_U','all_eta_fail_I','U_minus_I','unique_full_label_best_states','tied_full_label_best_states','number_different_full_label_argmax_eta','B15_cover','random_split_sensitivity']}
                          |{k:d[k] for k in ['strong_fixed_eta_uid','strong_fixed_included','raw_gate_pass','primary_gate_pass','veto_reason']})
    old=H.parent/'state_nuisance_rootcause_v1/heldout_adaptive_selection_v1'
    a=json.loads((old/'audit_results.json').read_text())
    verification=json.loads((old/'verification.json').read_text())
    c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    counts=c.execute('SELECT COUNT(*),SUM(conflict_quarantined),SUM(numerical_failure) FROM rollout').fetchone()
    out={'schema':'controller_eta_selection_db_audit_v1',
        'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'python':sys.version.split()[0],
        'preregistration':get('preregistration_receipt.json'),
        'e4_addendum_receipt':get('addendum_e4_receipt.json'),
        'database':{'path':str(DB),'bytes':DB.stat().st_size,'sha256':hashfile(DB),
                    'rollout_rows':counts[0],'quarantined_rollout_rows':counts[1],
                    'numerical_failure_rows':counts[2],'valid_binary_rows':inv['valid_rollouts'],
                    'Q16_cells':inv['Q16_cells'],'Q16_controllers':inv['Q16_controllers']},
        'reuse':reuse,
        'E0':{'status':'COMPLETE','structural_panels_ge12':inv['structural_panels_ge12_state_Kge2'],
              'metadata_eligible_panels':inv['metadata_eligible_panels'],
              'seed_aligned_panels':ext['aligned_panels'],
              'unaligned_panels':ext['seed_alignment_failed'],
              'raw_gate_pass':dec['raw_gate_pass'],'primary_gate_pass':dec['primary_gate_pass'],
              'nonindependent_nested_panels_note':'Most Ring passes overlap states, eta and outcomes; do not count 32 as independent confirmations.',
              'primary_by_scene':dec['primary_by_scene'],
              'panel_rows':panel_rows,
              'previous_13':legacy,
              'metadata_exclusion_reasons':inv['exclusion_reason_counts'],
              'source_file_paths':len(src),'source_file_paths_missing':sum(not x['exists'] for x in src),
              'primary_panel_raw_source_missing':0,
              'detailed_panels_file':'E0_PANEL_RESULTS.json',
              'panel_index_file':'PANEL_INDEX.json',
              'inventory_file':'inventory.json'},
        'E1':{'status':'NOT_TRIGGERED','reason':'E0 contains a qualified Ring panel with the known strongest fixed eta included; no precision supplementation allowed by decision rule','new_rollouts':0},
        'E2':{'status':'PRIOR_PROSPECTIVE_CONFIRMATION_REUSED',
              'independence_caveat':'Historical 256 states were prospectively independent of their own earlier 608-state training/development data; 61,648 rollout labels overlap the present E0 inventory, so they are not fresh confirmation relative to this retrospective E0 screen.',
              'source_path':str(old/'audit_results.json'),'source_sha256':hashfile(old/'audit_results.json'),
              'protocol_path':str(old/'protocol.json'),'protocol_sha256':hashfile(old/'protocol.json'),
              'verification':verification,
              'n_states':256,'n_eta':16,'n_seeds_per_cell':16,
              'physical_rollout_attempts':a['dataset_audit']['attempted'],
              'numerical_unknowns':a['dataset_audit']['numerical_unknown'],
              'frozen_state_selector_success':a['cohorts']['all']['selectors']['qstate512']['top1_rollout_success_lower'],
              'best_fixed_success':a['cohorts']['all']['selectors']['best_fixed']['top1_rollout_success_lower'],
              'gain_pp':100*a['cohorts']['all']['comparisons']['qstate512_minus_qeta']['mean'],
              'gain_state_bootstrap_ci95_pp':[100*x for x in a['cohorts']['all']['comparisons']['qstate512_minus_qeta']['paired_physical_state_bootstrap_95']],
              'gain_over_bootstrap_max_fixed_ci95_pp':[100*x for x in a['cohorts']['all']['all_fixed']['simultaneous_max_over_16_fixed_state_bootstrap_95']],
              'frozen_fit_seeds':[17,23,41],
              'fitwise_gain_pp':[100*x for x in a['cohorts']['all']['comparisons']['qstate512_minus_qeta']['by_fit']],
              'frozen_32_state_model_success':a['cohorts']['all']['selectors']['qstate32']['top1_rollout_success_lower'],
              'frozen_512_state_model_minus_32_state_model_pp':100*a['cohorts']['all']['comparisons']['qstate512_minus_qstate32']['mean'],
              'frozen_512_minus_32_ci95_pp':[100*x for x in a['cohorts']['all']['comparisons']['qstate512_minus_qstate32']['paired_physical_state_bootstrap_95']],
              'new_rollouts_this_task':0},
        'E3':e3,
        'E4':{'status':'COMPLETE_RETROSPECTIVE_FACTOR_LOCO_AND_HISTORICAL_CALIBRATION_REUSE',
              'factor_data_path':str(H.parent/'controller_state_geometry_transfer_v1/data/factors.npz'),
              'factor_source_sha256':e4['data_sha256'],
              'zero_shot_feature_states':e4['zero_shot_feature_states'],
              'leave_physical_geometry_out':e4['leave_physical_geometry_out']['method_summary'],
              'leave_topology_out':e4['leave_topology_out']['method_summary'],
              'detailed_results_file':'E4_REGIME_RESULTS.json',
              'historical_calibration_path':str(H.parent/'controller_state_geometry_transfer_v1/calibration_results.json'),
              'historical_calibration_sha256':hashfile(H.parent/'controller_state_geometry_transfer_v1/calibration_results.json'),
              'new_rollouts_this_task':0,
              'mechanism_invariance_verdict':'INCONCLUSIVE: proxy coordinates do not show a reliable gain over raw geometry, and only two topologies exist'},
        'E5':{'status':'DATABASE_AND_HISTORICAL_CONTINUOUS_AUDIT_ONLY',**e5,
              'historical_continuous_failure_path':str(H.parent/'orthoflow3_continuous_basin_critic_v1/failure_diagnosis.md'),
              'historical_continuous_failure_sha256':hashfile(H.parent/'orthoflow3_continuous_basin_critic_v1/failure_diagnosis.md'),
              'new_rollouts':0},
        'controller_preference_comparisons':ctl,
        'conclusions':{'within_ring_state_conditioned_eta_selection':'SUPPORTED_FOR_FIXED_RING_FF_K16_INITIAL_STATE_DISTRIBUTION',
                       'across_geometry_selection_transfer':'INCONCLUSIVE',
                       'controller_dependent_eta_regime':'DESCRIPTIVE_SUPPORT_ON_MATCHED_TOY_TT_FF_PAIR',
                       'mechanism_coordinate_invariance':'INCONCLUSIVE',
                       'continuous_success_basin_generalization':'INCONCLUSIVE'}}
    (H/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print({'results_bytes':(H/'results.json').stat().st_size,'database_sha256':out['database']['sha256'],
           'E0_primary_pass':out['E0']['primary_gate_pass'],'total_unique_reused':reuse['unique_reused_overall']},flush=True)
if __name__=='__main__':main()

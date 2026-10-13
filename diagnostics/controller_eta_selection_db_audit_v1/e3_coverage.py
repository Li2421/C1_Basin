"""Structural feasibility and reused historical transfer controls for conditional E3."""
from pathlib import Path
import sqlite3,json,hashlib
H=Path(__file__).resolve().parent
DB=H.parents[1]/'shared_rollout_db/rollout.sqlite'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dec=json.loads((H/'E0_DECISION.json').read_text())
    passed=[x for x in dec['panel_decisions'] if x['primary_gate_pass']]
    ids=sorted({x['controller_uid'] for x in passed})
    c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    configs=[]
    for uid in ids:
        row=c.execute('SELECT scenario_uid,horizon,flow_checkpoint_sha256,config_json FROM controller_config WHERE controller_uid=?',(uid,)).fetchone()
        cfg=json.loads(row[3]);sc=c.execute('SELECT name,code_config_fingerprint,metadata_json FROM scenario WHERE scenario_uid=?',(row[0],)).fetchone()
        configs.append({'controller_uid':uid,'scene':sc[0],'scenario_uid':row[0],
                        'scenario_code_config_fingerprint':sc[1],
                        'scenario_metadata_sha256':hashlib.sha256(sc[2].encode()).hexdigest(),
                        'horizon':row[1],'flow_checkpoint_sha256':row[2],
                        'chain':cfg.get('chain'),'width_param_in_controller':cfg.get('width'),
                        'single_scenario_uid_for_all_Q16_states':True})
    old_ring=H.parent/'state_nuisance_rootcause_v1/heldout_adaptive_selection_v1'
    old_loso=H.parent/'transferable_state_conditioning_v1/stage_v3_coverage/final_phase1_32/final/INTERPRETATION.md'
    old_width=H.parent/'controller_state_geometry_transfer_v1/REPORT.md'
    n_geom=6;n_state=16;n_eta=16;n_seed=16
    out={'status':'PARTIAL_REUSE_ONLY__FORMAL_MANY_FEW_EXTRAP_NOT_EXECUTED',
         'passing_controller_profiles':configs,
         'reason':'No audited passing Ring controller has a matched multi-geometry Q16 panel. The historical Toy width and analytic factor panels use other profiles / low selection headroom. Changing Ring geometry changes execution semantics and needs new labels, so a formal E3 cannot be answered from compatible cache.',
         'minimum_formal_design_assumption':'4 train + 2 test geometries, 16 independent states each, frozen 16 η, 16 future seeds; one controller; excludes pilot and training repeats',
         'lower_bound_new_rollouts_one_controller':n_geom*n_state*n_eta*n_seed,
         'lower_bound_new_rollouts_two_controllers':2*n_geom*n_state*n_eta*n_seed,
         'rough_raw_storage_MB_one_controller':264*(n_geom*n_state*n_eta*n_seed)/65536,
         'runtime':'Not estimated from a reliable per-episode benchmark; prior Ring batch had 65,536 attempts. A new geometry-validity pilot and resource admission would precede scheduling.',
         'new_rollouts':0,
         'retrospective_reuse':[{'path':str(old_ring/'audit_results.json'),'sha256':sha(old_ring/'audit_results.json'),
                                 'scope':'within-Ring fresh-state selector: Q512 vs Q32, not geometry transfer'},
                                {'path':str(old_loso),'sha256':sha(old_loso),
                                 'scope':'four-scene strict LOSO; controller/geometry/horizon confounded; all four LOSO critics lacked reliable matched-state advantage'},
                                {'path':str(old_width),'sha256':sha(old_width),
                                 'scope':'Toy width MANY/FEW/EXTRAP improves Q, but eta success ceiling masks selection headroom'}],
         'scientific_conclusion':'Cross-geometry state-conditioned η-selection gain remains INCONCLUSIVE. Existing Ring within-scene positive and historical cross-scene failure are not interchangeable.'}
    (H/'E3_COVERAGE_GAP.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print({'passing_profiles':len(configs),'lower_bound_new_rollouts_one_controller':out['lower_bound_new_rollouts_one_controller']})
if __name__=='__main__':main()

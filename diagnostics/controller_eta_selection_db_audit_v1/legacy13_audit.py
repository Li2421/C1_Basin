"""Trace all 13 previously mentioned Q16 cohorts without relaxing the new ≥12-state gate."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
prefixes=["ctl_6ef16b8eb8c94f40","ctl_d1de2ed5b6ecafdc","ctl_0e56fc46cc1c6772",
"field_external_toy_v","ctl_ceef00016f37f243","ctl_790d13021b72de22",
"ctl_f262448d380b7c42","ctl_68504985c54feb85","ctl_dfbc896b68ce8a04",
"ctl_f1d8d989cbbab340","ctl_ff4013f13f82acee","ctl_eae6875700502bfe",
"ctl_8eef4063bda464ba"]
def main():
    inv=json.loads((H/'inventory.json').read_text())
    results={r['panel_uid']:r for r in json.loads((H/'E0_PANEL_RESULTS.json').read_text())}
    panelindex={r['panel_uid']:r for r in json.loads((H/'PANEL_INDEX.json').read_text())}
    out=[]
    for prefix in prefixes:
        controls=[x for x in inv['controllers'] if x['controller_uid'].startswith(prefix)]
        ps=[x for x in inv['panels'] if x['controller_uid'].startswith(prefix)]
        controls=sorted(controls,key=lambda x:x['Q16_cells'],reverse=True)
        ps=sorted(ps,key=lambda x:x['n_states']*x['n_eta'],reverse=True)
        best=ps[0] if ps else None
        bres=results.get(best['panel_uid']) if best else None
        out.append({'previous_prefix':prefix,'matching_controller_uids':[x['controller_uid'] for x in controls],
        'total_Q16_cells_for_matching_controllers':sum(x['Q16_cells'] for x in controls),
        'largest_structural_panel':{k:best[k] for k in ['panel_uid','n_states','n_eta','eligible_metadata','eligibility_reasons']} if best else None,
        'largest_panel_seed_alignment_status':panelindex[best['panel_uid']]['status'] if best and best['panel_uid'] in panelindex else None,
        'largest_panel_E0_gain':bres['heldout_gain'] if bres else None,
        'largest_panel_E0_CI95':bres['heldout_gain_ci95'] if bres else None,
        'largest_panel_E0_gate_pass':bres['preregistered_gate_pass'] if bres else None,
        'reason_no_E0':None if bres else ('below_12_state_gate_or_no_complete_panel' if not best else ('metadata_excluded' if not best['eligible_metadata'] else 'seed_alignment_failed'))})
    (H/'LEGACY_13_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print({'previous_prefixes':len(out),'matched':sum(bool(x['matching_controller_uids']) for x in out),'E0_scored':sum(x['largest_panel_E0_gain'] is not None for x in out)},flush=True)
if __name__=='__main__':main()

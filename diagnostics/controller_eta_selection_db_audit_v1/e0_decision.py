"""Enforce predeclared inclusion of known strong fixed eta in E0 gates."""
from pathlib import Path
from collections import Counter
import json,sqlite3
H=Path(__file__).resolve().parent
DB=H.parents[1]/'shared_rollout_db/rollout.sqlite'
def main():
    p=json.loads((H/'E0_PANEL_RESULTS.json').read_text())
    db=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    historical=json.loads((H.parent/'state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/protocol.json').read_text())
    ring_vec=historical['eta'][13]
    rows=db.execute('SELECT eta_uid FROM eta WHERE eta1=? AND eta2=? AND eta3=?',tuple(ring_vec)).fetchall()
    assert len(rows)==1
    ring_best=rows[0][0]
    strongest={historical['controller']['controller_uid']:ring_best}
    # For other controllers, use the broadest already-catalogued full bank only
    # as a conservative anti-cherry-picking check. This may veto a small cohort
    # but cannot create any new passing panel.
    controllers=sorted({x['controller_uid'] for x in p})
    for cid in controllers:
        if cid in strongest:continue
        options=[x for x in p if x['controller_uid']==cid]
        ref=max(options,key=lambda x:(x['n_eta'],x['n_states']*x['n_eta']))
        j=ref['paired_even_odd'][0]['fixed_index']
        strongest[cid]=ref['eta_uids'][j]
    out=[]
    for x in p:
        known=strongest[x['controller_uid']]
        included=known in x['eta_uids']
        out.append({'panel_uid':x['panel_uid'],'scene':x['scene'],
                    'controller_uid':x['controller_uid'],
                    'raw_gate_pass':x['preregistered_gate_pass'],
                    'strong_fixed_eta_uid':known,'strong_fixed_included':included,
                    'primary_gate_pass':bool(x['preregistered_gate_pass'] and included),
                    'veto_reason':'known_strong_fixed_eta_absent' if not included else None})
    summary={'raw_gate_pass':sum(x['raw_gate_pass'] for x in out),
             'primary_gate_pass':sum(x['primary_gate_pass'] for x in out),
             'raw_gate_vetoed':sum(x['raw_gate_pass'] and not x['strong_fixed_included'] for x in out),
             'primary_by_scene':dict(Counter(x['scene'] for x in out if x['primary_gate_pass'])),
             'reference_policy':'Ring η13 from historically frozen 256-state protocol; other controllers broadest Q16 panel fixed choice used only as conservative veto, never to promote result',
             'ring_historical_eta13_uid':ring_best,'panel_decisions':out}
    (H/'E0_DECISION.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print({k:summary[k] for k in ['raw_gate_pass','primary_gate_pass','raw_gate_vetoed','primary_by_scene']})
if __name__=='__main__':main()

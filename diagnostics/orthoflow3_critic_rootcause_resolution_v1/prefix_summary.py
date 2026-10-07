"""Paired exact test; no optional statsmodels dependency or executor changes."""
import json
import numpy as np
from scipy.stats import binomtest
from .prefix_alias import OUT,read,write,csvwrite,connect,canonical

def main():
    p=read(OUT/'protocol.json');rows=[]
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    with connect(True) as db:
      for pair in read(OUT/'pairs.json'):
        records=[]
        for c in (p['parent_profile'],p['profiles'][0]):
            rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(pair['state_uid'],pair['eta_uid'],c['controller_uid']))}
            ordered=[rr[canonical({'future_index':k})] for k in range(16)]
            assert all(r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined'] for r in ordered)
            records.append(ordered)
        comparable=[(a,b) for a,b in zip(*records) if not a['numerical_failure'] and not b['numerical_failure']]
        a=np.array([r[0]['success'] for r in comparable]);b=np.array([r[1]['success'] for r in comparable]);rescue=int(((a==0)&(b==1)).sum());brk=int(((a==1)&(b==0)).sum())
        rows.append(dict(state_uid=pair['state_uid'],eta_index=pair['eta_index'],paired_valid=len(a),
            parent_success=sum(r['success'] for r in records[0] if not r['numerical_failure']),
            prefix_success=sum(r['success'] for r in records[1] if not r['numerical_failure']),
            parent_numerical=sum(r['numerical_failure'] for r in records[0]),prefix_numerical=sum(r['numerical_failure'] for r in records[1]),
            paired_Q_delta=float((b-a).mean()),rescue=rescue,breaks=brk,
            McNemar_exact_p=float(binomtest(rescue,rescue+brk,.5).pvalue) if rescue+brk else 1.))
    order=np.argsort([r['McNemar_exact_p'] for r in rows]);previous=0.
    for j,i in enumerate(order):
        previous=min(1.,max(previous,(len(rows)-j)*rows[i]['McNemar_exact_p']));rows[i]['Holm_p']=previous
    csvwrite(OUT/'paired_controller_Q.csv',rows)
    summary=dict(cells=len(rows),significant_Holm=sum(r['Holm_p']<.05 for r in rows),
        mean_abs_Q_change=float(np.mean([abs(r['paired_Q_delta']) for r in rows])),
        robust_to_complete_failure=sum(r['parent_success']>=15 and r['prefix_success']==0 and r['prefix_numerical']==0 for r in rows),
        identical_H20_inputs=True,source_only_posthoc_controller_program_intervention=True,
        implication='H20 input insufficiency is established for the delayed-switch program class if matched outcomes differ significantly. This does not establish full-input aliasing among the natural stationary controllers or explain all their model errors.',
        independent_family_selection=True,new_controller_labels_used_for_model_selection=False)
    write(OUT/'identifiability_result.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

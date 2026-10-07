"""Task-level consequence of a false controller-slot invariance assumption."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
from shared_rollout_db.src.rollout_db import connect
from .replication_analysis import read,write,csvwrite,OUT

def main():
    source=OUT/'permutation_outcomes';states=read(source/'states.json');pairs=read(source/'pairs.json')
    profiles=read(source/'protocol.json')['profiles'];rows=[]
    with connect(True) as db:
      for c in profiles:
       for pair in pairs:
        parent=pair['parent_state_uid'];values=[]
        for state_uid,lo,hi in ((parent,16,64),(pair['state_uid'],0,32)):
            records=db.execute('SELECT seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(state_uid,pair['eta_uid'],c['controller_uid'])).fetchall()
            records=[r for r in records if lo<=json.loads(r['seed_key'])['future_index']<hi]
            assert len(records)==hi-lo
            assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in records)
            good=[r for r in records if not r['numerical_failure']]
            s=sum(r['success'] for r in good);f=len(good)-s;values.append((s,f,len(records)-len(good)))
        (s0,f0,num0),(s1,f1,num1)=values;p=float(fisher_exact([[s0,f0],[s1,f1]]).pvalue)
        rows.append(dict(controller=c['name'],parent_state_uid=parent,permuted_state_uid=pair['state_uid'],eta_index=pair['eta_index'],
            original_success=s0,original_failure=f0,original_numerical=num0,
            permuted_success=s1,permuted_failure=f1,permuted_numerical=num1,
            original_Q=s0/(s0+f0),permuted_Q=s1/(s1+f1),Q_difference=s1/(s1+f1)-s0/(s0+f0),Fisher_p=p))
    order=np.argsort([r['Fisher_p'] for r in rows]);previous=0
    for rank,i in enumerate(order):
        previous=max(previous,min(1,rows[i]['Fisher_p']*(len(rows)-rank)));rows[i]['Holm_p']=previous
    csvwrite(OUT/'role_permutation_Q.csv',rows)
    significant=[r for r in rows if r['Holm_p']<.05]
    summary=dict(states=len(states),controller_eta_cells=len(rows),absolute_Q_difference_median=float(np.median([abs(r['Q_difference']) for r in rows])),
        absolute_Q_difference_at_least_quarter=sum(abs(r['Q_difference'])>=.25 for r in rows),
        absolute_Q_difference_at_least_half=sum(abs(r['Q_difference'])>=.5 for r in rows),
        significant_cells_FWER05=len(significant),significant_source_families=len(set(r['parent_state_uid'] for r in significant)),
        interpretation='Native Flow role assignment is a Q-relevant variable if outcome changes replicate; this alone does not prove full h+C alias because C is recomputed under each controller state.',
        no_controller_or_success_definition_changed=True,independent_not_matched_seed_namespaces=True,
        verdict='TASK_Q_NOT_AGENT_SLOT_INVARIANT' if significant else 'ACTION_ASYMMETRY_WITH_TASK_Q_EFFECT_NOT_ESTABLISHED')
    write(OUT/'role_permutation_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

"""Paired-seed sufficiency test and empirical unavoidable-error diagnostic."""
import json
import numpy as np
from scipy.stats import binomtest
from .middle_phase_alias import OUT,EXP,read,write,DBROOT
from .goal_response_cv import csvwrite
from shared_rollout_db.src.rollout_db import connect,canonical


def main():
    protocol=read(OUT/'protocol.json');pairs=read(OUT/'pairs.json');rows=[]
    with connect(True) as db:
      for i,p in enumerate(pairs):
        records=[]
        for c in (protocol['parent_profile'],protocol['profiles'][0]):
            rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],c['controller_uid']))}
            rr=[rr[canonical({'future_index':k})] for k in range(16)]
            assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr);records.append(rr)
        valid=[(a,b) for a,b in zip(*records) if not a['numerical_failure'] and not b['numerical_failure']]
        a=np.array([x['success'] for x,y in valid]);b=np.array([y['success'] for x,y in valid]);qa,qb=a.mean(),b.mean();pm=(qa+qb)/2
        r=int(((a==0)&(b==1)).sum());br=int(((a==1)&(b==0)).sum())
        def entropy(p):
            return -(p*np.log(max(p,1e-12))+(1-p)*np.log(max(1-p,1e-12)))
        rows.append(dict(pair_index=i,state_uid=p['state_uid'],eta_index=p['eta_index'],paired_valid=len(valid),parent_Q=float(qa),middle_Q=float(qb),Q_change=float(qb-qa),
            parent_success=sum(x['success'] for x in records[0] if not x['numerical_failure']),middle_success=sum(x['success'] for x in records[1] if not x['numerical_failure']),
            parent_numerical=sum(x['numerical_failure'] for x in records[0]),middle_numerical=sum(x['numerical_failure'] for x in records[1]),
            rescue=r,breaks=br,exact_p=float(binomtest(r,r+br,.5).pvalue) if r+br else 1.,
            empirical_min_Q_squared_error=(float(qa-qb)**2)/4,
            empirical_min_excess_NLL=entropy(pm)-(entropy(qa)+entropy(qb))/2))
    pp=np.array([r['exact_p'] for r in rows]);order=np.argsort(pp);adj=np.empty(len(pp));adj[order]=np.minimum(1,np.maximum.accumulate(pp[order]*(len(pp)-np.arange(len(pp)))))
    for r,a in zip(rows,adj):r['Holm_p']=float(a)
    csvwrite(OUT/'matched_Q_alias_cells.csv',rows)
    raw=[json.loads(line)['record'] for p in (DBROOT/'journals'/EXP).glob('*.jsonl') for line in p.read_text().splitlines()];active=[r for r in raw if r['middle_active_steps']>0]
    proof=read(OUT/'input_identity_proof.json');assert all(proof[k] for k in ('h_equal','H20_equal','H80_equal','agent_response_equal','goal_response_equal'))
    summary=dict(cells=len(rows),new_continuations=len(raw),active_rollouts=len(active),
        first_active_range=[min(r['middle_first_active_step'] for r in active),max(r['middle_first_active_step'] for r in active)] if active else None,
        parent_B15=sum(r['parent_success']>=15 for r in rows),middle_B15=sum(r['middle_success']>=15 for r in rows),
        numerical=sum(r['middle_numerical'] for r in rows),Holm_significant_cells=int((adj<.05).sum()),
        mean_absolute_Q_change=float(np.mean([abs(r['Q_change']) for r in rows])),
        observed_error_floor_Q_MSE=float(np.mean([r['empirical_min_Q_squared_error'] for r in rows])),
        error_floor_caveat='Empirical equalcondition-weight floor; not a population bound without labeluncertainty. Significance tests are pairedseed exact with16cellHolmcorrection.',
        all_current_input_groups_identical=True,
        adjudication='CURRENT_RESPONSE_SUMMARY_NOT_SUFFICIENT_FOR_STATIONARY_CONTROLLER_PROGRAMS' if (adj<.05).any() else 'NO_SIGNIFICANT_COUNTEREXAMPLE_IN_THIS_PANEL',
        scope='Stationary physicallygatedFlowprogram intervention on frozen8sourceVALfamilies,2seeneta. DoesNOTprovealiasing among naturalMLPcheckpointclass or impossibilityofallcrossscenelearning.',
        new_labels_enter_training=False,safety_eta_success_horizon_unchanged=True)
    write(OUT/'adjudication.json',summary);print(summary)


if __name__=='__main__':main()

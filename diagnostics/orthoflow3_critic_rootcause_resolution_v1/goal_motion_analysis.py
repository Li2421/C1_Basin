"""All-cell paired-seed audit; no model fitting or outcome-selected subset."""
import json
import numpy as np
from scipy.stats import binomtest
from .goal_motion_alias import OUT, EXP, read, write, DBROOT
from .goal_response_cv import csvwrite
from shared_rollout_db.src.rollout_db import connect, canonical


def main():
    p = read(OUT/'protocol.json')
    rows = []
    with connect(True) as db:
        for i, pair in enumerate(read(OUT/'pairs.json')):
            all_records = []
            for c in (p['parent_profile'], p['profiles'][0]):
                rr = {r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                      (pair['state_uid'], pair['eta_uid'], c['controller_uid']))}
                rr = [rr[canonical({'future_index':j})] for j in range(16)]
                assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr)
                all_records.append(rr)
            matched = [(a,b) for a,b in zip(*all_records) if not a['numerical_failure'] and not b['numerical_failure']]
            a = np.array([x['success'] for x,y in matched])
            b = np.array([y['success'] for x,y in matched])
            rescue, broken = int(((a==0)&(b==1)).sum()), int(((a==1)&(b==0)).sum())
            rows.append(dict(pair_index=i, state_uid=pair['state_uid'], eta_index=pair['eta_index'],
                valid_matched_seeds=len(matched), parent_Q=float(a.mean()), motion_Q=float(b.mean()),
                Q_delta=float((b-a).mean()), seed_rescue=rescue, seed_break=broken,
                parent_success=sum(r['success'] for r in all_records[0] if not r['numerical_failure']),
                motion_success=sum(r['success'] for r in all_records[1] if not r['numerical_failure']),
                parent_numerical=sum(r['numerical_failure'] for r in all_records[0]),
                motion_numerical=sum(r['numerical_failure'] for r in all_records[1]),
                p=float(binomtest(rescue,rescue+broken,.5).pvalue) if rescue+broken else 1.))
    pp = np.array([r['p'] for r in rows])
    order = np.argsort(pp)
    adj = np.empty(len(pp))
    adj[order] = np.minimum(1, np.maximum.accumulate(pp[order]*(len(pp)-np.arange(len(pp)))))
    for r, value in zip(rows, adj):
        r['Holm_p'] = float(value)
    raw_all = [json.loads(line)['record'] for file in (DBROOT/'journals'/EXP).glob('*.jsonl') for line in file.read_text().splitlines()]
    raw = [r for r in raw_all if r['controller_uid'] == p['profiles'][0]['controller_uid']]
    assert len(raw)==256
    assert len({(r['state_uid'],canonical(r['eta']),r['future_index']) for r in raw})==256
    active = [r for r in raw if r['motion_active_steps']]
    proof = read(OUT/'input_identity_proof.json')
    assert all(proof[k] for k in ('current_h_same','initial_H20_H80_same','entity_response_same','goal_response_same'))
    result = dict(cells=len(rows), new_continuations=len(raw_all), new_motion_continuations=len(raw), matched_parent_evidence_reused=p.get('matched_parent_evidence_reused',256),
        parent_preexisting_numerical=sum(r['parent_numerical'] for r in rows),
        new_numerical=sum(r['motion_numerical'] for r in rows),
        parent_B15=sum(r['parent_success']>=15 for r in rows), motion_B15=sum(r['motion_success']>=15 for r in rows),
        Holm_significant_cells=int((adj<.05).sum()), mean_absolute_Q_change=float(np.mean([abs(r['Q_delta']) for r in rows])),
        active_rollouts=len(active), first_active_range=[min(r['motion_first_active_step'] for r in active),max(r['motion_first_active_step'] for r in active)] if active else None,
        all_current_input_groups_identical=True, labels_enter_training=False,
        verdict='CURRENT_CONTEXT_MISSES_OUTCOME_RELEVANT_MOTION_RESPONSE' if (adj<.05).any() else 'NO_SIGNIFICANT_COUNTEREXAMPLE_IN_THIS_PANEL',
        scope=p.get('analysis_scope','Exploratory same-source-family stationary motion-gated program test, not natural neural-checkpoint alias or cross-scene result. Requires independent confirmation before a broad mechanism claim.'))
    csvwrite(OUT/'matched_Q_alias_cells.csv',rows)
    write(OUT/'adjudication.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()

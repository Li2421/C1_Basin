"""Paired B15/Q16 comparison: native generator sample-0 versus donor-state sample-0."""
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from shared_rollout_db.src.rollout_db import eta_identity

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def evidence(con,sid,eid,cid):
    rr=con.execute('''SELECT seed_key,success,numerical_failure,collision FROM rollout
      WHERE state_uid=? AND eta_uid=? AND controller_uid=?
      AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0''',(sid,eid,cid)).fetchall()
    by={json.loads(r['seed_key'])['future_index']:r for r in rr if 'future_index' in json.loads(r['seed_key'])}
    assert all(i in by for i in range(16))
    records=[by[i] for i in range(16)]
    assert not any(r['numerical_failure'] for r in records)
    s=sum(r['success'] for r in records)
    return {'k':s,'n':16,'q':s/16,'B15':s>=15,'collision':sum(r['collision'] for r in records)}


def main():
    summary=json.loads((OUT/'cache_postflight_generator_swap.json').read_text())['summary']
    assert summary['total_requested']==summary['exact_reusable']==1072
    assert not summary['genuinely_missing'] and not summary['incompatible']
    state={int(s['episode_index']):s for s in json.loads((OUT/'frozen_proposals.json').read_text())['states']}
    with (OUT/'candidate_cache_keys.csv').open() as f:
        keys={(int(r['episode_index']),r['kind']):r for r in csv.DictReader(f)}
    old=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/candidate_cache_keys.csv'
    with old.open() as f:
        oldkeys={(int(r['episode_index']),r['kind']):r for r in csv.DictReader(f)}
    original={int(r['episode_index']):r for r in csv.DictReader((ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/per_state_k.csv').open()) if r['K']=='1'}
    c=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row
    rows=[]
    for ep,z in state.items():
        donor=keys[ep,'donor_sample_0'];native=oldkeys[ep,'sample_0']
        assert donor['state_uid']==native['state_uid']
        d=evidence(c,donor['state_uid'],donor['eta_uid'],donor['controller_uid'])
        n=evidence(c,native['state_uid'],native['eta_uid'],native['controller_uid'])
        assert abs(n['q']-float(original[ep]['critic_Q16']))<1e-10
        dist=float(np.linalg.norm((np.asarray(z['eta']['sample_0'])-np.asarray(z['eta']['donor_sample_0']))/np.array([.625,.5,.375])))
        rows.append({'episode_index':ep,'recipient_source_group':z['source_group'],
                     'donor_episode_index':(ep+73)%200,'donor_eta_uid':donor['eta_uid'],
                     'native_eta_uid':native['eta_uid'],'eta_normalized_distance':dist,
                     'native_Q16':n['q'],'native_B15':n['B15'],'donor_Q16':d['q'],'donor_B15':d['B15'],
                     'fixed_common_Q16':float(original[ep]['fixed_common_Q16']),
                     'fixed_common_B15':float(original[ep]['fixed_common_Q16'])>=15/16,
                     'native_collision':n['collision'],'donor_collision':d['collision']})
    c.close()
    with (OUT/'generator_swap_paired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def pair(a,b):
        a=np.array(a,bool);b=np.array(b,bool);wins=int(np.sum(a&~b));loss=int(np.sum(~a&b))
        return {'a_B15':int(a.sum()),'b_B15':int(b.sum()),'a_rescue':wins,'a_break':loss,
                'paired_exact_p':float(binomtest(wins,wins+loss,.5).pvalue) if wins+loss else 1.}
    a=np.array([r['native_Q16'] for r in rows]);b=np.array([r['donor_Q16'] for r in rows]);d=a-b
    rng=np.random.default_rng(9703);boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
    out={'n':len(rows),'native_vs_donor':pair([r['native_B15'] for r in rows],[r['donor_B15'] for r in rows]),
         'native_vs_fixed':pair([r['native_B15'] for r in rows],[r['fixed_common_B15'] for r in rows]),
         'donor_vs_fixed':pair([r['donor_B15'] for r in rows],[r['fixed_common_B15'] for r in rows]),
         'native_mean_Q16':float(a.mean()),'donor_mean_Q16':float(b.mean()),
         'native_minus_donor_mean_Q16':float(d.mean()),
         'native_minus_donor_Q16_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist(),
         'eta_distance_median_normalized':float(np.median([r['eta_normalized_distance'] for r in rows])),
         'new_rollouts':1072,'postflight':summary,
         'interpretation_scope':'Frozen already-seen Toy states, newly evaluated cross-state eta combinations; empirical generator marginal versus state-conditioned sample. No new independent source-family generalization claim.'}
    (OUT/'generator_swap_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()

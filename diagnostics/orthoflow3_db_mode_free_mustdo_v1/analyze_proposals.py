#!/usr/bin/env python3
"""Frozen DB K4/K16 proposal oracle versus frozen DB-adapted critic."""
import csv,json,sqlite3,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
C=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);C.row_factory=sqlite3.Row
F=json.loads((HERE/'db_frozen_proposals.json').read_text())
M={r['state_id']:r for r in csv.DictReader((HERE/'db_mean_per_state.csv').open())}
CTL=json.loads((ROOT/'diagnostics/orthoflow3_pipeline_dataset_evidence_audit_v1/db_hard_frozen_generator_mean.json').read_text())['hard_panel_controller_uid']
keys=[json.dumps({'future_index':i},sort_keys=True,separators=(',',':')) for i in range(16)]
def get(uid,eta):
    euid=eta_identity(eta)[0]
    rr=C.execute('''SELECT seed_key,success,deadlock,timeout,collision,numerical_failure,j_def,episode_length
       FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND compatibility_quality='EXACT_REUSE'
       AND conflict_quarantined=0''',(uid,euid,CTL)).fetchall()
    d={r['seed_key']:dict(r) for r in rr}
    if any(k not in d for k in keys):raise RuntimeError(f'Incomplete Q16 {uid} {euid}')
    return [d[k] for k in keys]
def metrics(rr):
    return {'Q16':sum(r['success'] for r in rr)/16,'B15':int(sum(r['success'] for r in rr)>=15),
      'collision':sum(r['collision'] for r in rr),'numerical_failure':sum(r['numerical_failure'] for r in rr),
      'J_def':float(np.mean([r['j_def'] for r in rr if r['j_def'] is not None]))}
def writecsv(name,rr):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)))
        w.writeheader();w.writerows(rr)
def main():
    details=[];per=[]
    for state in F['states']:
        sid=state['state_id'];uid=state['state_uid'];q=[]
        for j,eta in enumerate(state['eta']):
            m=metrics(get(uid,eta));q.append(m)
            details.append({'state_id':sid,'sample_index':j,'eta':json.dumps(eta),
                           'critic_score':state['critic_scores'][j],**m})
        scores=np.asarray(state['critic_scores']);qs=np.asarray([m['Q16'] for m in q])
        for K in (4,16):
            oracle=int(np.argmax(qs[:K]));critic=int(np.argmax(scores[:K]))
            mm=M[sid]
            per.append({'state_id':sid,'source_group':state['source_group'],'K':K,
              'oracle_index':oracle,'oracle_B15':q[oracle]['B15'],'oracle_Q16':q[oracle]['Q16'],
              'critic_index':critic,'critic_B15':q[critic]['B15'],'critic_Q16':q[critic]['Q16'],
              'coverage_B15':int(any(x['B15'] for x in q[:K])),
              'feasible_proposals':sum(x['B15'] for x in q[:K]),
              'critic_gap_Q16':q[oracle]['Q16']-q[critic]['Q16'],
              'mean_B15':int(mm['mean_B15']),'mean_Q16':float(mm['mean_Q16']),
              'fixed_B15':int(mm['fixed_B15']),'fixed_Q16':float(mm['fixed_Q16'])})
    writecsv('db_proposal_q16.csv',details);writecsv('db_k4_k16_per_state.csv',per)
    summary={}
    for K in (4,16):
        rr=[r for r in per if r['K']==K]
        summary[str(K)]={'states':len(rr),'oracle_B15':sum(r['oracle_B15'] for r in rr),
          'critic_B15':sum(r['critic_B15'] for r in rr),'coverage_B15':sum(r['coverage_B15'] for r in rr),
          'mean_oracle_Q16':float(np.mean([r['oracle_Q16'] for r in rr])),
          'mean_critic_Q16':float(np.mean([r['critic_Q16'] for r in rr])),
          'mean_gap_Q16':float(np.mean([r['critic_gap_Q16'] for r in rr])),
          'critic_vs_mean_rescue':sum(r['critic_B15'] and not r['mean_B15'] for r in rr),
          'critic_vs_mean_break':sum(r['mean_B15'] and not r['critic_B15'] for r in rr),
          'critic_vs_fixed_rescue':sum(r['critic_B15'] and not r['fixed_B15'] for r in rr),
          'critic_vs_fixed_break':sum(r['fixed_B15'] and not r['critic_B15'] for r in rr)}
    result={'critic_provenance':F['critic'],'DB_adapted':True,'critic_checkpoints':F['critic_checkpoints'],
            'split':'hard source groups disjoint from DB critic train/val/test','K':summary}
    (HERE/'db_k4_k16_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()

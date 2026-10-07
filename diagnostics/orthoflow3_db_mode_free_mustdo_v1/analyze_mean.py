#!/usr/bin/env python3
"""Paired DB hard-panel mean versus same-seed cached Safety/fixed."""
import csv,json,sqlite3,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
AUD=ROOT/'diagnostics/orthoflow3_pipeline_dataset_evidence_audit_v1'
CTL='ctl_ceef00016f37f243923d0c7568f7b017db6854059447fd2daaa7d797b1d5258b'
C=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);C.row_factory=sqlite3.Row
F=json.loads((AUD/'db_hard_frozen_generator_mean.json').read_text())['states']
f=next(r for r in csv.DictReader((ROOT/'diagnostics/orthoflow3_db_shared_mode_transfer_v1/train_mode_counts.csv').open()) if r['mode_id']=='0')
fixed=C.execute('SELECT eta_uid FROM eta WHERE eta1=? AND eta2=? AND eta3=?',tuple(float(f[k]) for k in ('eta1','eta2','eta3'))).fetchone()[0]
safety=C.execute('SELECT eta_uid FROM eta WHERE eta1=0 AND eta2=0 AND eta3=0').fetchone()[0]
from shared_rollout_db.src.rollout_db import eta_identity
def get(uid,euid):
    rr=C.execute('''SELECT seed_key,success,deadlock,timeout,collision,numerical_failure,j_def,episode_length
      FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE' ''',(uid,euid,CTL)).fetchall()
    d={r['seed_key']:dict(r) for r in rr}
    keys=[json.dumps({'future_index':i},sort_keys=True,separators=(',',':')) for i in range(16)]
    if any(k not in d for k in keys):raise RuntimeError(f'missing standard16 {uid} {euid}')
    return [d[k] for k in keys]
def summarize(rr):
    a=np.asarray([r['success'] for r in rr]);return {'Q16':float(a.mean()),'B15':int(a.sum()>=15),
      'successes':int(a.sum()),'collision':int(sum(r['collision'] for r in rr)),
      'numerical_failure':int(sum(r['numerical_failure'] for r in rr)),
      'deadlock':int(sum(r['deadlock'] for r in rr)),'timeout':int(sum(r['timeout'] for r in rr)),
      'J_def':float(np.mean([r['j_def'] for r in rr if r['j_def'] is not None])),
      'episode_length':float(np.mean([r['episode_length'] for r in rr]))}
def main():
    state_rows=[];matched=[]
    for row in F:
        uid=row['state_uid'];raw={name:get(uid,e) for name,e in [('safety',safety),('fixed',fixed),('mean',eta_identity(row['eta'])[0])]}
        sr={name:summarize(rr) for name,rr in raw.items()}
        state_rows.append({'state_id':row['state_id'],'state_uid':uid,'source_group':row['source_group'],
            **{f'{name}_{k}':v for name,m in sr.items() for k,v in m.items()}})
        for i in range(16):matched.append({name:int(raw[name][i]['success']) for name in raw})
    fields=list(state_rows[0])
    with (HERE/'db_mean_per_state.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(state_rows)
    agg={}
    for name in ('safety','fixed','mean'):
        agg[name]={'B15_states':sum(r[f'{name}_B15'] for r in state_rows),'N_states':len(state_rows),
          'mean_Q16':float(np.mean([r[f'{name}_Q16'] for r in state_rows])),
          **{k:sum(r[f'{name}_{k}'] for r in state_rows) for k in ('successes','collision','numerical_failure','deadlock','timeout')},
          'J_def_mean':float(np.mean([r[f'{name}_J_def'] for r in state_rows])),
          'episode_length_mean':float(np.mean([r[f'{name}_episode_length'] for r in state_rows]))}
    paired={}
    for base in ('safety','fixed'):
        paired['mean_vs_'+base]={'state_B15_rescue':sum(r['mean_B15'] and not r[f'{base}_B15'] for r in state_rows),
          'state_B15_break':sum(r[f'{base}_B15'] and not r['mean_B15'] for r in state_rows),
          'seed_rescue':sum(r['mean'] and not r[base] for r in matched),
          'seed_break':sum(r[base] and not r['mean'] for r in matched)}
    result={'methods':agg,'paired':paired,'matched_seeds_per_state':16,'source_group_unique':len({r['source_group'] for r in state_rows})==48,
      'hard_panel_source_overlap_generator_train':0,'controller_uid':CTL,'NEW_ROLLOUT':768}
    (HERE/'db_mean_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()

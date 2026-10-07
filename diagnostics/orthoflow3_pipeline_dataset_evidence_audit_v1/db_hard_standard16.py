#!/usr/bin/env python3
"""Use the hard panel's own controller/RNG fingerprint; read-only."""
import csv,json,sqlite3
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');OUT=Path(__file__).resolve().parent
CTL='ctl_ceef00016f37f243923d0c7568f7b017db6854059447fd2daaa7d797b1d5258b'
con=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
fixed=next(r for r in csv.DictReader((ROOT/'diagnostics/orthoflow3_db_shared_mode_transfer_v1/train_mode_counts.csv').open()) if r['mode_id']=='0')
fixed_eta=con.execute('select eta_uid from eta where eta1=? and eta2=? and eta3=?',tuple(float(fixed[k]) for k in ('eta1','eta2','eta3'))).fetchone()[0]
safety_eta=con.execute('select eta_uid from eta where eta1=0 and eta2=0 and eta3=0').fetchone()[0]
seeds={json.dumps({'future_index':i},separators=(',',':'),sort_keys=True) for i in range(16)}
by=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
for g,e,sd,k in con.execute('select s.source_group,r.eta_uid,r.seed_key,r.success from rollout r join state s using(state_uid) where r.controller_uid=? and r.conflict_quarantined=0 and r.numerical_failure=0',(CTL,)):
    if sd in seeds:by[g][e][sd].add(k)
rows=[]
for g,etas in sorted(by.items()):
    q={e:sum(next(iter(v)) for v in ss.values()) for e,ss in etas.items() if len(ss)==16 and all(len(v)==1 for v in ss.values())}
    f=q.get(fixed_eta);s=q.get(safety_eta)
    rows.append({'source_group':g,'fixed_B15':int(f>=15) if f is not None else '',
        'safety_B15':int(s>=15) if s is not None else '',
        'fixed_successes16':f if f is not None else '', 'safety_successes16':s if s is not None else '',
        'robust_eta_count':sum(x>=15 for x in q.values()),'tested_eta_count':len(q)})
with (OUT/'db_hard_standard16_classes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summ={'hard_groups':len(rows),'fixed_B15':sum(x['fixed_B15']==1 for x in rows),
      'safety_B15':sum(x['safety_B15']==1 for x in rows),
      'correction_needed':sum(x['safety_B15']==0 and x['robust_eta_count']>0 for x in rows),
      'fixed_fail_adaptive_success':sum(x['fixed_B15']==0 and x['robust_eta_count']>0 for x in rows)}
(OUT/'db_hard_standard16_summary.json').write_text(json.dumps(summ,indent=2,sort_keys=True)+'\n')
print(json.dumps(summ))

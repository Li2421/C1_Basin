#!/usr/bin/env python3
"""Exact standard-seed DB state classification under frozen controller UID."""
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
CTL='ctl_0ce9b25aa22d23cd3d07a01c61fd72f3c76cec4be3c053a3d4b8fbb6184f543d'
con=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
fixed=next(r for r in csv.DictReader((ROOT/'diagnostics/orthoflow3_db_shared_mode_transfer_v1/train_mode_counts.csv').open())
           if r['mode_id']=='0')
fixedeta=con.execute('SELECT eta_uid FROM eta WHERE eta1=? AND eta2=? AND eta3=?',
    tuple(float(fixed[x]) for x in ('eta1','eta2','eta3'))).fetchone()[0]
safetyeta=con.execute('SELECT eta_uid FROM eta WHERE eta1=0 AND eta2=0 AND eta3=0').fetchone()[0]
seedkeys={json.dumps({'future_index':i},separators=(',',':'),sort_keys=True) for i in range(16)}
by=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
for group,eta,seed,k in con.execute('''SELECT s.source_group,r.eta_uid,r.seed_key,r.success
    FROM rollout r JOIN state s USING(state_uid)
    WHERE r.controller_uid=? AND r.conflict_quarantined=0 AND r.numerical_failure=0''',(CTL,)):
    if seed in seedkeys:by[group][eta][seed].add(k)
rows=[]
for group,etas in sorted(by.items()):
    p={}
    conflicts=0
    for eta,seeds in etas.items():
        if any(len(v)>1 for v in seeds.values()):conflicts+=1;continue
        if len(seeds)==16:p[eta]=sum(next(iter(v)) for v in seeds.values())
    fq=p.get(fixedeta);sq=p.get(safetyeta)
    robust=sum(k>=15 for k in p.values())
    rows.append({'source_group':group,'exact_B15_tested_eta':len(p),'robust_eta_count':robust,
        'fixed_standard16_successes':fq if fq is not None else '',
        'fixed_B15':int(fq>=15) if fq is not None else '',
        'safety_standard16_successes':sq if sq is not None else '',
        'safety_B15':int(sq>=15) if sq is not None else '',
        'fixed_fail_adaptive_success':int(fq is not None and fq<15 and robust>0),
        'correction_needed':int(sq is not None and sq<15 and robust>0),
        'narrow_proxy_lt_0p1':int(len(p)>=8 and robust/len(p)<.1),
        'no_known_basin':int(len(p)>=8 and robust==0),
        'cross_uid_seed_conflict_eta':conflicts})
with (OUT/'db_global_state_classes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={'groups_any_evidence':len(rows),'groups_fixed_standard16':sum(x['fixed_B15']!='' for x in rows),
    'groups_fixed_B15':sum(x['fixed_B15']==1 for x in rows),
    'groups_fixed_fail_adaptive_success':sum(x['fixed_fail_adaptive_success'] for x in rows),
    'groups_safety_standard16':sum(x['safety_B15']!='' for x in rows),
    'groups_safety_B15':sum(x['safety_B15']==1 for x in rows),
    'groups_correction_needed':sum(x['correction_needed'] for x in rows),
    'groups_narrow_proxy':sum(x['narrow_proxy_lt_0p1'] for x in rows),
    'groups_no_known_basin':sum(x['no_known_basin'] for x in rows),
    'groups_any_cross_uid_conflict':sum(x['cross_uid_seed_conflict_eta']>0 for x in rows)}
(OUT/'db_global_state_classes.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary))

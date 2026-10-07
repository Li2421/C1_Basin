#!/usr/bin/env python3
"""Deterministic 25-label-per-scenario reproducibility replay (no DB writes)."""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow.parquet as pq

from new_benchmark_common.basin_dataset_v1 import (
    TrainingRuntime, DoubleTrainingRuntime, double_runtime_states, sampled_state_rows,
)


ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
V2=ROOT/'datasets/orthoflow3_basin_dataset_v2_audited'
DB=ROOT/'shared_rollout_db/rollout.sqlite'
SCENARIOS=('double_bottleneck','four_way_intersection','ring_exchange')


def dump(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def parse(x):return json.loads(x) if isinstance(x,str) else x
def rank(row):return hashlib.sha256(f"hygiene-replay-v1|{row['scenario']}|{row['state_uid']}|{row['eta_uid']}".encode()).hexdigest()


def freeze():
    labels=pq.read_table(V2/'eta_labels.parquet').to_pylist();selected=[]
    for sc in SCENARIOS:
        rows=[r for r in labels if r['scenario']==sc and r['robust_15of16'] is not None]
        strata={
            'eta_zero':[r for r in rows if parse(r['eta_raw'])==[0,0,0]],
            'robust':[r for r in rows if r['robust_15of16'] is True and parse(r['eta_raw'])!=[0,0,0]],
            'boundary':[r for r in rows if r['geometry_membership']=='boundary_candidate'],
            'certified_negative':[r for r in rows if r['negative_evidence_class'] in ('CERTIFIED_NON_ROBUST','FULL_Q16_NON_ROBUST')],
        }
        chosen=[];seen=set()
        for name,n in [('eta_zero',5),('robust',10),('boundary',3),('certified_negative',7)]:
            for r in sorted(strata[name],key=rank):
                key=(r['state_uid'],r['eta_uid'])
                if key in seen:continue
                chosen.append((r,name));seen.add(key)
                if sum(x[1]==name for x in chosen)>=n:break
        for r in sorted(rows,key=rank):
            if len(chosen)>=25:break
            key=(r['state_uid'],r['eta_uid'])
            if key not in seen:chosen.append((r,'hash_fill'));seen.add(key)
        if len(chosen)!=25:raise RuntimeError((sc,len(chosen)))
        for r,stratum in chosen:
            selected.append({'scenario':sc,'state_uid':r['state_uid'],'state_id':r['state_id'],
                'eta_uid':r['eta_uid'],'eta':parse(r['eta_raw']),'controller_uid':r['controller_uid'],
                'stratum':stratum,'seed_outcomes':parse(r['seed_outcomes'])})
    payload={'protocol':'SHA256(hygiene-replay-v1|scenario|state_uid|eta_uid), fixed strata 5/10/3/7 then hash fill',
             'created_before_replay':True,'counts':{sc:sum(r['scenario']==sc for r in selected) for sc in SCENARIOS},'labels':selected}
    dump('replay_selection_manifest.json',payload);return payload


def expected_db(label):
    by={}
    with sqlite3.connect(DB) as con:
        con.row_factory=sqlite3.Row
        for seed in label['seed_outcomes']:
            row=con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(seed['rollout_uid'],)).fetchone()
            if row is None:raise RuntimeError(seed['rollout_uid'])
            idx=int(json.loads(row['seed_key'])['future_index']);by[idx]=dict(row)
    return by


def runtime_for(sc,label_controller):
    if sc=='double_bottleneck':
        states=double_runtime_states();groups=defaultdict(list)
        for s in states:groups[s['controller_uid']].append(s)
        return DoubleTrainingRuntime(groups[label_controller],label_controller),{s['uid']:s for s in groups[label_controller]}
    states=sampled_state_rows(sc);rt=TrainingRuntime(sc,states,parent=False)
    return rt,{s['uid']:s for s in states}


def run(sc):
    manifest=json.loads((HERE/'replay_selection_manifest.json').read_text());labels=[r for r in manifest['labels'] if r['scenario']==sc]
    runtimes={};records=[]
    for label in labels:
        ctl=label['controller_uid']
        if ctl not in runtimes:runtimes[ctl]=runtime_for(sc,ctl)
        rt,states=runtimes[ctl];expected=expected_db(label);actual={}
        for seed in sorted(expected):
            row=rt.rollout(states[label['state_uid']],np.asarray(label['eta'],float),seed,'orthoflow3')
            actual[seed]=row
        exp_success=sum(r['success'] for r in expected.values());act_success=sum(r['success'] for r in actual.values())
        records.append({'scenario':sc,'state_uid':label['state_uid'],'eta_uid':label['eta_uid'],'stratum':label['stratum'],
            'seeds':sorted(expected),'exact_outcome_agreement':sum(bool(actual[k]['success'])==bool(expected[k]['success']) and bool(actual[k]['numerical_failure'])==bool(expected[k]['numerical_failure']) for k in expected),
            'terminal_reason_agreement':sum(str(actual[k]['outcome'])==str(expected[k]['outcome']) for k in expected),
            'collision_agreement':sum(bool(actual[k]['collision'])==bool(expected[k]['collision']) for k in expected),
            'seed_count':len(expected),'expected_successes':int(exp_success),'actual_successes':int(act_success),
            'Q_agreement':exp_success==act_success})
    dump(f'replay_{sc}.json',{'scenario':sc,'labels':records});print(json.dumps({'scenario':sc,'labels':len(records),'seeds':sum(r['seed_count'] for r in records)},indent=2))


def finalize():
    rows=[]
    for sc in SCENARIOS:rows.extend(json.loads((HERE/f'replay_{sc}.json').read_text())['labels'])
    result={'labels':len(rows),'seed_replays':sum(r['seed_count'] for r in rows),
        'exact_seed_outcome_agreement':sum(r['exact_outcome_agreement'] for r in rows),
        'terminal_reason_agreement':sum(r['terminal_reason_agreement'] for r in rows),
        'collision_agreement':sum(r['collision_agreement'] for r in rows),
        'Q_exact_label_agreement':sum(r['Q_agreement'] for r in rows),
        'total_seed_comparisons':sum(r['seed_count'] for r in rows),'records':rows}
    dump('reproducibility_replay_results.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run','finalize'));p.add_argument('--scenario',choices=SCENARIOS);a=p.parse_args()
    out=freeze() if a.stage=='freeze' else run(a.scenario) if a.stage=='run' else finalize()
    if out is not None and a.stage!='run':print(json.dumps({k:v for k,v in out.items() if k!='labels' and k!='records'},indent=2))
if __name__=='__main__':main()

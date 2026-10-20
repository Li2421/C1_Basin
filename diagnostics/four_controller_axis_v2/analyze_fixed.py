"""Preregistered Four fixed-arm census and headroom decision."""
from __future__ import annotations
import csv,datetime,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import connect
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    assert read(HERE/'R5_FREEZE.receipt.json')['features_sha256']==sha(HERE/'R5_FROZEN.npz')
    assert read(HERE/'FIXED_MERGE.receipt.json')['experiment_uid']=='exp_round19b_four_controller_axis_fixed_v1'
    p=read(HERE/'FIXED_PROTOCOL.json');cases=read(HERE/'FIXED_CASES.json');assert len(cases)==4800
    Y=np.zeros((25,3,32,2),dtype=np.int8);U=np.zeros_like(Y);names=p['controller_order'];arms=p['arms'];unknown_records=[]
    with connect(True) as db:
        for c in cases:
            r=read(HERE/'stage2/raw'/f"{c['rollout_uid']}.json")
            assert r['case']==c and r['protocol_sha256']==sha(HERE/'FIXED_PROTOCOL.json') and r['trace_sha256']==sha(r['trace_path'])
            stored=db.execute('SELECT success,numerical_failure,outcome,episode_length FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert stored and tuple(stored)==(r['success'],r['numerical_failure'],r['outcome'],r['episode_length'])
            i=names.index(c['base_controller_name']);j=arms.index(c['arm']);Y[i,j,c['state_index'],c['seed']]=r['success'];U[i,j,c['state_index'],c['seed']]=r['numerical_failure']
            if r['numerical_failure']:unknown_records.append(dict(rollout_uid=c['rollout_uid'],controller=names[i],arm=c['arm'],state_index=c['state_index'],seed=c['seed'],error=r['error']))
    post=preflight(HERE/'FIXED_PLANNED_ROLLOUTS.json')
    assert post['summary']['ambiguous']==0 and post['summary']['genuinely_missing']==len(unknown_records)
    write(HERE/'FIXED_POSTFLIGHT.json',dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        official_planner='shared_rollout_db.src.planner.preflight',summary=post['summary'],unknown_not_rerun=len(unknown_records)))
    rows=[]
    for i,name in enumerate(names):
        counts=Y[i].sum((1,2));unk=U[i].sum((1,2));known=64-unk
        rates=np.divide(counts,known,out=np.full(3,np.nan),where=known>0)
        best=int(np.nanargmax(rates)) if np.isfinite(rates).any() else None
        rows.append(dict(controller=name,run=p['profiles'][name]['run'],flavor=p['profiles'][name]['flavor'],
            success_counts=counts.tolist(),unknown_counts=unk.tolist(),known_counts=known.tolist(),
            success_rates=rates.tolist(),best_arm=arms[best] if best is not None else None))
    available=[r['best_arm'] for r in rows if r['best_arm'] is not None];majority=max(arms,key=lambda arm:(available.count(arm),-arms.index(arm)))
    a=[r['controller'] for r in rows if r['best_arm']!=majority and r['best_arm'] is not None and
        r['success_rates'][arms.index(r['best_arm'])]-r['success_rates'][arms.index(majority)]>=.10]
    b=[r['controller'] for r in rows if r['best_arm'] is not None and r['success_rates'][arms.index(r['best_arm'])]<=.85]
    gate= len(a)>=3 or len(b)>=3
    result=dict(stage='fixed_census',status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_rollouts=4800,unknown=len(unknown_records),rates_definition='successes / known seeds; numerical failures unknown excluded',
        arms=arms,majority_arm=majority,condition_a_qualifying=a,condition_b_qualifying=b,
        condition_a_passed=len(a)>=3,condition_b_passed=len(b)>=3,headroom_gate_passed=gate,
        table=rows,r5_freeze_sha256=sha(HERE/'R5_FREEZE.receipt.json'),postflight_sha256=sha(HERE/'FIXED_POSTFLIGHT.json'))
    write(HERE/'FIXED_RESULTS.json',result);write(HERE/'FIXED_UNKNOWN.json',unknown_records)
    with (HERE/'FIXED_CENSUS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['controller','run','flavor','arm','success','known','unknown','rate'])
        for r in rows:
            for j,arm in enumerate(arms):w.writerow([r['controller'],r['run'],r['flavor'],arm,r['success_counts'][j],r['known_counts'][j],r['unknown_counts'][j],r['success_rates'][j]])
    print(json.dumps(dict(status='COMPLETE',headroom_gate_passed=gate,majority_arm=majority,
        condition_a_count=len(a),condition_b_count=len(b),unknown=len(unknown_records))),flush=True)
if __name__=='__main__':main()

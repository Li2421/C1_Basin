"""Summarize the prospectively frozen evaluation; never alter its selection."""
import json
import hashlib
import csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/c1_frozen_unseen_64'

def outcome(r):
    if r is None:return 'unavailable'
    if r['controller_error']:return 'controller_error'
    o=r['outcome']
    if o['collision']:return 'collision'
    if o['success']:return 'success'
    return 'deadlock' if o['deadlock'] else 'timeout'

def performance(rows):
    valid=[r for r in rows if r is not None]
    st=[r['outcome']['stagnation_seconds'] for r in valid]
    return dict(outcomes={k:sum(outcome(r)==k for r in rows) for k in ['success','deadlock','timeout','collision','controller_error','unavailable']},
        any_deadlock_events=sum(r['outcome']['deadlock'] for r in valid),
        mean_stagnation_seconds=float(np.mean(st)),median_stagnation_seconds=float(np.median(st)),
        total_stagnation_seconds=float(np.sum(st)),
        mean_future_stagnation_seconds=float(np.mean([r['outcome']['future_stagnation_seconds'] for r in valid])))

def main():
    p=json.loads((OUT/'protocol.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in p['hashes'].items())
    states=[json.loads((OUT/'states'/f'{rid}.json').read_text()) for rid in p['ids']]
    rows=[];failures=[]
    for s in states:
        rr=[json.loads(f.read_text()) for f in (OUT/'traces'/str(s['rid'])).glob('*.json')]
        assert len(rr)==161 and {r['cid'] for r in rr}=={c['cid'] for c in p['library']}
        ranked=sorted([r for r in rr if r['costs'] is not None],key=lambda r:(r['costs']['score'],r['cid']))
        assert s['selected']==(ranked[0] if ranked else None)
        assert s['successful_candidates']==sum(r['outcome']['success'] for r in rr)
        if not s['selected'] or not s['selected']['outcome']['success']:
            good=[r for r in ranked if r['outcome']['success']]
            failures.append(dict(rid=s['rid'],coverage=s['coverage'],coverage_unknown=s['coverage_unknown'],
                successful_candidates=s['successful_candidates'],best_success_rank=s['best_success_rank'],
                selected=s['selected'],best_success=good[0] if good else None))
        rows.extend(rr)
    covered=sum(s['coverage'] for s in states)
    hit=sum(s['coverage'] and s['selected'] is not None and s['selected']['outcome']['success'] for s in states)
    safety=dict(counts={k:sum(r['safety'][k] for r in rows) for k in states[0]['safety_counts']},
        minima={k:min(r['safety'][k] for r in rows) for k in states[0]['safety_minima']},
        maximum_speed=max(r['safety']['max_speed'] for r in rows),
        maximum_integration_error=max(r['safety']['integration_error'] for r in rows))
    summary=dict(states=len(states),branches=len(rows),coverage_states=covered,
        coverage_unknown=sum(s['coverage_unknown'] for s in states),conditional_ranking_hits=hit,
        conditional_ranking_accuracy=hit/covered if covered else None,
        original9_coverage=sum(s['original9_coverage'] for s in states),
        selected=performance([s['selected'] for s in states]),baseline_zero=performance([s['baseline'] for s in states]),
        original9_selected=performance([s['original9_selected'] for s in states]),
        all_candidates=performance(rows),successful_candidates=sum(r['outcome']['success'] for r in rows),
        unscorable=sum(r['costs'] is None for r in rows),geometry_error_frames=sum(len(r['geometry_errors']) for r in rows),
        ambiguous_geometry_frames=sum(r['ambiguous_geometry_frames'] for r in rows),
        controller_errors=sum(r['controller_error'] is not None for r in rows),
        numerical_counts={k:sum(r['numerical_counts'][k] for r in rows) for k in states[0]['numerical_counts']},
        safety=safety,failures=failures,manifest_verified=True)
    selected_rows=[s['selected'] for s in states if s['selected'] is not None]
    summary['selected_safety_minima']={k:min(r['safety'][k] for r in selected_rows) for k in states[0]['safety_minima']}
    summary['coverage_candidate_counts']={k:float(fn([s['successful_candidates'] for s in states])) for k,fn in [('min',np.min),('median',np.median),('max',np.max)]}
    table=[]
    for s in states:
        r=s['selected']
        table.append(dict(rid=s['rid'],successful_candidates=s['successful_candidates'],best_success_rank=s['best_success_rank'],
            selected_cid=r['cid'] if r else None,selected_score=r['costs']['score'] if r else None,
            selected_outcome=outcome(r),selected_stagnation=r['outcome']['stagnation_seconds'] if r else None,
            baseline_outcome=outcome(s['baseline']),baseline_stagnation=s['baseline']['outcome']['stagnation_seconds'],
            original9_coverage=s['original9_coverage'],original9_outcome=outcome(s['original9_selected']),
            unscorable=s['unscorable'],controller_errors=s['controller_errors']))
    with (OUT/'per_state.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='failures'},indent=2))

if __name__=='__main__':main()

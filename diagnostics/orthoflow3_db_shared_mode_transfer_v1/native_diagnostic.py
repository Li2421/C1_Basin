#!/usr/bin/env python3
"""Conditional DB-native oracle diagnostic using only frozen DB development clusters."""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

H = Path(__file__).parent

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def make():
    split = json.load(open(H/'db_state_split.json'))
    test = [s for s in split['states'] if s['split'] == 'test']
    # These are the 12 deterministic B63 cluster representatives from the
    # quarantined transform-development panel; no main TEST outcome is used.
    eta = json.load(open(H/'toy_to_db_transform.json'))['candidates']['affine_plus_mode_residual']['eta']
    tasks = []
    for s in test:
        for m, e in enumerate(eta):
            for fi in range(64):
                tasks.append(dict(state_id=s['state_id'], eta=e, future_index=fi,
                                  phase='db_native_diagnostic',
                                  probe_id=f"{s['state_id']}_native{m}_{fi}",
                                  controller='db_native', mode_id=m, split='test'))
    groups = defaultdict(list)
    for t in tasks: groups[(t['controller'], t['mode_id'])].append(t)
    load = [0]*6; owner = {}
    for g, rows in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        j = int(np.argmin(load)); owner[g] = j; load[j] += len(rows)
    out = H/'plans'/'native_diag'; out.mkdir(parents=True, exist_ok=True)
    for j in range(6):
        with (out/f'shard{j}.jsonl').open('w') as f:
            for t in tasks:
                if owner[(t['controller'],t['mode_id'])] == j:
                    f.write(json.dumps(t)+'\n')
    (H/'db_native_manifest.json').write_text(json.dumps({
        'conditional': True,
        'codebook_source': '12 deterministic B63 cluster representatives from quarantined DB transform-development states',
        'test_outcomes_used_for_construction': False,
        'states': len(test), 'modes': 12, 'continuations': len(tasks), 'shard_loads': load
    }, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'tasks':len(tasks),'loads':load}, indent=2))

def aggregate():
    split = json.load(open(H/'db_state_split.json'))
    testids = [s['state_id'] for s in split['states'] if s['split']=='test']
    records = []
    for p in (H/'raw').glob('native_diag_*.jsonl'):
        records.extend(json.loads(x) for x in open(p) if x.strip())
    by = defaultdict(dict)
    for r in records:
        by[(r['state_id'],int(r['mode_id']))][int(r['future_index'])] = r
    rows=[]; oracle=[]
    for sid in testids:
        state=[]
        for m in range(12):
            v=[by[(sid,m)][i] for i in range(64)]
            k=sum(bool(r['success']) for r in v); out=Counter(str(r['outcome']) for r in v)
            q=dict(state_id=sid,mode_id=m,successes=k,trials=64,empirical_Q=k/64,
                   B63=k>=63,deadlock=sum(out[x] for x in ('strict_deadlock','safe_deadlock','deadlock')),
                   timeout=out['timeout'],collision=sum('collision' in str(r['outcome']) for r in v))
            rows.append(q); state.append(q)
        oracle.append(max(state,key=lambda r:(r['empirical_Q'],-r['mode_id'])))
    write_csv(H/'db_native_test_q64.csv',rows)
    result={'B63_states':sum(r['B63'] for r in oracle),'states':len(oracle),
            'B63_coverage':sum(r['B63'] for r in oracle)/len(oracle),
            'mean_best_Q64':float(np.mean([r['empirical_Q'] for r in oracle])),
            'new_continuations':len(records)}
    (H/'db_native_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    {'make':make,'aggregate':aggregate}[sys.argv[1]]()

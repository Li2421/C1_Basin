#!/usr/bin/env python3
"""Deterministically repartition only unfinished repair tasks by mode group."""
import json
from collections import defaultdict
from pathlib import Path
H=Path(__file__).parent
src=H/'plans'/'matrix_repair'/'shard0.jsonl'
tasks=[json.loads(x) for x in open(src) if x.strip()]
known=set()
for p in (H/'raw').glob('*.jsonl'):
    for line in open(p):
        r=json.loads(line)
        known.add((r['state_id'],r['controller'],r.get('mode_id'),int(r['future_index'])))
missing=[t for t in tasks if (t['state_id'],t['controller'],t.get('mode_id'),int(t['future_index'])) not in known]
groups=defaultdict(list)
for t in missing:groups[(t['controller'],t['mode_id'])].append(t)
load=[0]*4; owner={}
for g,v in sorted(groups.items(),key=lambda x:(-len(x[1]),x[0])):
    j=min(range(4),key=lambda z:(load[z],z));owner[g]=j;load[j]+=len(v)
out=H/'plans'/'matrix_repair_reshard';out.mkdir(parents=True,exist_ok=True)
for j in range(4):
    with open(out/f'shard{j}.jsonl','w') as f:
        for t in missing:
            if owner[(t['controller'],t['mode_id'])]==j:f.write(json.dumps(t)+'\n')
(H/'repair_reshard_manifest.json').write_text(json.dumps({
    'scientific_change':False,'source_plan_tasks':len(tasks),'already_completed_keys':len(tasks)-len(missing),
    'remaining_tasks':len(missing),'shard_loads':load,
    'rule':'fixed (controller,mode) group per shard; deterministic greedy load balance'
},indent=2,sort_keys=True)+'\n')
print(json.dumps({'remaining':len(missing),'loads':load},indent=2))

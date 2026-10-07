#!/usr/bin/env python3
"""Balance unfinished frozen fresh-evaluation tasks over six CPU shards."""
import json
from collections import defaultdict
from pathlib import Path
H=Path(__file__).parent
tasks=[]
for p in sorted((H/'plans'/'fresh_eval').glob('shard*.jsonl')):
    tasks.extend(json.loads(x) for x in open(p) if x.strip())
known=set()
for p in (H/'raw').glob('*.jsonl'):
    for line in open(p):
        r=json.loads(line);known.add((r['state_id'],r['controller'],r.get('mode_id'),int(r['future_index'])))
missing=[t for t in tasks if (t['state_id'],t['controller'],t.get('mode_id'),int(t['future_index'])) not in known]
# Deterministic round-robin within each eta/controller group avoids long tails
# while keeping each shard close to balanced.  It changes scheduling only.
groups=defaultdict(list)
for t in missing:groups[(t['controller'],t['mode_id'])].append(t)
bins=[[] for _ in range(6)]
for g,v in sorted(groups.items()):
    for t in sorted(v,key=lambda x:(x['state_id'],int(x['future_index']))):
        j=min(range(6),key=lambda z:(len(bins[z]),z));bins[j].append(t)
out=H/'plans'/'fresh_eval_balanced';out.mkdir(parents=True,exist_ok=True)
for j,b in enumerate(bins):
    with open(out/f'shard{j}.jsonl','w') as f:
        for t in b:f.write(json.dumps(t)+'\n')
(H/'fresh_reshard_manifest.json').write_text(json.dumps({'scientific_change':False,'total_frozen_tasks':len(tasks),'completed_before_reshard':len(tasks)-len(missing),'remaining':len(missing),'loads':[len(b) for b in bins]},indent=2,sort_keys=True)+'\n')
print(json.dumps({'remaining':len(missing),'loads':[len(b) for b in bins]},indent=2))

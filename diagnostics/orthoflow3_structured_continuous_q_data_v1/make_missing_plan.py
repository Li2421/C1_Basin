#!/usr/bin/env python3
import json,hashlib
from collections import defaultdict
from pathlib import Path
H=Path(__file__).parent
req=json.load(open(H/'request_index.json'));pf=json.load(open(H/'cache_preflight.json'))['details'];assert len(req)==len(pf)
tasks=[]
for meta,detail in zip(req,pf):
 assert meta['state_uid']==detail['state_uid'] and meta['eta_uid']==detail['eta_uid']
 for sk in detail['missing_seeds']:
  fi=int(json.loads(sk)['future_index'])
  tasks.append({**meta,'future_index':fi,'phase':'structured_continuous_q_v1','controller':'structured_'+meta['probe_id']})
groups=defaultdict(list)
for t in tasks:groups[(t['state_id'],t['eta_uid'])].append(t)
load=[0]*6;owner={}
for g,v in sorted(groups.items(),key=lambda q:(-len(q[1]),hashlib.sha256(('|'.join(q[0])).encode()).hexdigest())):
 j=min(range(6),key=lambda x:(load[x],x));owner[g]=j;load[j]+=len(v)
p=H/'plans';p.mkdir(exist_ok=True)
for j in range(6):
 with (p/f'shard{j}.jsonl').open('w') as f:
  for t in tasks:
   if owner[(t['state_id'],t['eta_uid'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
json.dump({'shards':6,'missing_continuations':len(tasks),'shard_load':load,'pair_groups':len(groups)},open(H/'missing_plan_summary.json','w'),indent=2,sort_keys=True)
print(json.dumps({'missing':len(tasks),'load':load,'groups':len(groups)},indent=2))

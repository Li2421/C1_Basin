#!/usr/bin/env python3
"""Merge frozen task metadata onto cache-reused evaluation rows by exact task order."""
import argparse,json
from pathlib import Path
HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_true_t0_point_learning_v1')
ap=argparse.ArgumentParser();ap.add_argument('plan');a=ap.parse_args()
for shard in range(6):
 tasks=[json.loads(x) for x in open(HERE/'plans'/a.plan/f'shard{shard}.jsonl') if x.strip()]
 p=HERE/'raw'/a.plan/f'shard{shard}.jsonl';rows=[json.loads(x) for x in open(p) if x.strip()]
 if len(tasks)!=len(rows):raise RuntimeError((shard,len(tasks),len(rows)))
 fixed=[]
 for task,row in zip(tasks,rows,strict=True):
  if row['state_id']!=task['state_id'] or int(row['future_index'])!=int(task['future_index']):raise RuntimeError(('order',shard))
  # Eta equality is mandatory; metadata from the frozen task is authoritative.
  if any(abs(float(x)-float(y))>1e-12 for x,y in zip(row['eta'],task['eta'])):raise RuntimeError(('eta',shard))
  fixed.append({**row,**task})
 with open(p,'w') as f:
  for row in fixed:f.write(json.dumps(row,sort_keys=True)+'\n')
print(json.dumps({'plan':a.plan,'shards':6,'status':'metadata_repaired_no_rollout'}))

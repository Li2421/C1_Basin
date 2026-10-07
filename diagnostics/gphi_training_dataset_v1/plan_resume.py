"""Shard interrupted candidate arms while retaining every completed tuple."""
import json
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent

rows=[];seen=set()
for path in (HERE/'candidate_reuse.jsonl', HERE/'raw/candidate_pre_deadlock/records.jsonl', HERE/'raw/candidate_recovery/records.jsonl'):
    for line in path.read_text().splitlines():
        row=json.loads(line);key=(row['state_id'],tuple(row['eta']),int(row['seed']))
        if key in seen:raise AssertionError(('duplicate resume tuple',key))
        seen.add(key);rows.append(row)
(HERE/'candidate_resume_reuse.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))

summary={}
for category,nshards in (('pre_deadlock',4),('recovery',3)):
    arms=json.loads((HERE/f'candidate_{category}_arms.json').read_text())
    by_state=defaultdict(list)
    for arm in arms:by_state[arm['state_id']].append(arm)
    shards=[[] for _ in range(nshards)];loads=[0]*nshards
    for state,state_arms in sorted(by_state.items(),key=lambda item:(-len(item[1]),item[0])):
        index=min(range(nshards),key=lambda i:(loads[i],i));shards[index].extend(state_arms);loads[index]+=len(state_arms)
    for index,shard in enumerate(shards):
        (HERE/f'candidate_{category}_resume_{index}_arms.json').write_text(json.dumps(shard,indent=2)+'\n')
    summary[category]={'states':len(by_state),'arms':len(arms),'shard_arm_counts':loads}
print(json.dumps({'reused_completed_tuples':len(rows),'shards':summary},indent=2))

"""Frozen task groups, longest estimated work first; no scientific choices here."""
from common import *

def build(stage):
    folder=HERE/'QUEUES'/stage
    if(folder/'INDEX.json').exists():return read(folder/'INDEX.json')
    if stage.startswith('train_'):
        tasks=read(HERE/('TRAIN_INNER_TASKS.json'if stage=='train_inner'else'TRAIN_OUTER_TASKS.json'))
        tasks.sort(key=lambda t:(-t['source_scene_count'],t['fold'],t['mode'],t['seed']))
        units=[(t,60.)for t in tasks]
    else:
        tasks=read(HERE/f'{stage}_MISSING_CASES.json');units=[]
        for c in tasks:
            kind=c['controller_payload']['nominal']['type_code'];n=c['state']['scene']
            # Plan A* requests individually so a long request cannot block a
            # different request behind the same group owner.
            cost=600. if kind=='prioritized_astar'else 150. if kind=='conflict_based_search'and n.startswith('xsc_')else 60. if n=='lit_circle_five'else 5.
            units.append((c,cost))
        units.sort(key=lambda r:(-r[1],r[0]['rollout_uid']))
    groups=[];part=[];seconds=0
    for task,cost in units:
        if part and seconds+cost>600:groups.append((part,seconds));part=[];seconds=0
        part.append(task);seconds+=cost
    if part:groups.append((part,seconds))
    rows=[]
    for i,(part,seconds)in enumerate(groups):
        p=folder/'groups'/f'{i:06d}.json';write(p,part);rows.append(dict(group=f'{i:06d}',path=str(p),sha256=sha(p),count=len(part),estimated_seconds=seconds))
    rows.sort(key=lambda r:(-r['estimated_seconds'],r['group']))
    write(folder/'INDEX.json',rows);write(folder/'QUEUE.receipt.json',dict(utc=utc(),stage=stage,requests=len(units),groups=len(rows),index_sha256=sha(folder/'INDEX.json')))
    return rows

def complete(stage):
    return all((HERE/'QUEUES'/stage/'done'/e['group']).exists()for e in read(HERE/'QUEUES'/stage/'INDEX.json'))

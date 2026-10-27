"""Cost-order groups, with independent UID leases to use every free worker."""
from common import *
def build(stage):
    folder=HERE/'QUEUES'/stage
    if(folder/'INDEX.json').exists():return read(folder/'INDEX.json')
    tasks=read(HERE/f'{stage}_MISSING_CASES.json');tasks.sort(key=lambda c:(-c['scenario_payload']['config']['max_steps']*c['state']['N'],c['rollout_uid']))
    rows=[]
    for i in range(0,len(tasks),100):
        part=tasks[i:i+100];p=folder/'groups'/f'{i//100:06d}.json';write(p,part);rows.append(dict(group=f'{i//100:06d}',path=str(p),sha256=sha(p),count=len(part),estimated_seconds=6*len(part)))
    write(folder/'INDEX.json',rows);write(folder/'QUEUE.receipt.json',dict(stage=stage,index_sha256=sha(folder/'INDEX.json'),groups=len(rows),requests=len(tasks)))
    return rows
def complete(stage):return all((HERE/'QUEUES'/stage/'done'/r['group']).exists()for r in read(HERE/'QUEUES'/stage/'INDEX.json'))

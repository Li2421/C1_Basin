"""Immutable training/task groups ordered by predicted cost, about ten minutes."""
from common import *

def training_tasks(stage):
    out=[]
    for f in read(HERE/'FOLDS.json'):
        if stage=='train_generator':
            if not f['generator']:continue
            kinds=[('generator','generator')]
        else:kinds=[('critic',mode)for mode in f['modes']]
        for kind,mode in kinds:
            for seed in (17,23,41):out.append(dict(fold=f['name'],kind=kind,mode=mode,seed=seed,source_scene_count=len(f['sources'])))
    return out

def build(stage):
    folder=HERE/'QUEUES'/stage
    if(folder/'INDEX.json').exists():return read(folder/'INDEX.json')
    if stage.startswith('train_'):
        tasks=training_tasks(stage);tasks.sort(key=lambda t:(-t['source_scene_count'],t['fold'],t['mode'],t['seed']))
        units=[(t,max(60.,30*t['source_scene_count']))for t in tasks]
    else:
        rows=read(HERE/f'{stage}_MISSING_CASES.json');units=[]
        timings=read(HERE/'PILOT_TIMINGS.json')['per_scene_controller']if(HERE/'PILOT_TIMINGS.json').exists()else{}
        for c in rows:
            name=c.get('pipeline_scene',c['state']['scene']);kind=c['controller_payload']['nominal']['type_code'];obs=timings.get(name+'|'+kind)
            cost=float(obs['mean_seconds'])if obs else max(1.,c['scenario_payload']['config']['max_steps']*.003*c['state']['N']*(3 if kind=='conflict_based_search'else 1.5))
            units.append((c,cost))
        units.sort(key=lambda row:(-row[1],row[0]['rollout_uid']))
    groups=[];part=[];seconds=0.
    for task,cost in units:
        if part and seconds+cost>600:groups.append((part,seconds));part=[];seconds=0.
        part.append(task);seconds+=cost
    if part:groups.append((part,seconds))
    index=[]
    for i,(part,seconds)in enumerate(groups):
        p=folder/'groups'/f'{i:06d}.json';write(p,part)
        index.append(dict(group=f'{i:06d}',path=str(p),sha256=sha(p),count=len(part),estimated_seconds=seconds))
    index.sort(key=lambda row:(-row['estimated_seconds'],row['group']))
    write(folder/'INDEX.json',index);write(folder/'QUEUE.receipt.json',dict(utc=utc(),stage=stage,requests=len(units),groups=len(groups),
        index_sha256=sha(folder/'INDEX.json'),costs_are_queue_estimates_not_pilot_measurements=True))
    return index

def complete(stage):
    return all((HERE/'QUEUES'/stage/'done'/e['group']).exists()for e in read(HERE/'QUEUES'/stage/'INDEX.json'))

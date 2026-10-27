"""Only preflight-missing requests; immutable groups ordered by estimated wall cost."""
from common import *
import numpy as np


@lru_cache(None)
def measured_costs():
    p=HERE/'PILOT_TIMINGS.json'
    return read(p).get('per_scene_controller',{})if p.exists()else{}


def cost(c):
    s=c['state'];cfg=c['scenario_payload']['config'];kind=c['controller_payload']['nominal']['type_code']
    factor={'cross_entropy':1.4,'distributed_cross_entropy':2.,'gradient_mpc':1.8,
            'conflict_based_search':3.,'prioritized_astar':1.5}.get(kind,1.)
    observed=measured_costs().get(s['scene']+'|'+kind)
    if observed:return max(1.,observed['mean_seconds'])
    return max(1.,cfg['max_steps']*.003*max(1,s['N'])*factor)


def budget(c):
    kind=c['controller_payload']['nominal']['type_code'];scene=c['state']['scene']
    observed=measured_costs().get(scene+'|'+kind)
    if observed:return max(cost(c),1.2*observed['max_seconds'])
    return 2*cost(c)


def build(stage):
    folder=HERE/'QUEUES'/stage
    if (folder/'INDEX.json').exists():return read(folder/'INDEX.json')
    rows=read(HERE/f'{stage}_MISSING_CASES.json')
    units={}
    for c in rows:
        deterministic=c['controller_payload']['nominal']['type_code']in ('gradient_mpc','prioritized_astar','conflict_based_search')
        key=(c['state_uid'],c['eta_uid'],c['controller_uid']) if deterministic else(c['rollout_uid'],)
        units.setdefault(key,[]).append(c)
    batches=list(units.values())
    for unit in batches:unit.sort(key=lambda c:c['seed'])
    batches.sort(key=lambda unit:(-cost(unit[0]),unit[0]['rollout_uid']))
    groups=[];part=[];seconds=0.
    for unit in batches:
        estimate=cost(unit[0]) if len(unit)>1 else sum(cost(c)for c in unit)
        if part and seconds+estimate>600.:
            groups.append((part,seconds));part=[];seconds=0.
        part.extend(unit);seconds+=estimate
    if part:groups.append((part,seconds))
    index=[]
    for i,(part,estimate)in enumerate(groups):
        p=folder/'groups'/f'{i:06d}.json';write(p,part)
        # Preserve seed0->seed1 deterministic units while avoiding allocating a
        # known long episode too near the Slurm end time.
        seconds=sum(budget(c)for c in part if c['seed']==0 or c['controller_payload']['nominal']['type_code']not in ('gradient_mpc','prioritized_astar','conflict_based_search'))
        index.append(dict(group=f'{i:06d}',path=str(p),sha256=sha(p),count=len(part),estimated_seconds=estimate,budget_seconds=seconds))
    index.sort(key=lambda row:(-row['estimated_seconds'],row['group']))
    write(folder/'INDEX.json',index)
    write(folder/'QUEUE.receipt.json',dict(utc=utc(),stage=stage,requests=len(rows),groups=len(index),
        preflight_sha256=sha(HERE/f'{stage}_PREFLIGHT.receipt.json'),index_sha256=sha(folder/'INDEX.json')))
    return index


def complete(stage):
    folder=HERE/'QUEUES'/stage
    return all((folder/'done'/row['group']).exists()for row in read(folder/'INDEX.json'))

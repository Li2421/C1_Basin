"""Never submit additional work while global limits are exhausted."""
from pathlib import Path
import datetime
import json
import os
import subprocess

HERE=Path(__file__).resolve().parent

def queue():
    output=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],
                                    '-o','%i|%j|%T|%C|%b|%P|%r'],text=True)
    rows=[]
    cpu=0;gpu=0;pending_cpu=0;pending_gpu=0
    for line in output.splitlines():
        job,name,state,cpus,gres,partition,reason=line.split('|')
        rows.append(dict(job=job,name=name,state=state,cpus=int(cpus),gres=gres,partition=partition,reason=reason))
        running=state in ('RUNNING','COMPLETING','CONFIGURING')
        reserved=state=='PENDING' and name.startswith(('crosstype1_','crosstype1b_')) and reason not in ('JobHeldUser','JobHeldAdmin')
        if not running and not reserved:
            continue
        if running:cpu+=int(cpus)
        else:pending_cpu+=int(cpus)
        if 'shard:' in gres or 'gpu:' in gres:
            # On this workstation, shard is the actual fractional GPU resource.
            shards=[int(v.split(':')[-1]) for v in gres.split(',') if 'shard:' in v]
            allocated=sum(shards) if shards else 8*sum(int(v.split(':')[-1]) for v in gres.split(',') if 'gpu:' in v)
            if running:gpu+=allocated
            else:pending_gpu+=allocated
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_shards=cpu,
                gpu_shards=gpu,reserved_cpu_shards=cpu+pending_cpu,reserved_gpu_shards=gpu+pending_gpu,rows=rows,raw=output)


def cpu_capacity():
    path=HERE/'RESOURCE_LIMITS.json'
    limits=json.loads(path.read_text()) if path.exists() else dict(cpu_shards=18,gpu_shards=2)
    capacity=limits['cpu_shards'];other=[]
    if limits.get('cpu_24_only_without_other_tasks'):
        output=subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%j|%T|%C|%r'],text=True)
        for line in output.splitlines():
            job,user,name,state,cpus,reason=line.split('|')
            own=user==os.environ['USER'] and name.startswith('crosstype1b_')
            if not own and state in ('RUNNING','COMPLETING','CONFIGURING','PENDING') and reason not in ('JobHeldUser','JobHeldAdmin'):
                other.append(dict(job=job,user=user,name=name,state=state,cpus=int(cpus)))
        if other:capacity=min(capacity,18)
    return capacity,other


def require(cpu=0,gpu=0):
    snapshot=queue()
    limits_path=HERE/'RESOURCE_LIMITS.json'
    limits=json.loads(limits_path.read_text()) if limits_path.exists() else dict(cpu_shards=18,gpu_shards=2)
    capacity,other=cpu_capacity()
    snapshot.update(requested_cpu=cpu,requested_gpu=gpu,
                    cpu_limit=capacity,gpu_limit=limits['gpu_shards'],other_tasks_for_conditional_cpu_limit=other,
                    admitted=snapshot['reserved_cpu_shards']+cpu<=capacity and snapshot['reserved_gpu_shards']+gpu<=limits['gpu_shards'])
    with (HERE/'RESOURCE_CHECKS.jsonl').open('a') as stream:
        stream.write(json.dumps(snapshot)+'\n')
    if not snapshot['admitted']:
        raise RuntimeError(f"Resources unavailable: {snapshot['cpu_shards']} CPU, {snapshot['gpu_shards']} GPU")
    return snapshot

if __name__=='__main__':
    print(json.dumps(queue(),indent=2))

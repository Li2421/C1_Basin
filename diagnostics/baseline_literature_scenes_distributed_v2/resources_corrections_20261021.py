"""User-wide quota, including the explicitly scheduled resource release."""
from experiment import *
import re
import subprocess

CPU_LIMIT = 24
GPU_SHARD_LIMIT = 5
LOCAL_CPU_RESERVE = 1
POLICY = HERE/'RESOURCE_SCHEDULE_CORRECTED.json'

def limits(at=None):
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError('Resource time must include a timezone')
    policy=read(POLICY)
    reduced=at >= datetime.fromisoformat(policy['release_at_utc'])
    selected=policy['after'] if reduced else policy['before']
    return dict(cpu_limit=selected['total_cpu_shares'],
                gpu_shard_limit=selected['total_gpu_shards'],
                release_at_utc=policy['release_at_utc'],reduced=reduced)

def snapshot():
    raw = subprocess.check_output([
        'squeue','-u',os.environ['USER'],'-h','-r','-o','%i|%j|%T|%C|%b'
    ]).decode()
    jobs=[]
    for line in raw.splitlines():
        job,name,state,cpu,gres=line.split('|')
        shard=re.search(r'(?:gres[/:])?shard[:=](\d+)',gres)
        gpu=re.search(r'(?:gres[/:])?gpu(?::[^:|,]+)?:([0-9]+)',gres)
        shares=int(shard.group(1)) if shard else 8*int(gpu.group(1)) if gpu else 0
        jobs.append(dict(job_id=job,name=name,state=state,cpus=int(cpu),gpu_shards=shares,gres=gres))
    quota=limits()
    cpu_limit=quota['cpu_limit'];gpu_limit=quota['gpu_shard_limit']
    used_cpu=sum(r['cpus'] for r in jobs)
    used_gpu=sum(r['gpu_shards'] for r in jobs)
    return dict(utc=utc(),jobs=jobs,scheduled_cpus=used_cpu,gpu_shards=used_gpu,
                local_cpu_reserve=LOCAL_CPU_RESERVE,
                **quota,
                free_cpu_shares=max(0,cpu_limit-LOCAL_CPU_RESERVE-used_cpu),
                free_gpu_shards=max(0,gpu_limit-used_gpu))

def require_capacity(cpus=1,gpu_shards=0):
    state=snapshot()
    assert cpus<=state['free_cpu_shares'] and gpu_shards<=state['free_gpu_shards'],state
    return state

if __name__=='__main__':
    print(json.dumps(snapshot(),ensure_ascii=False,indent=2))

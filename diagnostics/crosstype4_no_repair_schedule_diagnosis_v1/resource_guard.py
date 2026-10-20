"""Global per-user18 CPU/2 GPU shard admission, checked at every submission."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json
HERE=Path(__file__).resolve().parent
CPU_LIMIT=18;GPU_LIMIT=2
def utc():return datetime.now(timezone.utc).isoformat()
def queue():
    raw=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%b|%r'],text=True);rows=[];cpu=gpu=0
    for line in raw.splitlines():
        job,name,state,c,gres,reason=line.split('|');gres=gres.replace('gres/','');c=int(c);g=sum(int(s.split(':')[-1]) for s in gres.split(',') if s.startswith('shard:'))
        if not g:g=8*sum(int(s.split(':')[-1]) for s in gres.split(',') if s.startswith('gpu:'))
        active=state in ('RUNNING','COMPLETING','CONFIGURING') or (state=='PENDING' and name.startswith('crosstype4_') and reason not in ('JobHeldUser','JobHeldAdmin'))
        rows.append(dict(job=job,name=name,state=state,cpu=c,gpu=g,reason=reason,counts=active))
        if active:cpu+=c;gpu+=g
    return dict(utc=utc(),cpu_shards=cpu,gpu_shards=gpu,rows=rows,raw=raw)
def require(cpu=0,gpu=0):
    r=queue();r.update(requested_cpu=cpu,requested_gpu=gpu,cpu_limit=CPU_LIMIT,gpu_limit=GPU_LIMIT,admitted=r['cpu_shards']+cpu<=CPU_LIMIT and r['gpu_shards']+gpu<=GPU_LIMIT)
    with (HERE/'RESOURCE_CHECKS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    if not r['admitted']:raise RuntimeError('Resources unavailable: '+json.dumps(r))
    return r
def drain():return False

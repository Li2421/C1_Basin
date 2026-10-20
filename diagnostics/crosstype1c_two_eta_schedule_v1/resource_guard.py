"""Latest user limits and the explicit 09:00 resource return."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json
HERE=Path(__file__).resolve().parent
DEADLINE=datetime.fromisoformat('2026-10-19T09:00:00+08:00').timestamp()
def utc():return datetime.now(timezone.utc).isoformat()
def limits():return (24,8) if datetime.now(timezone.utc).timestamp()<DEADLINE else (18,4)
def queue():
    raw=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%b|%r'],text=True)
    rows=[];cpu=gpu=0
    for line in raw.splitlines():
        job,name,state,c,gres,reason=line.split('|');c=int(c)
        shards=sum(int(x.split(':')[-1]) for x in gres.split(',') if x.startswith('shard:'))
        if not shards:shards=8*sum(int(x.split(':')[-1]) for x in gres.split(',') if x.startswith('gpu:'))
        active=state in ('RUNNING','COMPLETING','CONFIGURING') or (state=='PENDING' and name.startswith('crosstype1c_') and reason not in ('JobHeldUser','JobHeldAdmin'))
        rows.append(dict(job=job,name=name,state=state,cpu=c,gpu=shards,reason=reason,counts=active))
        if active:cpu+=c;gpu+=shards
    return dict(utc=utc(),cpu_shards=cpu,gpu_shards=gpu,rows=rows,raw=raw)
def require(cpu=0,gpu=0):
    r=queue();c,g=limits();r.update(requested_cpu=cpu,requested_gpu=gpu,cpu_limit=c,gpu_limit=g,admitted=r['cpu_shards']+cpu<=c and r['gpu_shards']+gpu<=g)
    with (HERE/'RESOURCE_CHECKS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    if not r['admitted']:raise RuntimeError('Resources unavailable: '+json.dumps(r))
    return r
def drain():
    return os.environ.get('CROSSTYPE_TEMPORARY')=='1' and datetime.now(timezone.utc).timestamp()>=DEADLINE-600

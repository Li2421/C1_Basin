"""Global per-user24 CPU/5 GPU shard admission, checked at every submission."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json
HERE=Path(__file__).resolve().parent
CPU_LIMIT=18;GPU_LIMIT=2
def limits(now=None):
    p=HERE/'RESOURCE_RELEASE_POLICY.json'
    if not p.exists():return CPU_LIMIT,GPU_LIMIT
    policy=json.loads(p.read_text());now=now or datetime.now(timezone.utc)
    cap=policy['after'] if now>=datetime.fromisoformat(policy['deadline_utc']) else policy['before']
    return cap['cpu'],cap['gpu']
def utc():return datetime.now(timezone.utc).isoformat()
def queue():
    raw=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%b|%r'],text=True);rows=[];cpu=gpu=0
    for line in raw.splitlines():
        job,name,state,c,gres,reason=line.split('|');gres=gres.replace('gres/','');c=int(c);g=sum(int(s.split(':')[-1]) for s in gres.split(',') if s.startswith('shard:'))
        if not g:g=8*sum(int(s.split(':')[-1]) for s in gres.split(',') if s.startswith('gpu:'))
        active=state in ('RUNNING','COMPLETING','CONFIGURING') or (state=='PENDING' and name.startswith(('crosstype5_','crosstype6_')) and reason not in ('JobHeldUser','JobHeldAdmin'))
        rows.append(dict(job=job,name=name,state=state,cpu=c,gpu=g,reason=reason,counts=active))
        if active:cpu+=c;gpu+=g
    return dict(utc=utc(),cpu_shards=cpu,gpu_shards=gpu,rows=rows,raw=raw)
def require(cpu=0,gpu=0):
    cpu_limit,gpu_limit=limits()
    r=queue();r.update(requested_cpu=cpu,requested_gpu=gpu,cpu_limit=cpu_limit,gpu_limit=gpu_limit,admitted=r['cpu_shards']+cpu<=cpu_limit and r['gpu_shards']+gpu<=gpu_limit)
    with (HERE/'RESOURCE_CHECKS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    if not r['admitted']:raise RuntimeError('Resources unavailable: '+json.dumps(r))
    return r
def drain():
    # Operational stop only, between complete episodes/fits; never change physics.
    if os.environ.get('CROSSTYPE_TEMPORARY_RESOURCE')!='1':return False
    p=HERE/'RESOURCE_RELEASE_POLICY.json'
    if not p.exists():return False
    policy=json.loads(p.read_text());deadline=datetime.fromisoformat(policy['deadline_utc'])
    return (deadline-datetime.now(timezone.utc)).total_seconds()<=policy['graceful_lead_seconds']

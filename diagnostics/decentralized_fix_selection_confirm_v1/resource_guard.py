"""Best-effort CPU throttling of this round only; never cancel or restart work.

Scheduler suspension is not permitted for this account. STOP/CONT preserve all
simulation state, RNG state, and journals. They do not release Slurm allocations,
so allocation-based cap violations remain failures in the original audit.
"""
from common import *
import subprocess,time
G=guard()
paused=set()
def log(value):
 with (HERE/'RESOURCE_THROTTLE.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),**value))+'\n')
def signal(job,sig):
 r=subprocess.run(['scancel','--signal='+sig,'--full',job],capture_output=True,text=True)
 log(dict(job=job,signal=sig,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 return r.returncode==0
try:
 while True:
  raw=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True)
  rows=[s.split('|') for s in raw.splitlines()]
  own={q[0] for q in rows if q[1]=='r30i_main' and q[2] in ['RUNNING','COMPLETING','CONFIGURING','STOPPED']}
  paused=(paused & own)|{q[0] for q in rows if q[0] in own and q[2]=='STOPPED'}
  if not own:break
  external=sum(int(q[3]) for q in rows if q[1]!='r30i_main' and q[2] in ['RUNNING','COMPLETING','CONFIGURING'] and 'shard' not in q[4])
  count=min(len(own),max(0,len(own)+external-18))
  desired=set(sorted(own,key=lambda j:int(j.rsplit('_',1)[1]),reverse=True)[:count])
  for job in sorted(desired-paused):
   if signal(job,'STOP'):paused.add(job)
  for job in sorted(paused-desired):
   if signal(job,'CONT'):paused.remove(job)
  time.sleep(5)
finally:
 for job in sorted(paused):signal(job,'CONT')
 log(dict(finished=True,guard=guard()))

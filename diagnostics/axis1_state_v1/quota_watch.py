"""Maintain total user CPU shard occupancy at the authorized target18."""
from common import *
import subprocess,time,re

def queue():
 s=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True)
 return [x.split('|') for x in s.splitlines()]
def allowance(rows,job):
 other=[x for x in rows if not x[0].startswith(job+'_') and x[0]!=job and not x[1].startswith('r32_finalize')]
 running=sum(int(x[3]) for x in other if x[2] in ['RUNNING','COMPLETING','CONFIGURING'])
 pending=sum(int(x[3]) for x in other if x[2]=='PENDING')
 # Reserve known next-stage jobs while their preceding workers drain.
 return max(0,18-max(running,pending)),running,pending
if __name__=='__main__':
 job=read(HERE/'SPEEDUP_SUBMISSION.json')['job'];last=None
 while True:
  rows=queue();own=[x for x in rows if x[0].startswith(job+'_') or x[0]==job]
  if not own:break
  n,other,pending=allowance(rows,job)
  if n and n!=last:
   result=subprocess.run(['scontrol','update',f'JobId={job}',f'ArrayTaskThrottle={n}'],capture_output=True,text=True)
   if result.returncode:print(result.stderr,flush=True)
   else:last=n;print(json.dumps(dict(utc=utc(),throttle=n,other_running=other,other_pending_reserved=pending)),flush=True)
  running=sum(int(x[3]) for x in rows if x[2] in ['RUNNING','COMPLETING','CONFIGURING']);write(HERE/'CPU_OCCUPANCY.json',dict(utc=utc(),total_user_CPU_running=running,target=18,helper_array=job,helper_CPU_running=sum(int(x[3]) for x in own if x[2]=='RUNNING'),array_throttle=last,other_CPU_running=other,queue=rows))
  time.sleep(5)
 print('Helper array ended; finalization is dependency-controlled',flush=True)

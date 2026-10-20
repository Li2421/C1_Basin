"""Transfer one own active CPU to existing expert queue without exceeding18 actual."""
from common import *
import subprocess,signal,time
before=read(HERE/'RESOURCE_STATUS.json');paused=set(before['paused_groups']);ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pgid=,stat=,args='],text=True);own={}
for line in ps.splitlines():
 t=line.strip().split(None,2)
 if len(t)==3 and str(HERE/'run.py')+' pilot ' in t[2] and 'python -B' in t[2] and 'bash -c' not in t[2]:own[int(t[0])]=t[1]
active=sorted(k for k,v in own.items() if 'T' not in v and k not in paused);assert active
chosen=active[0];os.killpg(chosen,signal.SIGSTOP);paused.add(chosen)
try:
 subprocess.run(['scontrol','update','JobId=22054','ArrayTaskThrottle=1'],check=True);subprocess.run(['scontrol','release','22054'],check=True)
 for i in range(30):
  q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);rows=[x.split('|') for x in q.splitlines()];rr=[x for x in rows if x[2] in ['RUNNING','COMPLETING','CONFIGURING']]
  if any(x[1]=='r36_expert' and x[2]=='RUNNING' for x in rr):break
  time.sleep(.5)
 else:raise RuntimeError('Expert did not start; restore own CPU transfer')
 allocated=sum(int(x[3]) for x in rr);external=sum(int(x[3]) for x in rr if not x[1].startswith('r36_'));events=before['events']+[dict(utc=utc(),signal='STOP',pgid=chosen,allocated_CPU=allocated,reason='Own CPU handoff: stop pilot before releasing already-preregistered expert; no extra actual CPU')]
 write(HERE/'RESOURCE_STATUS.json',dict(utc=utc(),allocated_CPU=allocated,external_CPU=external,paused_groups=sorted(paused),active_CPU=allocated-len(paused),events=events,policy='Total actual18, never signal external tasks; no GPU used'))
 write(HERE/'EXPERT_CPU_HANDOFF.json',dict(utc=utc(),chosen_own_pilot_group=chosen,preserved_inflight=True,expert_job='22054',queue_after=q,actual_CPU_after=allocated-len(paused),source_runtime_changed=False,order='Stop own active pilot first, then start expert within same actual CPU budget; persistent monitor resumes own pilots as slots free.'))
 print('Expert owns transferred CPU; preserved own pilot group',chosen,'actual total',allocated-len(paused),flush=True)
except BaseException:
 subprocess.run(['scontrol','hold','22054']);os.killpg(chosen,signal.SIGCONT);raise

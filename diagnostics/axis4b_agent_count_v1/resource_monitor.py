"""Fill18 total user CPU shards and preserve own in-flight states on contention."""
from core import *
import subprocess,signal,time
previous=read(HERE/'RESOURCE_STATUS.json') if (HERE/'RESOURCE_STATUS.json').exists() else {};paused=set(previous.get('paused_groups',[]));events=previous.get('events',[])
while True:
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);rows=[s.split('|') for s in q.splitlines()];own=[s for s in rows if s[1].startswith('r36_')];running=[s for s in rows if s[2] in ['RUNNING','COMPLETING','CONFIGURING']];allocated=sum(int(s[3]) for s in running);external=sum(int(s[3]) for s in running if not s[1].startswith('r36_'));budget=max(0,18-external)
 for base in {s[0].split('_')[0] for s in own}:
  if budget:
   subprocess.run(['scontrol','update','JobId='+base,'ArrayTaskThrottle='+str(budget)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);subprocess.run(['scontrol','release',base],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  else:subprocess.run(['scontrol','hold',base],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 groups=set();ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,pgid=,args='],text=True)
 for line in ps.splitlines():
  parts=line.strip().split(None,2)
  if len(parts)==3 and str(HERE)+'/' in parts[2] and any('/'+x+' ' in parts[2] for x in ['run.py','run_derive.py','analysis_worker.py','analysis_worker_v2.py']) and 'python -B' in parts[2] and 'bash -c' not in parts[2]:groups.add(int(parts[1]))
 paused&=groups;needed=max(0,allocated-18)
 while len(paused)<needed:
  candidates=sorted(groups-paused,reverse=True)
  if not candidates:break
  group=candidates[0]
  try:os.killpg(group,signal.SIGSTOP);paused.add(group);events.append(dict(utc=utc(),signal='STOP',pgid=group,allocated_CPU=allocated))
  except ProcessLookupError:groups.remove(group)
 while len(paused)>needed:
  group=min(paused)
  try:os.killpg(group,signal.SIGCONT)
  except ProcessLookupError:pass
  paused.remove(group);events.append(dict(utc=utc(),signal='CONT',pgid=group,allocated_CPU=allocated))
 write(HERE/'RESOURCE_STATUS.json',dict(utc=utc(),allocated_CPU=allocated,external_CPU=external,paused_groups=sorted(paused),active_CPU=allocated-len(paused),events=events,policy='Total actual18, never signal external tasks; no GPU used'))
 if (HERE/'NUMERICAL_COMPLETE.receipt.json').exists() and not own:break
 time.sleep(2)

"""Coordinate own Slurm process groups against all concurrent user jobs."""
from core import *
import subprocess,signal,time
paused=set();events=[];started=False
while True:
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);rows=[s.split('|') for s in q.splitlines()];ownjobs=[s for s in rows if s[1].startswith('r34_')];started|=bool(ownjobs);running=[s for s in rows if s[2] in ['RUNNING','COMPLETING','CONFIGURING']];allocated=sum(int(s[3]) for s in running);ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,pgid=,args='],text=True);groups=set()
 for line in ps.splitlines():
  parts=line.strip().split(None,2)
  if len(parts)<3:continue
  pid,pgid,args=parts
  if str(HERE)+'/' in args and any('/'+s+' ' in args for s in ['run.py','probes.py','advance.py','train.py']) and 'python -B' in args and 'bash -c' not in args and 'resource_monitor' not in args:groups.add(int(pgid))
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
 write(HERE/'RESOURCE_STATUS.json',dict(utc=utc(),allocated_CPU=allocated,paused_groups=sorted(paused),active_CPU=allocated-len(paused),events=events,policy='Preserve in-flight state, actual total18; do not signal external tasks'))
 if (HERE/'NUMERICAL_COMPLETE.receipt.json').exists() and not ownjobs:break
 time.sleep(2)

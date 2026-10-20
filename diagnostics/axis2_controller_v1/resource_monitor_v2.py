"""Adapt paused in-flight own workers to other tasks; never repeat an episode."""
from core import *
import subprocess,signal,time
paused={int(x) for x in sys.argv[1:]};events=read(HERE/'RESOURCE_REBALANCE.json')['events']
while True:
 ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True);own=[]
 for line in ps.splitlines():
  parts=line.strip().split(None,1)
  if len(parts)==2 and str(HERE/'run.py')+' main ' in parts[1] and 'python -B' in parts[1] and not any(t in parts[1] for t in ['bash -c','resource_monitor']):own.append(int(parts[0]))
 own=sorted(own);paused&=set(own)
 if not own:break
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);rows=[s.split('|') for s in q.splitlines()];allocated=sum(int(s[3]) for s in rows if s[2] in ['RUNNING','COMPLETING','CONFIGURING']);needed=max(0,allocated-18)
 while len(paused)<needed:
  candidates=[p for p in reversed(own) if p not in paused]
  if not candidates:break
  pid=candidates[0]
  try:os.kill(pid,signal.SIGSTOP);paused.add(pid);events.append(dict(utc=utc(),signal='STOP',pid=pid,allocated_CPU=allocated,active_CPU=allocated-len(paused)))
  except ProcessLookupError:own.remove(pid)
 while len(paused)>needed:
  pid=min(paused)
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
  paused.remove(pid);events.append(dict(utc=utc(),signal='CONT',pid=pid,allocated_CPU=allocated,active_CPU=allocated-len(paused)))
 write(HERE/'RESOURCE_REBALANCE.json',dict(events=events,utc=utc(),paused_pids=sorted(paused),allocated_CPU=allocated,active_CPU=allocated-len(paused),policy='Actual active workers18; dynamic number of suspended own workers; preserve in-flight episodes',prior_monitor_limitation='v1 paused only one worker; second external CPU task temporarily made19 active, corrected by v2'))
 time.sleep(2)

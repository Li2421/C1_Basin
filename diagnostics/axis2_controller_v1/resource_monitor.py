"""Preserve in-flight simulation while keeping actual running worker total at18."""
from core import *
import subprocess,signal,time
pid=int(sys.argv[1]);stopped=False;events=[]
try:
 while Path(f'/proc/{pid}').exists():
  q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);rows=[s.split('|') for s in q.splitlines()];running=sum(int(s[3]) for s in rows if s[2] in ['RUNNING','COMPLETING','CONFIGURING']);need_stop=running>18
  if need_stop!=stopped:
   os.kill(pid,signal.SIGSTOP if need_stop else signal.SIGCONT);stopped=need_stop;events.append(dict(utc=utc(),signal='STOP' if stopped else 'CONT',allocated_CPU=running,active_CPU=running-int(stopped),pid=pid));write(HERE/'RESOURCE_REBALANCE.json',dict(events=events,policy='Actual active workers18; suspended task retains its in-flight episode without rerun; extra Slurm allocation from concurrent external task',source_preserved=True))
  time.sleep(2)
finally:
 if stopped and Path(f'/proc/{pid}').exists():os.kill(pid,signal.SIGCONT)

from core import *
import subprocess,signal,time
pgid=int(sys.argv[1]);job=sys.argv[2];stopped=True;events=[]
while Path(f'/proc/{pgid}').exists():
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);rows=[s.split('|') for s in q.splitlines()];running=[s for s in rows if s[2] in ['RUNNING','COMPLETING','CONFIGURING']];allocated=sum(int(s[3]) for s in running);desired=allocated>18
 if desired!=stopped:
  try:os.killpg(pgid,signal.SIGSTOP if desired else signal.SIGCONT)
  except ProcessLookupError:break
  stopped=desired;events.append(dict(utc=utc(),signal='STOP' if stopped else 'CONT',allocated_CPU=allocated))
 write(HERE/'STAGE_RESOURCE_REBALANCE.json',dict(utc=utc(),job=job,pgid=pgid,paused=stopped,allocated_CPU=allocated,active_CPU=allocated-int(stopped),events=events,reason='External jobs use18; pause own analysis group until a slot frees; in-memory inference preserved'))
 time.sleep(2)

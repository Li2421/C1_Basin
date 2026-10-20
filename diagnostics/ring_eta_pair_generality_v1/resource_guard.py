"""Resource-only governor; never reads outcomes or changes numerical execution."""
import os, signal, subprocess, json, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
helpers=[639993,639992,639994,642650]
def workers():
 out={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if 'diagnostics/ring_eta_pair_generality_v1/' not in cmd:continue
   if not any(x in cmd for x in ['/run.py main ', '/helper.py ', '/helper2.py ']):continue
   st=(p/'stat').read_text().split(') ')[1].split()[0]
   if st!='Z':out[int(p.name)]=st
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return out
while True:
 q=subprocess.check_output(['squeue','-h','-o','%u|%T'],text=True)
 others=[s for s in q.splitlines() if s.split('|')[0]!='zhihan' and s.endswith('|RUNNING')]
 cap=18 if others else 22
 w=workers();active=sum(s not in ['T','t'] for s in w.values())
 if not w:break
 actions=[]
 for pid in reversed(helpers):
  if active<=cap:break
  if pid in w and w[pid] not in ['T','t']:
   os.kill(pid,signal.SIGSTOP);active-=1;actions.append(['STOP',pid])
 for pid in helpers:
  if active>=cap:break
  if pid in w and w[pid] in ['T','t']:
   os.kill(pid,signal.SIGCONT);active+=1;actions.append(['CONT',pid])
 if actions:
  with (HERE/'RESOURCE_GUARD_LOG.jsonl').open('a') as f:
   f.write(json.dumps(dict(time=time.time(),cap=cap,other_users=others,actions=actions,active=active))+'\n')
 time.sleep(10)

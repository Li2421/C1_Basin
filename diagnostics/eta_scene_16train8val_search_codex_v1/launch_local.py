"""Only pause own workers, honor aggregate user CPU18 including parallel local work."""
from pathlib import Path
import os,sys,time,json,subprocess,signal,math,fcntl
H=Path(__file__).resolve().parent;PY=sys.executable;stage=sys.argv[1];MAX=8;CAP=18

if (H/'USER_ANALYSIS_ONLY.receipt.json').exists():raise SystemExit('User requested analysis only; no computation may start.')

def other_units():
 # Conservative active local numerical processes, plus allocated Slurm shares.
 out=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,pcpu=,stat=,args='],text=True);pids=set();units=0
 for line in out.splitlines():
  parts=line.strip().split(None,3)
  if len(parts)<4:continue
  pid,cpu,state,cmd=parts;pid=int(pid)
  if str(H)in cmd or 'launch_local.py'in cmd or pid==os.getpid()or state.startswith('T'):continue
  if 'python' in cmd.split()[0] and (float(cpu)>=5.0 or (state.startswith('R') and float(cpu)>=.5)):
   pids.add(pid);units+=max(1,math.ceil(float(cpu)/100))
 # Allocation count is a conservative reservation even while workers sleep.
 q=subprocess.run(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-o','%T|%C'],capture_output=True,text=True)
 allocated=sum(int(s.split('|')[1])for s in q.stdout.splitlines()if s.startswith('RUNNING|'))
 return max(units,allocated),sorted(pids)

lock=(H/'CPU_GOVERNOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
workers={};paused=set();tries={};handles=[]
try:
 while True:
  others,pids=other_units();allow=max(0,min(MAX,CAP-1-others))
  alive=[i for i,p in workers.items()if p.poll()is None]
  # Suspending happens before any new spawn. No other agent processes are signaled.
  for i in alive[allow:]:
   if i not in paused:os.kill(workers[i].pid,signal.SIGSTOP);paused.add(i)
  for i in alive[:allow]:
   if i in paused:os.kill(workers[i].pid,signal.SIGCONT);paused.remove(i)
  for i,p in list(workers.items()):
   if p.poll()is not None and p.returncode!=0:
    if tries[i]>=3:raise RuntimeError(('SHARD_FAILED',stage,i,p.returncode))
    del workers[i]
  if len(workers)==MAX and all(p.poll()==0 for p in workers.values()):break
  active=sum(p.poll()is None and i not in paused for i,p in workers.items())
  for i in range(MAX):
   if active>=allow:break
   if i in workers:continue
   f=(H/f'{stage}_worker_{i}.log').open('a');handles.append(f);p=subprocess.Popen([PY,'-B',str(H/'worker_local.py'),stage,str(i),str(MAX)],cwd=H,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);workers[i]=p;tries[i]=tries.get(i,0)+1;active+=1
  status=dict(stage=stage,updated=time.time(),our_worker_limit=MAX,other_user_compute_units=others,other_local_pids=pids,our_active_workers=active,our_paused_workers=sorted(paused),aggregate_limit=CAP,coordinator_allowance=1,worker_pids={str(i):p.pid for i,p in workers.items()},completed_shards=[i for i,p in workers.items()if p.poll()==0])
  (H/'RESOURCE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
  time.sleep(1)
 (H/f'{stage}_LOCAL_COMPLETE.json').write_text(json.dumps(dict(stage=stage,passed=True,shards=MAX,tries=tries),indent=2)+'\n')
finally:
 for i,p in workers.items():
  if p.poll()is None:
   if i in paused:os.kill(p.pid,signal.SIGCONT)
   p.terminate()
 for f in handles:f.close()

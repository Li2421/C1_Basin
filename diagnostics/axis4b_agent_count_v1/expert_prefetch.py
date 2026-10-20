"""Run already preregistered, selection-independent experts in released pilot slots."""
from common import *
import subprocess,time
from submit import phase

def main():
 assert not (HERE/'expert_SUBMISSION.json').exists()
 if not (HERE/'expert_CASES.json').exists():
  with (HERE/'factors_expert_prefetch.log').open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/'factors.py'),'expert'],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 write(HERE/'EXPERT_PREFETCH_STATUS.json',dict(utc=utc(),status='Prepared; waiting until all pilot cases assigned and a totalCPU slot available',selection_independent=True))
 while True:
  assert not (HERE/'PIPELINE_ERROR.json').exists()
  cursor=HERE/'pilot/queue/cursor';assigned=int(cursor.read_text()) if cursor.exists() else 0
  q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);allocated=sum(int(s.split('|')[3]) for s in q.splitlines() if s.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING'])
  if assigned>=len(read(HERE/'pilot_MISSING_CASES.json')) and allocated<18:break
  time.sleep(5)
 if not (HERE/'expert_SUBMISSION.json').exists():phase('expert')
 write(HERE/'EXPERT_PREFETCH_STATUS.json',dict(utc=utc(),status='Submitted independent expert work into released slots',allocated_before=allocated,pilot_assigned=assigned,selection_independent=True))
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'EXPERT_PREFETCH_ERROR.json',dict(utc=utc(),error=repr(ex)));raise

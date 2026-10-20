"""User-authorized resource loan with unconditional cross-type6 restoration."""
from pathlib import Path
import os,sys,signal,json,subprocess,time,datetime,shlex
P=Path(__file__).resolve().parent;ROOT=P.parents[1];C6=ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1';PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
COORDINATOR=2906524;OLD_JOB='26641'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,d):(P/n).write_text(json.dumps(d,indent=2)+'\n')
def queue(job=None):
 cmd=['squeue','-h','-r']+(['-j',job] if job else ['-u',os.environ['USER']])+['-o','%i|%j|%T|%C']
 r=subprocess.run(cmd,capture_output=True,text=True);return r.stdout.strip()
def wait(job):
 while queue(job):
  save('LOAN_LIVE_STATUS.json',dict(utc=utc(),job=job,queue=queue(),completed=len(list((P/'evaluation/raw').glob('*.json')))))
  print(utc(),job,'completed',len(list((P/'evaluation/raw').glob('*.json'))),flush=True);time.sleep(20)
def submit(name,command,workers,memory=4):
 f=P/f'{name}.sbatch';f.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=rhr3_{name}','#SBATCH --partition=cpu','#SBATCH --cpus-per-task=1',f'#SBATCH --mem={memory}G','#SBATCH --time=03:00:00',f'#SBATCH --output={P}/{name}_%A_%a.log',*([f'#SBATCH --array=0-{workers-1}%{workers}'] if workers>1 else []),'set -euo pipefail',f'cd {ROOT}','export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1','unset XLA_FLAGS',command])+'\n')
 active=queue();assert not active,active
 job=subprocess.check_output(['sbatch','--parsable',str(f)],text=True).strip().split(';')[0];save(f'{name}_SUBMISSION.json',dict(utc=utc(),job=job,workers=workers,queue_before=active));return job

def restore():
 # Do not release the original coordinator until a resumable replacement is submitted.
 workers=[i for i in range(18) if not (C6/f'training_WORKER_{i}_COMPLETE.json').exists()]
 if workers:
  version=next(i for i in range(1,4) if not (C6/f'training_retry{i}_SUBMISSION.json').exists())
  commands=['worker_ids=('+' '.join(map(str,workers))+')','unset XLA_FLAGS',f'{PYTHON} -B {C6}/worker_c6.py training "${{worker_ids[${{SLURM_ARRAY_TASK_ID:-0}}]}}" 18']
  code='from submit_aux import submit; submit('+repr(f'training_retry{version}')+','+repr(commands)+',workers='+str(len(workers))+',memory_gb=4)'
  subprocess.run([PYTHON,'-B','-c',code],cwd=C6,check=True)
 os.kill(COORDINATOR,signal.SIGCONT)
 save('CPU_RESTORED.json',dict(utc=utc(),workers=workers,coordinator_pid=COORDINATOR,coordinator_resumed=True,queue=queue()))
 print('C6 CPU restored',flush=True)

def main():
 assert 'coordinator_c6.py' in Path(f'/proc/{COORDINATOR}/cmdline').read_text()
 assert OLD_JOB in queue(),queue()
 before=[str(f) for f in (C6/'training/raw').glob('*.json')]
 save('CPU_LOAN_BEFORE.json',dict(utc=utc(),coordinator_pid=COORDINATOR,old_job=OLD_JOB,queue=queue(),raw_count=len(before),raw_files=before,authorization='User explicitly requested temporary18CPU loan and return'))
 os.kill(COORDINATOR,signal.SIGSTOP)
 try:
  subprocess.run(['scancel',OLD_JOB],check=True)
  while queue(OLD_JOB):time.sleep(2)
  save('CPU_LOAN_PAUSED.json',dict(utc=utc(),old_job=OLD_JOB,completed_records_preserved=all(Path(f).exists() for f in before),queue=queue(),coordinator_paused=True))
  j=submit('reproduction',f'{PYTHON} -B {P}/supplement.py reproduce',1,6);wait(j)
  assert json.loads((P/'REPRODUCTION.json').read_text())['passed']
  j=submit('evaluation',f'{PYTHON} -B {P}/supplement.py worker "${{SLURM_ARRAY_TASK_ID}}"',18,4);wait(j)
  assert all((P/f'evaluation/WORKER_{i}_COMPLETE.json').exists() for i in range(18))
  save('CPU_LOAN_EVALUATION_FINISHED.json',dict(utc=utc(),raw=len(list((P/'evaluation/raw').glob('*.json')))))
 except BaseException as exc:
  save('CPU_LOAN_ERROR.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc)))
  raise
 finally:
  restore()
if __name__=='__main__':main()

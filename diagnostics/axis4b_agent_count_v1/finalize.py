from common import *
import subprocess,time,shlex

def main():
 while not (HERE/'ALL_LABELS_COMPLETE.receipt.json').exists():
  assert not (HERE/'PIPELINE_ERROR.json').exists(),read(HERE/'PIPELINE_ERROR.json') if (HERE/'PIPELINE_ERROR.json').exists() else None
  time.sleep(15)
 while True:
  q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);allocated=sum(int(s.split('|')[3]) for s in q.splitlines() if s.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING'])
  if allocated<18:break
  time.sleep(15)
 assert not (HERE/'FINALIZATION_SUBMISSION.json').exists();args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=6G','--time','06:00:00','-J','r36_analysis','-o',str(HERE/'analysis_%j.log'),'--wrap',f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"analysis_worker_v2.py"))} final'];job=subprocess.check_output(args,text=True).strip();write(HERE/'FINALIZATION_SUBMISSION.json',dict(utc=utc(),job=job,args=args,queue=q));print('Finalization queued',job,flush=True)
if __name__=='__main__':main()

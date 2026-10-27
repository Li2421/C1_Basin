"""Operational queue only; copied v2 fitting functions execute unchanged."""
from common import *
import signal
STOP=False
def stop(*_):
 global STOP;STOP=True

def main(stage):
 from train_v2 import fit as fit_critic
 from train_model import fit as fit_generator
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
 verify_inputs()
 for p,h in read(HERE/'TRAINING_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
 tasks=read(HERE/('TRAIN_INNER_TASKS.json'if stage=='train_inner'else'TRAIN_OUTER_TASKS.json'));folder=HERE/'TRAIN_QUEUE'/stage
 job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'];deadline=time.monotonic()+float(os.environ.get('XDT_TIME_LIMIT_S','3600'))-120
 for i,task in enumerate(tasks):
  if STOP or time.monotonic()>deadline or(HERE/'STOP.receipt.json').exists():return
  target=HERE/'folds'/task['fold']/task['mode']/f"seed{task['seed']}"/'FIT.receipt.json'
  if target.exists():continue
  claim=folder/str(i);claim.parent.mkdir(parents=True,exist_ok=True)
  try:claim.mkdir()
  except FileExistsError:continue
  write(claim/'OWNER.json',dict(job=job,pid=os.getpid(),task=task))
  try:
   row=(fit_generator if task['kind']=='generator'else fit_critic)(task)
   write(claim/'DONE.json',row)
  except BaseException as exc:
   write(HERE/'TRAIN_FATAL'/f'{os.getpid()}.json',dict(stage=stage,task=task,type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
if __name__=='__main__':main(sys.argv[1])

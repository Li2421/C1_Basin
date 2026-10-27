from common import *
import signal
STOP=False
def stop(*_):
 global STOP;STOP=True

def main():
 from train_critic import fit
 from inference import predict_seed
 signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop);verify_inputs()
 for p,h in read(HERE/'TRAINING_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
 tasks=read(HERE/'TRAIN_TASKS.json');job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'];deadline=time.monotonic()+float(os.environ.get('XDT_TIME_LIMIT_S','3600'))-180
 for i,task in enumerate(tasks):
  if STOP or time.monotonic()>deadline:return
  path=HERE/'fits'/task['id']/f"seed{task['seed']}"/'FIT.receipt.json'
  if path.exists()and path.with_name('PREDICTIONS.receipt.json').exists():continue
  claim=HERE/'TRAIN_QUEUE'/str(i);claim.parent.mkdir(exist_ok=True)
  try:claim.mkdir()
  except FileExistsError:continue
  write(claim/'OWNER.json',dict(job=job,pid=os.getpid(),task=task))
  try:r=fit(task);predict_seed(task,r);write(claim/'DONE.json',r)
  except BaseException as exc:
   write(HERE/'TRAIN_FATAL'/f'{os.getpid()}.json',dict(stage='train',task=task,type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
if __name__=='__main__':main()

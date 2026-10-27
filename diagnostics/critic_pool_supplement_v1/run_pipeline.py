from common import *
import subprocess

def run(script):subprocess.run([sys.executable,'-B',str(HERE/script)],cwd=ROOT,check=True)
def main():
 while not(HERE/'ROLLOUTS_COMPLETE.json').exists():
  if(HERE/'ROLLOUT_FATAL.json').exists():raise RuntimeError('ROLLOUT_STAGE_FATAL')
  time.sleep(10)
 while not(HERE/'OLD_POOL_READY.receipt.json').exists():time.sleep(1)
 if not(HERE/'TRAINING_POOL.json').exists():run('labels_new.py')
 if not(HERE/'TRAINING_FROZEN.receipt.json').exists():run('prepare_training.py')
 run('train_dispatch.py')
 if not(HERE/'FROZEN.receipt.json').exists():run('inference.py')
 while not(HERE/'STATISTICS_READY.receipt.json').exists():time.sleep(10)
 run('statistics_pool.py')
 while not(HERE/'DELIVERY_READY.receipt.json').exists():time.sleep(10)
 run('deliver_pool.py')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(HERE/'PIPELINE_FATAL.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise

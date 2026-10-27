"""Finite pipeline; no fits before heterogeneity check, no proposals before all freezes."""
from common import *
import subprocess

def run(script):subprocess.run([sys.executable,'-B',str(HERE/script)],cwd=ROOT,check=True)
def main():
 while not(HERE/'STAGE1_LABELS_COMPLETE.json').exists():
  if(HERE/'RUN_FATAL.json').exists():raise RuntimeError('STAGE1_FATAL')
  time.sleep(10)
 if not(HERE/'LABEL_TABLES.json').exists():run('labels.py')
 if not(HERE/'HETEROGENEITY_CHECK.json').exists():run('heterogeneity.py')
 if not(HERE/'TRAINING_FROZEN.receipt.json').exists():run('prepare_learning.py')
 from train_dispatch import run as train
 train('train_inner')
 if not(HERE/'LAMBDA_SELECTION_SUMMARY.json').exists():run('select_lambdas_v2.py')
 train('train_outer');run('freeze_choices_v2.py')
 from dispatch import run as rollout
 if not(HERE/'proposal_PREFLIGHT.receipt.json').exists():run('requests_proposal.py')
 rollout('proposal');run('statistics_hetero.py')
 while not(HERE/'DELIVERY_READY.receipt.json').exists():time.sleep(10)
 run('deliver.py')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(HERE/'PIPELINE_FATAL.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise

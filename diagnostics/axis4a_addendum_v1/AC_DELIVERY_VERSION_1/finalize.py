from common import *
from pipeline import queue
from single_job import submit
import time

def main():
 while not (HERE/'A_COMPLETE.receipt.json').exists():
  assert not (HERE/'PIPELINE_ERROR.json').exists() and not (HERE/'A_CONTINUATION_ERROR.json').exists();time.sleep(5)
 while (HERE/'PSTAR_DECISION.receipt.json').exists() and read(HERE/'PSTAR_DECISION.receipt.json')['authorized_rollouts'] and not (HERE/'B_COMPLETE.receipt.json').exists():
  assert not (HERE/'B_PIPELINE_ERROR.json').exists();time.sleep(5)
 if not (HERE/'FINAL_SUBMISSION.json').exists():submit('FINAL','analysis_worker.py')
if __name__=='__main__':main()

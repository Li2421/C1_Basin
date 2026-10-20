from common import *
from single_job import submit
from pipeline import queue
import time

def main():
 while not (HERE/'B_COMPLETE.receipt.json').exists():
  assert not (HERE/'B_PIPELINE_ERROR.json').exists()
  time.sleep(5)
 seal('B_FINALIZATION_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'B_final/analysis_worker.py',HERE/'finalize_B.py',HERE/'B_DELIVERY_ADAPTER_FROZEN.receipt.json'],before_finalization=True,only_read_existing_labels=True)
 submit('Bfinal','B_final/analysis_worker.py')
 while not (HERE/'B_FINAL_COMPLETE.receipt.json').exists():
  assert not (HERE/'B_FINALIZATION_ERROR.json').exists()
  time.sleep(5)
 print('B finalization PASS',flush=True)
if __name__=='__main__':main()

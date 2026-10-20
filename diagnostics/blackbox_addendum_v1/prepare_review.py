from common import *
from run_D import call
import time,traceback
if __name__=='__main__':
 try:
  while not(HERE/'E_COMPLETE.receipt.json').exists():time.sleep(10)
  if (HERE/'A_STATS_PENDING.json').exists():
   while not(HERE/'A_STATS_READY.json').exists()and not(HERE/'A_STATS_ERROR.json').exists():time.sleep(5)
   assert not(HERE/'A_STATS_ERROR.json').exists(),read(HERE/'A_STATS_ERROR.json')if(HERE/'A_STATS_ERROR.json').exists()else''
  if not(HERE/'E_PER_INSTANCE_RESULTS.receipt.json').exists():call('per_instance_E.py')
  if not(HERE/'A_STANDARD_RESULTS.receipt.json').exists():call('standard_A.py')
  if not(HERE/'A_CONTROLLER_RADIUS_CURVES.receipt.json').exists():call('geometry_curves.py')
  call('verify_delivery.py');call('deliver.py');call('supplement_tables.py');write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='READY_FOR_REVIEW',completed_parts=['A','B','C','D','E','F']))
 except Exception:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=traceback.format_exc()));raise

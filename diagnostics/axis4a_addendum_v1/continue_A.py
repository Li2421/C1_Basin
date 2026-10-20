from common import *
from pipeline import wait,call
from submit import phase
import time

def main():
 while not (HERE/'GEOMETRY_ELIGIBILITY_FROZEN.receipt.json').exists():
  assert not (HERE/'PIPELINE_ERROR.json').exists();time.sleep(5)
 if not (HERE/'GA3_CASES.json').exists() and not (HERE/'GA3_NO_REQUESTS.json').exists():call('plan.py','GA3')
 if (HERE/'GA3_CASES.json').exists():
  if not (HERE/'GA3_SUBMISSION.json').exists():phase('GA3')
  wait('GA3');call('analyze_geometry.py')
 else:write(HERE/'GEOMETRY_MAIN_RESULTS.json',{})
 seal('A_COMPLETE.receipt.json',[HERE/'CONTROL_RESULTS.json',HERE/'GEOMETRY_MAIN_RESULTS.json',HERE/'GEOMETRY_ELIGIBILITY_FROZEN.receipt.json'],B_status='Independent; still requires userdecision')
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'A_CONTINUATION_ERROR.json',dict(utc=utc(),error=repr(ex),unknown_retries=0));raise

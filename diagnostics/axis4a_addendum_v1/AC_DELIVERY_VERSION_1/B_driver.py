from common import *
from pipeline import call,wait,queue
from submit import phase
from single_job import submit
import time

def waitname(name):
 while any(x.split('|')[1]=='r37_'+name for x in queue().splitlines()):time.sleep(5)
def main():
 while not (HERE/'PSTAR_DECISION.receipt.json').exists():
  if (HERE/'NUMERICAL_COMPLETE.receipt.json').exists():return
  time.sleep(5)
 d=read(HERE/'PSTAR_DECISION.receipt.json')
 if not d['authorized_rollouts']:
  seal('B_NOT_EXECUTED.receipt.json',[HERE/'PSTAR_DECISION.receipt.json'],reason='Explicituserabandonment');return
 assert read(HERE/'CONSISTENCY_GATE.json')['passed']
 if not (HERE/'SELECTED_PROGRESS_TEST.receipt.json').exists():
  if not (HERE/'Bunit_SUBMISSION.json').exists():submit('Bunit','test_selected_progress.py','2G')
  waitname('Bunit');assert read(HERE/'SELECTED_PROGRESS_TEST.json')['passed']
 if not (HERE/'B_CASES.json').exists():call('plan_B.py')
 if not (HERE/'B_SUBMISSION.json').exists():phase('B')
 wait('B')
 if not (HERE/'B_RESULTS.json').exists():
  if not (HERE/'Banalysis_SUBMISSION.json').exists():submit('Banalysis','analyze_B.py')
  waitname('Banalysis');assert (HERE/'B_RESULTS.json').exists()
 seal('B_COMPLETE.receipt.json',[HERE/'B_NUMERICAL_COMPLETE.receipt.json',HERE/'B_RESULTS.json',HERE/'PSTAR_DECISION.receipt.json',HERE/'SELECTED_PROGRESS_TEST.receipt.json'],definition_amendment=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'B_PIPELINE_ERROR.json',dict(utc=utc(),error=repr(ex),unknown_retries=0));raise

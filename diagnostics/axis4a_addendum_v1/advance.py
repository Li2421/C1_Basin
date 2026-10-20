from common import *
from pipeline import wait,call
import time

def main():
 wait('gate')
 if not (HERE/'CONSISTENCY_GATE.json').exists():call('check_gate.py')
 if not read(HERE/'CONSISTENCY_GATE.json')['passed']:
  write(HERE/'A_INVALIDATED.json',dict(utc=utc(),reason='Config or trajectory gate failed; allAconclusions invalid, no controls submitted'));return
 wait('A')
 if not (HERE/'GEOMETRY_ELIGIBILITY_FROZEN.receipt.json').exists():call('analyze_controls.py')
 write(HERE/'A_STAGE_READY.json',dict(utc=utc(),status='Controls analyzed; conditionalGA3 branch requires dispatch',eligible=read(HERE/'CONTROL_RESULTS.json')['GA3']['eligible_radii']))
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'PIPELINE_ERROR.json',dict(utc=utc(),error=repr(ex),unknown_retries=0));raise

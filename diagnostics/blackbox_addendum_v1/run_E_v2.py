from common import *
from run_D import call,submit,finish
import time,subprocess,traceback

def timing_submit():
 while sum(int(x) for x in subprocess.check_output(['squeue','-r','-h','-t','RUNNING,COMPLETING,CONFIGURING','-o','%C'],text=True).split())>=18:time.sleep(10)
 call('submit_phase.py','gap_timing');job=read(HERE/'gap_timing_SUBMISSION.json')['job']
 while subprocess.check_output(['squeue','-r','-h','-j',job],text=True).strip():time.sleep(10)
 assert all((HERE/'gap_timing'/f'worker_{i}.json').exists() for i in range(18));call('timing_postflight.py')
def main():
 while not (HERE/'D_NUMERICAL_COMPLETE.json').exists() and not (HERE/'D_GAP.json').exists():time.sleep(10)
 if (HERE/'D_NUMERICAL_COMPLETE.json').exists() and not (HERE/'D_COMPLETE.receipt.json').exists():call('analyze_D.py')
 assert read(HERE/'MPC_RING_EQUIVALENCE.json')['passed'];call('plan_E_v2.py','replay');submit('replay_E');call('check_E_v2.py','replay')
 if not read(HERE/'E_REPRODUCTION_GATE.json')['passed']:write(HERE/'E_GAP.json',dict(utc=utc(),reason='Full Gap execution reproduction gate failed'));return
 call('plan_E_v2.py','timing');timing_submit();call('check_E_v2.py','prefix');call('analyze_timing.py');call('plan_E_v2.py','gate')
 if read(HERE/'gap_capability_MISSING_CASES.json'):submit('gap_capability')
 else:call('postflight.py','gap_capability')
 call('analyze_E.py','capability');gate=read(HERE/'MPC_CAPABILITY_GATE.json');Ns=[n for n in [2,10,20] if any(v['status']=='PASS' for v in gate[str(n)].values())]
 for j,n in enumerate(Ns):
  if j:assert (HERE/'MPC_SOURCE_CHOICES_FROZEN.receipt.json').exists()
  for which in ['fixed','schedules']:
   call('plan_E_v2.py','mainphase',str(n),which);submit(f'gap_N{n}_{which}')
  if j==0:call('analyze_E.py','freeze',str(n))
 call('analyze_E.py','finish');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='final_audit',status='READY',completed_parts=['F','C','A','B','D','E']))
if __name__=='__main__':
 try:main()
 except Exception:write(HERE/'E_COORDINATOR_ERROR.json',dict(utc=utc(),error=traceback.format_exc()));raise

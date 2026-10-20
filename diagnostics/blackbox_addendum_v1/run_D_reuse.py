from common import *
from run_D import call,finish
import subprocess,time,traceback
def main():
 finish('right_hand_tune');call('choose_right_hand.py');call('plan_D_reuse.py')
 while sum(int(x)for x in subprocess.check_output(['squeue','-r','-h','-t','RUNNING,COMPLETING,CONFIGURING','-o','%C'],text=True).split())>=18:time.sleep(10)
 call('submit_reuse.py','right_hand_evaluation');finish('right_hand_evaluation');write(HERE/'D_NUMERICAL_COMPLETE.json',dict(utc=utc(),status='COMPLETE'));write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='D_analysis',status='READY'))
if __name__=='__main__':
 try:main()
 except Exception:write(HERE/'D_COORDINATOR_ERROR.json',dict(utc=utc(),error=traceback.format_exc()));raise

from common import *
import subprocess,time,traceback
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def call(f,*args):
 print('RUN',f,args,flush=True);subprocess.run([PY,'-B',str(HERE/f),*args],check=True,cwd=ROOT)
def finish(phase):
 job=read(HERE/f'{phase}_SUBMISSION.json')['job']
 while subprocess.check_output(['squeue','-r','-h','-j',job],text=True).strip():time.sleep(10)
 if not all((HERE/phase/f'worker_{i}.json').exists() for i in range(18)):raise RuntimeError('Incomplete worker phase '+phase)
 call('postflight.py',phase)
def submit(phase):
 while True:
  cpus=sum(int(x) for x in subprocess.check_output(['squeue','-r','-h','-t','RUNNING,COMPLETING,CONFIGURING','-o','%C'],text=True).split())
  if cpus<18:break
  time.sleep(10)
 call('submit_phase.py',phase);finish(phase)
def main():
 finish('replay_D');call('check_D_gate.py')
 gate=read(HERE/'D_REPRODUCTION_GATE.json')
 if not (gate['Flow_inactive_passed'] and gate['Novel_inactive_passed']):
  write(HERE/'D_GAP.json',dict(reason='Inactive trajectory reproduction gate failed',gate=gate));return
 call('plan_D.py','tune');submit('right_hand_tune');call('choose_right_hand.py');call('plan_D.py','evaluation');submit('right_hand_evaluation');write(HERE/'D_NUMERICAL_COMPLETE.json',dict(utc=utc(),status='COMPLETE'));write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='D_analysis',status='READY'))
if __name__=='__main__':
 try:main()
 except Exception:write(HERE/'D_COORDINATOR_ERROR.json',dict(utc=utc(),error=traceback.format_exc()));raise

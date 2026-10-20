from common import *
import subprocess,time,traceback
G=guard()
def run(script,*args):
 cmd=[sys.executable,'-B',str(HERE/script),*args];print('RUN',cmd,flush=True);r=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'});assert r.returncode==0,(cmd,r.returncode)
def wait_stage(phase):
 while True:
  sub=read(HERE/f'{phase}_SUBMISSION.json');job=sub['job'].split(';')[0];sq=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T'],text=True)
  done=sum(1 for c in read(HERE/f'{phase}_MISSING_CASES.json') if (HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists());write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='RUNNING',phase=phase,completed=done,total=len(read(HERE/f'{phase}_MISSING_CASES.json')),job=job,squeue=sq,guard=guard()))
  if not sq.strip():
   assert done==len(read(HERE/f'{phase}_MISSING_CASES.json')),(phase,done,'Stage incomplete; no automatic scientific retry');return
  time.sleep(30)
def stage(phase):
 run('design.py',phase);run('submit.py',phase);wait_stage(phase);run('postflight.py',phase)
def main():
 while not (HERE/'round33_REPLAY_GATE.receipt.json').exists():
  sq=subprocess.check_output(['squeue','-h','-j','22826','-o','%i|%T'],text=True);assert sq.strip(),'Replay stopped before receipt';time.sleep(10)
 assert read(HERE/'round28_REPLAY_GATE.json')['passed']
 write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='RUNNING',phase='probe_setup'))
 stage('probe');run('probes.py');run('freeze_choices.py')
 # Supplement the copied unchanged choice script with the other constant rule.
 choices=read(HERE/'FROZEN_CHOICES.json');write(HERE/'CONSTANT_CHOICES.json',{n:dict(always_S500='S500',always_F10='F10') for n in choices['controller_order']});seal('CONSTANT_CHOICES_FROZEN.receipt.json',[HERE/'CONSTANT_CHOICES.json',HERE/'CHOICES_FROZEN.receipt.json'],before_any_evaluation_labels=True)
 stage('pilot');run('gates.py','pilot');stage('tune');run('gates.py','tune');stage('main')
 run('analyze.py');run('integrity.py','END');run('delivery.py')
if __name__=='__main__':
 try:main()
 except Exception as e:write(HERE/'COORDINATOR_ERROR.json',dict(utc=utc(),error=str(e),traceback=traceback.format_exc(),guard=guard()));raise

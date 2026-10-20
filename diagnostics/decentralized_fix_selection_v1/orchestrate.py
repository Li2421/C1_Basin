from support import *
import subprocess,time
G=guard()
def call(*args):
 r=subprocess.run([sys.executable,'-B',str(HERE/args[0]),*args[1:]],text=True,capture_output=True);print(r.stdout+r.stderr,flush=True);assert r.returncode==0
def main():
 while not (HERE/'REPLAY_GATE.receipt.json').exists():time.sleep(10)
 assert read(HERE/'REPLAY_GATE.json')['passed']
 for phase in ['diagnostic','supplement']:
  # Do not reserve the completed predecessor's last still-running Slurm slot.
  while subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%j'],text=True).strip():time.sleep(10)
  call('submit_new.py',phase,'18')
  slots=read(HERE/f'{phase}_SUBMISSION.json')['slots']
  while not all((HERE/phase/f'worker{i}.json').exists() for i in range(slots)):
   jobs=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%j'],text=True)
   if 'r30h_'+phase not in jobs:raise RuntimeError(f'{phase} ended with missing worker completion: preserve partial evidence; inspect logs')
   time.sleep(20)
  print('COMPLETE',phase,utc(),flush=True)
  if phase=='diagnostic':call('freeze_diagnostics.py')
 write(HERE/'SIMULATION_COMPLETE.json',dict(utc=utc(),guard=guard()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(HERE/'ORCHESTRATION_FAILED.json',dict(utc=utc(),type=type(e).__name__,message=str(e)));raise

from support import *
import subprocess,time
G=guard()
def call(script,*args):
 r=subprocess.run([sys.executable,'-B',str(HERE/script),*args],capture_output=True,text=True);print(r.stdout+r.stderr,flush=True);assert r.returncode==0,(script,r.returncode)
def waitphase(phase):
 slots=read(HERE/f'{phase}_SUBMISSION.json')['slots']
 while not all((HERE/phase/f'worker{i}.json').exists() for i in range(slots)):
  q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%j'],text=True)
  if 'r30i_'+phase not in q:raise RuntimeError(f'{phase} jobs ended with incomplete workers; preserve partial evidence, no retry')
  time.sleep(15)
def idle():
 while subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%j'],text=True).strip():time.sleep(10)
def main():
 while not (HERE/'UNIT_TESTS.json').exists():time.sleep(10)
 call('probe_design.py');idle();call('submit.py','diagnostic','18');waitphase('diagnostic');idle();call('submit.py','probe','18');waitphase('probe');call('freeze_features_choices.py');call('design.py');idle();call('submit.py','main','18');waitphase('main');call('postflight.py','main');call('analyze.py');call('plots.py');call('extra_dependency.py','END');call('integrity.py','END');call('report.py');call('verify.py');print('ALL COMPLETE',utc(),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(HERE/'PIPELINE_STOPPED.json',dict(utc=utc(),type=type(e).__name__,message=str(e)));raise

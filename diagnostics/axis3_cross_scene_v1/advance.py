from common import *
import subprocess,fcntl
from submit import phase

def call(script,args,log):
 with (HERE/log).open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def merge(phase):
 with (HERE/'OFFICIAL_MERGER.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  with (HERE/f'{phase}_MERGE.log').open('w') as log:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 write(HERE/f'{phase}_POSTFLIGHT.json',planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json'))
def main(step):
 if step=='gate_v2_done':merge('gate_v2');call('check_gate.py',[],'CONSISTENCY_GATE.log');call('states.py',[],'STATE_GENERATION.log');call('novel_design.py',['pilot'],'DESIGN_PILOT.log');phase('pilot',nextstep='pilot_done')
 elif step=='pilot_done':merge('pilot');call('novel_design.py',['capability'],'CAPABILITY.log');call('probes.py',['design'],'PROBE_DESIGN.log');phase('probe',entry='probes.py',nextstep='probe_done')
 elif step=='probe_done':merge('probe');call('probes.py',['freeze'],'R5_FREEZE.log');call('prepare_training.py',[],'PREPARE_TRAINING.log');call('train.py',['submit'],'TRAIN_SUBMIT.log')
 elif step=='fit_done':
  call('freeze_choices.py',[],'FREEZE_CHOICES.log');call('novel_design.py',['evaluation'],'DESIGN_MAIN.log');assert (HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json').exists(),'No evaluation until analysis frozen';phase('main',nextstep='main_done')
 elif step=='main_done':merge('main');call('analyze.py',[],'ANALYSIS.log');call('audit.py',['END'],'INTEGRITY_END.log');call('delivery.py',[],'DELIVERY.log');call('review.py',[],'REVIEW.log')
 else:raise ValueError(step)
 write(HERE/'LAST_STAGE.json',dict(utc=utc(),completed_step=step))
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception as ex:write(HERE/f'{sys.argv[1]}_ERROR.json',dict(utc=utc(),error=repr(ex),no_automatic_rollout_retry=True));raise

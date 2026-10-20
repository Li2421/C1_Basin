from common import *
import subprocess,fcntl,time
from submit import phase as submit_phase

def call(script,args=()):
 with (HERE/(script.replace('.py','')+'_'+('_'.join(map(str,args)) or 'main')+'.log')).open('a') as out:subprocess.run([sys.executable,'-B',str(HERE/script),*map(str,args)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
def merge(stage):
 from shared_rollout_db.src.rollout_db import canonical
 with (HERE/'OFFICIAL_MERGER.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  with (HERE/f'{stage}_MERGE.log').open('a') as log:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows=read(HERE/f'{stage}_CASES.json');unknown=0
 with rollout_db.connect(True) as db:
  for c in rows:
   q=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r and not r['conflict_quarantined'],c['rollout_uid'];assert all(r[k]==q[k] for k in ['success','collision','timeout','deadlock','numerical_failure']);assert r['raw_record_hash']==hashlib.sha256(canonical(q).encode()).hexdigest();unknown+=q['numerical_failure']
 write(HERE/f'{stage}_POSTFLIGHT.json',dict(utc=utc(),planned=len(rows),verified=len(rows),unknown=unknown,planner=planner.preflight(HERE/f'{stage}_PLANNED_ROLLOUTS.json')));seal(f'{stage}_NUMERICAL_COMPLETE.receipt.json',[HERE/f'{stage}_CASES.json',HERE/f'{stage}_POSTFLIGHT.json'],records=len(rows),unknown=unknown)
def running(stage):
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%j|%T'],text=True);return any(x.split('|')[0]=='r36_'+stage for x in q.splitlines())
def wait(stage):
 if (HERE/f'{stage}_NUMERICAL_COMPLETE.receipt.json').exists() or (HERE/f'{stage}_NO_REQUESTS.json').exists():return
 while running(stage):time.sleep(15)
 cases=read(HERE/f'{stage}_CASES.json');missing=[c['rollout_uid'] for c in cases if not (HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists()];assert not missing,('No automatic retry',stage,len(missing),missing[:5]);merge(stage)
def runstage(stage):
 if (HERE/f'{stage}_NO_REQUESTS.json').exists() or (HERE/f'{stage}_NUMERICAL_COMPLETE.receipt.json').exists():return
 if not (HERE/f'{stage}_SUBMISSION.json').exists():submit_phase(stage)
 wait(stage)
def main():
 wait('gate')
 if not (HERE/'CONSISTENCY_GATE.json').exists():call('check_gate.py')
 if not (HERE/'STATES.json').exists():call('states.py')
 assert (HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json').exists()
 if not (HERE/'pilot_CASES.json').exists() and not (HERE/'pilot_NO_REQUESTS.json').exists():call('factors.py',['pilot'])
 runstage('pilot')
 if not (HERE/'CAPABILITY_GATE_FROZEN.receipt.json').exists():call('factors.py',['capability'])
 n=read(HERE/'CAPABILITY_GATE.json')['source_N']
 if n is not None:
  if not (HERE/'source_CASES.json').exists() and not (HERE/'source_NO_REQUESTS.json').exists():call('factors.py',['source'])
  runstage('source')
 if not (HERE/'TRANSFER_CHOICES_FROZEN.receipt.json').exists():call('freeze_transfer.py')
 if n is not None:
  if not (HERE/'target_CASES.json').exists() and not (HERE/'target_NO_REQUESTS.json').exists():call('factors.py',['target'])
  runstage('target')
 if not (HERE/'expert_CASES.json').exists() and not (HERE/'expert_NO_REQUESTS.json').exists():call('factors.py',['expert'])
 runstage('expert');seal('ALL_LABELS_COMPLETE.receipt.json',[HERE/'CAPABILITY_GATE_FROZEN.receipt.json',HERE/'TRANSFER_CHOICES_FROZEN.receipt.json'],source_N=n,source_labels_used_for_selection=True,target_eval_labels_opened_only_after_choice_freeze=True);write(HERE/'PIPELINE_STAGE.json',dict(utc=utc(),stage='labels complete; awaiting analysis'))
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'PIPELINE_ERROR.json',dict(utc=utc(),error=repr(ex),no_automatic_unknown_or_crash_retry=True));raise

from common import *
import subprocess,fcntl,time
from submit import phase as submit_phase

def call(script,args=()):
 with (HERE/(script.replace('.py','')+'_'+('_'.join(map(str,args)) or 'main')+'.log')).open('a') as out:subprocess.run([sys.executable,'-B',str(HERE/script),*map(str,args)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
def merge(stage):
 from shared_rollout_db.src.rollout_db import uid,canonical
 with (HERE/'OFFICIAL_MERGER.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  with (HERE/f'{stage}_MERGE.log').open('a') as log:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows=read(HERE/f'{stage}_CASES.json');checks=[];aliases=[]
 with rollout_db.connect(True) as db:
  for c in rows:
   actual=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=c['seed']))));aliases.append(dict(request_rollout_uid=c['rollout_uid'],db_rollout_uid=actual));row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(actual,)).fetchone();assert row is not None,actual;assert not row['conflict_quarantined'];r=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");assert all(int(row[k])==r[k] for k in ['success','collision','timeout','numerical_failure']);checks.append(c['rollout_uid'])
 write(HERE/f'{stage}_DB_IDENTITY_MAP.json',aliases);write(HERE/f'{stage}_POSTFLIGHT.json',dict(utc=utc(),planned=len(rows),verified=len(checks),unknown_preserved=True,planner=planner.preflight(HERE/f'{stage}_PLANNED_ROLLOUTS.json')))
 if not (HERE/f'{stage}_NUMERICAL_COMPLETE.receipt.json').exists():seal(f'{stage}_NUMERICAL_COMPLETE.receipt.json',[HERE/f'{stage}_CASES.json',HERE/f'{stage}_POSTFLIGHT.json'],records=len(rows))
def running(stage):
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%j|%T'],text=True);return any(x.split('|')[0]=='r35_'+stage for x in q.splitlines())
def wait(stage):
 if (HERE/f'{stage}_NUMERICAL_COMPLETE.receipt.json').exists():return
 while running(stage):time.sleep(15)
 cases=read(HERE/f'{stage}_CASES.json');missing=[c['rollout_uid'] for c in cases if not (HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists()];assert not missing,('No automatic retry',stage,len(missing),missing[:5]);merge(stage)
def runstage(stage):
 if (HERE/f'{stage}_NUMERICAL_COMPLETE.receipt.json').exists():return
 if not (HERE/f'{stage}_SUBMISSION.json').exists():submit_phase(stage)
 wait(stage)
def main():
 wait('baseline')
 if not (HERE/'horizon_CASES.json').exists():call('factor_design.py',['horizon'])
 runstage('horizon');assert read(HERE/'EXPERT_CONSISTENCY_GATE.json')['passed']
 from factor_design import experts
 if not (HERE/'horizon_expert_CASES.json').exists():experts([(H,.9) for H in [500,700,1000]],'horizon_expert')
 runstage('horizon_expert')
 if not (HERE/'HORIZON_NUMERICAL_COMPLETE.receipt.json').exists():seal('HORIZON_NUMERICAL_COMPLETE.receipt.json',[HERE/'baseline_NUMERICAL_COMPLETE.receipt.json',HERE/'horizon_NUMERICAL_COMPLETE.receipt.json',HERE/'horizon_expert_NUMERICAL_COMPLETE.receipt.json'],Sprog='GAP undefined full896median; no explicit clarification')
 if not (HERE/'pilot_CASES.json').exists():call('geometry.py',['pilot'])
 runstage('pilot')
 if not (HERE/'CAPABILITY_GATE_FROZEN.receipt.json').exists():call('geometry.py',['capability'])
 if not (HERE/'GEOMETRY_ELIGIBLE_RADII.json').exists():call('geometry.py',['evaluation'])
 radii=read(HERE/'GEOMETRY_ELIGIBLE_RADII.json')
 if radii:runstage('geometry')
 if not (HERE/'geometry_expert_CASES.json').exists():experts([(700,r) for r in read(HERE/'DESIGN_CONFIG.json')['radii']],'geometry_expert')
 runstage('geometry_expert');seal('ALL_FACTOR_LABELS_COMPLETE.receipt.json',[HERE/'HORIZON_NUMERICAL_COMPLETE.receipt.json',HERE/'CAPABILITY_GATE_FROZEN.receipt.json'],eligible_radii=radii,expert_radii='all four, including capability failures')
 write(HERE/'PIPELINE_STAGE.json',dict(utc=utc(),stage='factor labels complete; awaiting final analysis'))
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'PIPELINE_ERROR.json',dict(utc=utc(),error=repr(ex),no_automatic_unknown_or_crash_retry=True));raise

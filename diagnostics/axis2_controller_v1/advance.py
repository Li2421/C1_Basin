"""Sequential stages within round33; no cross-round scheduling."""
from common import *
import subprocess,shlex,fcntl

def call(script,args,log):
 with (HERE/log).open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def merge(phase):
 with (HERE/'OFFICIAL_MERGER.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with (HERE/f'{phase}_MERGE.log').open('w') as f:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(HERE/f'{phase}_POSTFLIGHT.json',planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json'))
def verify_frozen():
 receipts=[HERE/'MIXED_FROZEN.receipt.json',HERE/'NEW_CONTROLLERS_FROZEN.receipt.json']+list((HERE/'gaussian').glob('*/COMPLETE.receipt.json'))
 for receipt in receipts:
  for path,digest in read(receipt)['files'].items():assert sha(path)==digest,(receipt,path)
def phase_submit(phase,next_step):
 prereg();assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists();assert not (HERE/f'{phase}_SUBMISSION.json').exists();current=os.environ.get('SLURM_JOB_ID');q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);others=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING'] and x.split('|')[0]!=current);slots=min(18-others,len(read(HERE/f'{phase}_MISSING_CASES.json')));assert slots>0,(others,q)
 entry=HERE/'probes.py' if phase=='probe' else HERE/'run.py';tail='worker $SLURM_ARRAY_TASK_ID' if phase=='probe' else phase+' $SLURM_ARRAY_TASK_ID';cmd=f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(entry))} {tail}'
 args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=4G','--time','12:00:00','--array',f'0-{slots-1}%{slots}','-J','r33_'+phase,'-o',str(HERE/(phase+'_%A_%a.log')),'--wrap',cmd]
 if current:args[1:1]=['--dependency','afterok:'+current]
 files=[HERE/f'{phase}_PREFLIGHT.receipt.json',HERE/'run.py',HERE/'runtime.py',HERE/'controllers.py',entry]
 if phase=='main':files.extend([HERE/'CHOICES_FROZEN.receipt.json',HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json'])
 seal(f'{phase}_SUBMISSION_PREFLIGHT.receipt.json',files,before_submission=True,queue=q,slots=slots,current_orchestrator_dependency=current,authorizes_total_CPU18=True)
 job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),job=job,slots=slots,args=args,queue=q));stage_submit(next_step,job);print('Submitted',phase,job,slots,flush=True)
def stage_submit(step,parent):
 cmd=f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"advance.py"))} {shlex.quote(step)}';args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=12G','--time','03:00:00','--dependency','afterok:'+parent,'-J','r33_stage_'+step,'-o',str(HERE/(step+'_%j.log')),'--wrap',cmd];job=subprocess.check_output(args,text=True).strip();write(HERE/f'{step}_STAGE_JOB.json',dict(utc=utc(),job=job,parent=parent,args=args));return job

def main(step):
 if step=='baseline_done':
  assert len(list((HERE/'baseline/raw').glob('*.json')))==6784;merge('baseline');verify_frozen();phase_submit('pilot','pilot_done')
 elif step=='pilot_done':
  verify_frozen();merge('pilot');call('novel_design.py',['capability'],'CAPABILITY.log');call('probes.py',['design'],'PROBE_DESIGN.log')
  if read(HERE/'CAPABILITY_GATE.json')['eligible']:phase_submit('probe','probes_done')
  else:
   write(HERE/'NEW_TYPES_EMPTY.json',dict(utc=utc(),all_candidates_excluded_or_ambiguous=True,CT1='INCONCLUSIVE',CT2='INCONCLUSIVE',CT3='INCONCLUSIVE'));main('probes_done')
 elif step=='probes_done':
  verify_frozen();call('probes.py',['freeze'],'R5_FREEZE.log');call('freeze_choices.py',[],'FREEZE_CHOICES.log');call('novel_design.py',['main_design'],'DESIGN_MAIN.log')
  if read(HERE/'CAPABILITY_GATE.json')['eligible']:
   assert (HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json').exists(),'Analysis implementation not sealed; no new evaluation submitted';phase_submit('main','main_done')
  else:main('main_done')
 elif step=='main_done':
  verify_frozen();merge('main');call('analyze.py',[],'ANALYSIS.log');call('audit.py',['END'],'INTEGRITY_END.log');call('delivery.py',[],'DELIVERY.log');call('review.py',[],'REVIEW.log');write(HERE/'LIVE_STATUS.json',dict(utc=utc(),phase='round33 complete',next_round34_started=False))
 else:raise ValueError(step)
 write(HERE/'LAST_STAGE.json',dict(utc=utc(),completed_step=step))
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception as exc:write(HERE/f'{sys.argv[1]}_ERROR.json',dict(utc=utc(),error=repr(exc),no_automatic_rollout_retry=True));raise

"""Finish authorized phases and fill only unused account CPU capacity."""
from common import *
import subprocess,time

def call(file,*args):
 tag=file.removesuffix('.py')+'_'+'_'.join(args);log=HERE/'orchestration_logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('a') as out:subprocess.run([sys.executable,'-B',str(HERE/file),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)

def complete(phase):
 p=HERE/f'{phase}_MISSING_CASES.json'
 return p.exists() and len(list((HERE/phase/'raw').glob('*.json')))==len(read(p))

def finish(phase):
 marker=HERE/f'{phase}_FINISHED.json'
 if marker.exists():return
 if not complete(phase):return
 call('finish_phase.py',phase);write(marker,dict(utc=utc(),passed=True));print('FINISHED',phase,flush=True)

def queue():
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True)
 with (HERE/'RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),queue=q,allocated_cpu=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING']),gpu_shard_jobs=sum('shard' in x.split('|')[4] for x in q.splitlines())))+'\n')
 return q

def fill(phase,max_slots):
 if complete(phase):return
 q=queue();used=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','PENDING','COMPLETING','CONFIGURING']);slots=min(18-used,max_slots)
 if slots<=0:return
 if not (HERE/f'{phase}_SUBMISSION.json').exists():call('submit.py',phase,str(slots))
 else:
  # Claim only the shared cursor's unclaimed tail; never requeue an in-flight case.
  cursor=HERE/phase/'queue/cursor';i=int(cursor.read_text()) if cursor.exists() else 0
  if i>=len(read(HERE/f'{phase}_MISSING_CASES.json')):return
  labels=list(HERE.glob(phase+'_EXTRA_auto*.json'));call('add_workers.py',phase,str(slots),'auto'+str(len(labels)))
 print('FILLED',phase,slots,flush=True)

def main():
 stages=['pilot_centralized_sampling','pilot_deterministic','pilot_prioritized_astar']
 while True:
  for stage in stages:
   if complete(stage):
    finish(stage)
    if not (HERE/f'{stage}_CAPABILITY.json').exists():call('capability.py',stage)
    pp=('probe_only_' if stage=='pilot_centralized_sampling' else 'probe_')+stage;tp='tune_'+stage
    if not (HERE/f'{pp}_MISSING_CASES.json').exists():call('probe.py',stage)
    if not (HERE/f'{tp}_MISSING_CASES.json').exists():call('fair_tuning.py','design',stage)
    if complete(pp):finish(pp)
    if complete(tp):
     finish(tp)
     if not (HERE/f'{stage}_FAIR_RULES_FROZEN.receipt.json').exists():call('fair_tuning.py','select',stage)
  ready=all((HERE/f'{stage}_FAIR_RULES_FROZEN.receipt.json').exists() and (HERE/(('probe_only_' if stage=='pilot_centralized_sampling' else 'probe_')+stage+'_FINISHED.json')).exists() for stage in stages)
  if ready and (HERE/'EXPANDED_MODEL_FROZEN.receipt.json').exists():
   if not (HERE/'ALL_CHOICES_FROZEN.receipt.json').exists():call('freeze_choices.py')
   if not (HERE/'main_MISSING_CASES.json').exists():call('main_design.py')
   if complete('main'):
    finish('main');write(HERE/'READY_FOR_FINAL_ANALYSIS.json',dict(utc=utc(),all_simulation_complete=True));print('ALL EXECUTIONS COMPLETE',flush=True);return
   fill('main',18)
  else:
   # Short probe phases first; heavy fair tuning shares remaining capacity.
   for stage in stages:
    pp=('probe_only_' if stage=='pilot_centralized_sampling' else 'probe_')+stage
    if (HERE/f'{pp}_MISSING_CASES.json').exists():fill(pp,8)
   fill('pilot_prioritized_astar',8)
   for stage in stages:
    tp='tune_'+stage
    if (HERE/f'{tp}_MISSING_CASES.json').exists():fill(tp,18)
  counts={phase:len(list((HERE/phase/'raw').glob('*.json'))) for phase in ['pilot_prioritized_astar','tune_pilot_centralized_sampling','tune_pilot_deterministic','tune_pilot_prioritized_astar','main']}
  write(HERE/'LIVE_STATUS.json',dict(utc=utc(),raw_counts=counts,selections_frozen=(HERE/'ALL_CHOICES_FROZEN.receipt.json').exists(),main_labels_started=(HERE/'main_SUBMISSION.json').exists()));print(json.dumps(counts),flush=True);time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'ORCHESTRATION_ERROR.json',dict(utc=utc(),error=repr(ex),no_automatic_rollout_retry=True));raise

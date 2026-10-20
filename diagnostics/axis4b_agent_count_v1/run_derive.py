from common import *
import fcntl,time
from exact_reuse import derive

def ready(c):
 for key in ['deterministic_source_case','shared_prefix_source_case']:
  if key in c:
   s=c[key]
   if not (HERE/s['phase']/'raw'/f"{s['rollout_uid']}.json").exists():return False
 return True

def cases(phase):
 rows=read(HERE/f'{phase}_MISSING_CASES.json');rows.sort(key=lambda c:(c['arm'].startswith('S'),c['seed'],c['N'],c['state_index'],c['kind'],c['arm']));folder=HERE/phase/'queue_derive';folder.mkdir(parents=True,exist_ok=True);statefile=folder/'state.json'
 while True:
  selected=None;finished=False
  with (folder/'lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);s=read(statefile) if statefile.exists() else dict(next=0,pending=[])
   for i in list(s['pending']):
    if ready(rows[i]):selected=i;s['pending'].remove(i);break
   if selected is None:
    while s['next']<len(rows):
     i=s['next'];s['next']+=1
     if ready(rows[i]):selected=i;break
     s['pending'].append(i)
   finished=selected is None and not s['pending'] and s['next']>=len(rows);write(statefile,s)
  if finished:return
  if selected is None:
   assert not list((HERE/phase).glob('worker_*_FATAL.json')),'Dependency worker failed; no automatic retry';time.sleep(.2);continue
  yield rows[selected]

def main():
 phase=sys.argv[1];index=int(sys.argv[2]);prereg();p=read(HERE/f'{phase}_protocol.json');cache={};count=0;assert read(HERE/'CONSISTENCY_GATE.json')['primitive_passed'];assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists()
 if phase=='target':assert (HERE/'TRANSFER_CHOICES_FROZEN.receipt.json').exists()
 try:
  for original in cases(phase):
   c=original;path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
   if path.exists():continue
   if c.get('deterministic_source_case'):
    src=c['deterministic_source_case'];f=HERE/src['phase']/'raw'/f"{src['rollout_uid']}.json";c=dict(c,deterministic_source=dict(raw_path=str(f),raw_sha256=sha(f)))
   q=derive(c,p,phase)
   if q is None:
    from runtime import execute
    q=execute(c,p,cache,phase)
   write(path,q);journal=cache_writer.append_journal([q],EXP,f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP);count+=1;print(json.dumps(dict(phase=phase,index=index,N=c['N'],arm=c['arm'],completed=count,new_physical_steps=q['new_physical_steps'],derived=q.get('execution_kind'))),flush=True)
 except Exception as ex:write(HERE/phase/f'worker_{index}_FATAL.json',dict(utc=utc(),error=repr(ex),no_physical_retry=True));raise
 write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count))
if __name__=='__main__':main()

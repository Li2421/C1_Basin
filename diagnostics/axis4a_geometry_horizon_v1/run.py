from common import *
import fcntl

def cases(phase):
 rows=read(HERE/f'{phase}_MISSING_CASES.json');q=HERE/phase/'queue';q.mkdir(parents=True,exist_ok=True)
 while True:
  with (q/'cursor').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
  if i>=len(rows):return
  yield rows[i]
def main():
 phase=sys.argv[1];index=int(sys.argv[2]);prereg();p=read(HERE/f'{phase}_protocol.json');cache={};count=0
 assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists()
 if phase not in ['gate','expert_gate']:assert read(HERE/'CONSISTENCY_GATE.json')['passed']
 for c in cases(phase):
  path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
  if path.exists():continue
  from runtime import execute
  r=execute(c,p,cache,phase);write(path,r);journal=cache_writer.append_journal([r],EXP,f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP);count+=1;print(json.dumps(dict(phase=phase,index=index,completed=count)),flush=True)
 write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count))
if __name__=='__main__':main()

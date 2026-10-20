from common import *
from runtime import execute
import fcntl

def cases(phase):
 rows=read(HERE/f'{phase}_MISSING_CASES.json');q=HERE/phase/'queue';q.mkdir(parents=True,exist_ok=True)
 while True:
  with (q/'cursor').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
  if i>=len(rows):return
  yield rows[i]
def main():
 phase=sys.argv[1];index=int(sys.argv[2]);prereg();assert read(HERE/f'{phase}_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
 if phase=='baseline':assert read(HERE/'CONSISTENCY_GATE.json')['passed']
 elif phase in ['pilot','main']:
  assert read(HERE/'CONSISTENCY_GATE.json')['passed'];assert (HERE/'MIXED_FROZEN.receipt.json').exists()
  if phase=='main':assert (HERE/'CHOICES_FROZEN.receipt.json').exists()
 p=read(HERE/f'{phase}_protocol.json');cache={};count=0
 for c in cases(phase):
  out=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
  if out.exists():continue
  r=execute(c,p,cache,phase);write(out,r);cache_writer.append_journal([r],EXP,f'{phase}_{index}');count+=1;print(json.dumps(dict(phase=phase,index=index,completed=count)),flush=True)
 write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count))
if __name__=='__main__':main()

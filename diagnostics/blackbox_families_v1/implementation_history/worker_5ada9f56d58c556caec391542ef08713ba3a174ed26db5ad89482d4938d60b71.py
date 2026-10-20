from common import *
import fcntl,time
from legacy_runtime import execute
phase=sys.argv[1];index=sys.argv[2];prereg();p=read(HERE/f'{phase}_protocol.json');rows=read(HERE/f'{phase}_MISSING_CASES.json');cache={};q=HERE/phase/'queue';q.mkdir(parents=True,exist_ok=True)
if phase!='gate':assert read(HERE/'REPRODUCTION_GATE.json')['passed']
while True:
 with (q/'cursor').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
 if i>=len(rows):break
 c=rows[i];out=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
 if out.exists():continue
 r=execute(c,p,cache,phase);write(out,r);journal=cache_writer.append_journal([r],EXP,f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');print(json.dumps(dict(index=i,total=len(rows),outcome=r['outcome'],steps=r['physical_steps'])),flush=True)
write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True))

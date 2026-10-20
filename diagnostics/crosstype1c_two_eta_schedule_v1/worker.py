"""Permanent UID claims, official journals, and temporary-job draining."""
from common import *
from generated_execution import execute
from resource_guard import drain
import fcntl,time
def main(phase,index):
    for path,h in read(HERE/f'{phase}_EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(path)==h,path
    p=read(HERE/f'{phase}_protocol.json');missing=read(HERE/f'{phase}_MISSING_CASES.json');cache={};folder=HERE/phase;claims=folder/'claims';claims.mkdir(parents=True,exist_ok=True)
    while not drain():
        with (claims/'cursor').open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);number=int(stream.read().strip() or 0);stream.seek(0);stream.truncate();stream.write(str(number+1));stream.flush()
        if number>=len(missing):break
        c=missing[number];rid=c['rollout_uid'];raw=folder/'raw'/f'{rid}.json'
        if raw.exists():continue
        with rollout_db.connect(True) as db:
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(rid,)).fetchone() is not None:continue
        try:fd=os.open(claims/f'{rid}.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as f:f.write(json.dumps(dict(utc=utc(),worker=index,pid=os.getpid(),case_uid=rid)))
        out=execute(c,p,cache,phase);write(raw,out);journal=cache_writer.append_journal([out],EXP,f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals')
        if out['error'] is not None and out['error']['type'] not in ('CBFSolverError','FloatingPointError'):raise RuntimeError('Unexpected implementation error preserved: '+str(out['error']))
        print(json.dumps(dict(utc=utc(),phase=phase,worker=index,index=number,total=len(missing),steps=out['physical_steps'],unknown=out['numerical_failure'])),flush=True)
    write(folder/f'WORKER_{index}_COMPLETE.json',dict(utc=utc(),temporary_drain=drain(),worker=index))
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

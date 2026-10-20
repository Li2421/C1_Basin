"""Only genuinely missing exact requests; permanent claims prevent reruns."""
from common import *
from mixed_execution import execute
import fcntl,os,time


def main(phase,index):
    for path,digest in read(HERE/f'{phase}_EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(path)==digest,path
    p=read(HERE/f'{phase}_protocol.json');missing=read(HERE/f'{phase}_MISSING_CASES.json');cache={}
    folder=HERE/phase;claims=folder/'claims';claims.mkdir(parents=True,exist_ok=True)
    while True:
        with (claims/'cursor').open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);number=int(stream.read().strip() or 0)
            stream.seek(0);stream.truncate();stream.write(str(number+1));stream.flush()
        if number>=len(missing):break
        c=missing[number]
        uid=c['rollout_uid'];raw=folder/'raw'/f'{uid}.json'
        if raw.exists():continue
        with rollout_db.connect(True) as db:
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(uid,)).fetchone() is not None:continue
        claim=claims/f'{uid}.json'
        try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as stream:stream.write(json.dumps(dict(worker=index,pid=os.getpid(),utc=utc(),case_uid=uid)))
        result=execute(c,p,cache,phase,probe=p['phase']=='probe');write(raw,result)
        journal=cache_writer.append_journal([result],EXP,f'{phase}_{index}')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals')
        if result['error'] is not None and result['error']['type'] not in ('CBFSolverError','FloatingPointError'):
            raise RuntimeError('Unexpected implementation error preserved in raw result: '+str(result['error']))
        print(json.dumps(dict(phase=phase,index=number,total=len(missing),steps=result['physical_steps'],unknown=result.get('numerical_failure'))),flush=True)
    print('Worker finished',phase,index,flush=True)


if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

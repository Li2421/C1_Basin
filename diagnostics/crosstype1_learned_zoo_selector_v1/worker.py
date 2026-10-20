"""Persistent claims; completed/unknown records are never executed again."""
from common import *
from rollout_runtime import execute
from prefix_reuse import try_reuse
import fcntl


def run(phase,index):
    prereg()
    receipt=read(HERE/f'{phase}_EXECUTION_FROZEN.receipt.json')
    for path,digest in receipt['files'].items():assert sha(path)==digest
    p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_MISSING_CASES.json')
    probe=p['phase']=='probe';cache={}
    folder=HERE/phase;queue=folder/'queue';queue.mkdir(parents=True,exist_ok=True)
    while True:
        with (queue/'cursor').open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);i=int(stream.read().strip() or 0)
            stream.seek(0);stream.truncate();stream.write(str(i+1));stream.flush()
        if i>=len(cases):break
        c=cases[i];target=folder/'raw'/f"{c['rollout_uid']}.json"
        if target.exists():continue
        # Protect against an overlapping stage or a previously preserved unknown.
        if not probe:
            with rollout_db.connect(True) as db:
                exists=db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if exists is not None:continue
        claim=queue/f"{c['rollout_uid']}.claim.json"
        if claim.exists():continue
        write(claim,dict(utc=utc(),worker=index,case=c,status='STARTED'))
        result=None if probe else try_reuse(c,p)
        if result is None:result=execute(c,p,cache,phase,probe=probe)
        write(target,result)
        journal=cache_writer.append_journal([result],EXP,f'{phase}_{index}')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals')
        write(claim,dict(utc=utc(),worker=index,status='COMPLETE',raw_path=str(target),raw_sha256=sha(target)))
        print(json.dumps(dict(phase=phase,index=i,total=len(cases),physical_steps=result['physical_steps'],
                              reused=result.get('physical_reuse',False))),flush=True)
    write(folder/f'worker_{index}.json',dict(utc=utc(),complete=True))

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))

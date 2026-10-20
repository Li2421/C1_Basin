"""Static resumable shards: completed/unknown records are always retained."""
from common import *
from actions_c6 import execute
from resource_guard import drain

def run(stage,worker,workers):
    assert not os.environ.get('XLA_FLAGS');receipt=read(HERE/f'{stage}_EXECUTION_FROZEN.receipt.json')
    for p,h in receipt['files'].items():assert sha(p)==h,p
    rows=read(HERE/f'{stage}_EXECUTION_CASES.json');p=read(HERE/f'{stage}_protocol.json');cache={};completed=0
    for i in range(worker,len(rows),workers):
        if drain():break
        c=rows[i];raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
        if raw.exists():continue
        if stage!='probe':
            with rollout_db.connect(True) as db:existing=db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if existing:continue
        out=execute(c,p,cache,stage,probe=stage=='probe');write(raw,out);cache_writer.append_journal([out],EXP,f'{stage}_{worker}');completed+=1
        if completed%25==0:print(json.dumps(dict(utc=utc(),stage=stage,worker=worker,completed=completed,index=i,error=out.get('error'))),flush=True)
    if '_frozen_mean' in cache:cache['_frozen_mean'].close()
    write(HERE/f'{stage}_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),completed=completed,drain=drain()))

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

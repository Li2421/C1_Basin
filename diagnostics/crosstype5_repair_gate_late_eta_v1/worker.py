"""Exclusive genuine-missing claims; only physical records enter journals."""
from common import *
from resource_guard import drain

def run(stage,worker):
    assert not os.environ.get('XLA_FLAGS'),'Historical controller CPU flags must remain unset'
    from execution_v3 import execute
    receipt=read(HERE/f'{stage}_EXECUTION_FROZEN.receipt.json')
    for p,h in receipt['files'].items():assert sha(p)==h,p
    rows=read(HERE/f'{stage}_EXECUTION_CASES.json');protocol=read(HERE/f'{stage}_protocol.json');claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);cache={};completed=0
    order=np.random.default_rng(202610195107).permutation(len(rows))
    for i in order:
        if drain():break
        c=rows[int(i)];rid=c['rollout_uid'];raw=HERE/stage/'raw'/f'{rid}.json'
        if raw.exists():continue
        try:fd=os.open(claims/f'{rid}.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as f:f.write(json.dumps(dict(utc=utc(),worker=worker,pid=os.getpid(),rollout_uid=rid)))
        out=execute(c,protocol,cache,stage);write(raw,out);cache_writer.append_journal([out],EXP,f'{stage}_{worker}');completed+=1
        if completed%25==0:print(json.dumps(dict(utc=utc(),worker=worker,completed=completed,last=rid)),flush=True)
    write(HERE/f'{stage}_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),completed=completed,drain=drain()))
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))

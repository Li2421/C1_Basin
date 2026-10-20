"""Exclusive claims; only actual executions enter the official journal."""
from common import *
from resource_guard import drain
import time

def run(stage,worker):
    from new_execution import execute
    from dynamic_reuse import infer_actual_file
    receipt=read(HERE/f'{stage}_EXECUTION_FROZEN_v2.receipt.json') if (HERE/f'{stage}_EXECUTION_FROZEN_v2.receipt.json').exists() else read(HERE/f'{stage}_EXECUTION_FROZEN.receipt.json')
    for path,h in receipt['files'].items():assert sha(path)==h,path
    rows=read(HERE/f'{stage}_EXECUTION_CASES.json');protocol=read(HERE/f'{stage}_protocol.json');claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);cache={};done=0
    # Shuffle at fixed seed for load balance across long MPC/planning cases.
    order=np.random.default_rng(202610194601).permutation(len(rows))
    # Execute full zero-eta sources first for development training.
    if stage=='B_training':order=sorted(order,key=lambda i:rows[int(i)].get('training_slot')!=26)
    for index in order:
        if drain():break
        c=rows[int(index)];rid=c['rollout_uid'];raw=HERE/stage/'raw'/f'{rid}.json'
        if raw.exists():continue
        try:fd=os.open(claims/f'{rid}.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as f:f.write(json.dumps(dict(worker=worker,utc=utc(),pid=os.getpid(),rollout_uid=rid)))
        alias=infer_actual_file(c,protocol,stage)
        if alias is not None:
            write(HERE/stage/'local_equivalence'/f'{rid}.json',alias);done+=1;continue
        out=execute(c,protocol,cache,stage);write(raw,out);cache_writer.append_journal([out],EXP,f'{stage}_{worker}');done+=1
        if done%25==0:print(json.dumps(dict(utc=utc(),worker=worker,completed=done,last=rid)),flush=True)
    write(HERE/f'{stage}_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),completed=done,drain=drain()))
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))

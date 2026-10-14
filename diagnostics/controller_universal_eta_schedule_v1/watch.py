"""Outcome-blind live queue/filename monitoring until registered endpoint."""
from core import *
import subprocess,time

def snapshot():
    queue=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);cpu=sum(int(x.split('|')[3]) for x in queue.splitlines() if x.split('|')[2]=='RUNNING');gpu=sum(int(x.split('|')[4].split(':')[-1]) for x in queue.splitlines() if x.split('|')[2]=='RUNNING' and 'shard:' in x.split('|')[4]);count=len(list((HERE/'main/raw').glob('*.json')));r=dict(utc=utc(),raw_count=count,target=18688,percent=100*count/18688,own_slurm_cpu=cpu,own_gpu_shards=gpu,resource_cap_ok=cpu<=18 and gpu<=3,queue=queue,endpoint_opened=(HERE/'ENDPOINT_OPENING.receipt.json').exists(),numeric_complete=(HERE/'NUMERICAL_COMPLETE.receipt.json').exists(),finalizer_error=read(HERE/'FINALIZE_ERROR.json') if (HERE/'FINALIZE_ERROR.json').exists() else None,monitor_does_not_read_labels=True)
    present={int(line.split('|')[0].split('_')[1]) for line in queue.splitlines() if line.split('|')[0].startswith('20112_')}
    r['unexpected_worker_exits']=[i for i in range(17) if i not in present and not(HERE/'main'/f'worker_{i}.json').exists()]
    write(HERE/'LIVE_STATUS.json',r)
    with (HERE/'WATCH_HISTORY.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='queue'}),flush=True);return r
if __name__=='__main__':
    while True:
        r=snapshot()
        if r['numeric_complete'] or r['finalizer_error'] or r['unexpected_worker_exits'] or not r['resource_cap_ok']:break
        time.sleep(55)

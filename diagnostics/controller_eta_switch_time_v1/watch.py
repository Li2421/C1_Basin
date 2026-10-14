"""Live progress/resource monitor; does not launch simulations or change labels."""
from core import *
import time,subprocess
from collections import Counter

def snapshot():
    files=list((HERE/'main/raw').glob('*.json'));records=[read(f) for f in files];counts=Counter(r['outcome'] for r in records);profiles=Counter(r['case']['base_controller_name'] for r in records)
    queue=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True)
    cpus=sum(int(line.split('|')[3]) for line in queue.splitlines() if line.split('|')[2]=='RUNNING')
    gpu=sum(int(line.split('|')[4].split(':')[-1]) for line in queue.splitlines() if line.split('|')[2]=='RUNNING' and 'shard:' in line.split('|')[4])
    result=dict(utc=utc(),raw=len(files),target=10240,percent=100*len(files)/10240,outcomes=dict(counts),controllers=dict(profiles),own_slurm_cpu=cpus,own_slurm_gpu_shards=gpu,resource_cap_ok=cpus<=18 and gpu<=3,queue=queue,complete=(HERE/'COMPLETE.json').exists(),finalizer_error=read(HERE/'FINALIZE_ERROR.json') if (HERE/'FINALIZE_ERROR.json').exists() else None)
    write(HERE/'LIVE_STATUS.json',result)
    with (HERE/'WATCH_HISTORY.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='queue'},ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    while True:
        r=snapshot()
        if r['complete'] or r['finalizer_error'] or not r['resource_cap_ok']:break
        time.sleep(55)

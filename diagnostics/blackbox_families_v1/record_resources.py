from common import *
import subprocess
q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);running=[x.split('|') for x in q.splitlines() if x.split('|')[2] in ['RUNNING','CONFIGURING','COMPLETING']];r=dict(utc=utc(),queue=q,allocated_cpu=sum(int(x[3]) for x in running),gpu_shard_jobs=sum('shard' in x[4] for x in running));p=HERE/'RESOURCE_SAMPLES.jsonl'
with p.open('a') as f:f.write(json.dumps(r)+'\n')
print(r['allocated_cpu'],r['gpu_shard_jobs'])

from core import *
import subprocess
def get(cmd):
 r=subprocess.run(cmd,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
r=dict(time=time.time(),local=time.strftime('%Y-%m-%d %H:%M:%S %Z'),gpu=get(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']),ram=get(['free','-m']),jobs=get(['squeue','-h','-o','%i|%u|%T|%j']),allocation_max_gpu_shards=2,allocation_max_cpu_threads=4,allocation_ram_gib=16)
with (H/'resource_observations.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')

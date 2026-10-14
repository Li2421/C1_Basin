from core import *
import subprocess
def main():
 sq=subprocess.run(['squeue','-h','-u','zhihan','-o','%i|%t|%C|%b|%j'],capture_output=True,text=True,check=True).stdout.splitlines();running=[x for x in sq if '|R|' in x];cpu=sum(int(x.split('|')[2]) for x in running if 'shard:' not in x);gpu=sum('shard:' in x for x in running);resident=subprocess.run(['ps','-eo','pid,pcpu,comm','--sort=-pcpu'],capture_output=True,text=True,check=True).stdout.splitlines()[:40]
 total_cpu=sum(int(x.split('|')[2]) for x in running)
 f=HERE/'RESOURCE_USAGE.jsonl';entry=dict(utc=utc(),local=datetime.now().astimezone().isoformat(),running_cpu_shards=cpu,running_gpu_shards=gpu,total_allocated_cpus=total_cpu,running_jobs=running,resident_top=resident);f.open('a').write(json.dumps(entry)+'\n');print(json.dumps({k:entry[k] for k in ['utc','running_cpu_shards','running_gpu_shards','total_allocated_cpus']}))
if __name__=='__main__':main()

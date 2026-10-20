from common import *
import subprocess,time,signal
G=guard()
def main():
 dest=HERE/'RESOURCE_AUDIT.jsonl';peak_cpu=0;peak_gpu=0
 while not (HERE/'DELIVERY.receipt.json').exists():
  q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);cpu=0;gpu=0
  for line in q.splitlines():
   job,name,status,cpus,gres=line.split('|')
   if status not in ['RUNNING','COMPLETING','CONFIGURING']:continue
   if 'shard' in gres:
    import re
    match=re.search(r'shard(?::[^,:]+)?:(\d+)',gres);gpu+=int(match.group(1)) if match else 1
   else:cpu+=int(cpus)
  peak_cpu=max(peak_cpu,cpu);peak_gpu=max(peak_gpu,gpu)
  with dest.open('a') as f:f.write(json.dumps(dict(utc=utc(),cpu=cpu,gpu_shards=gpu,queue=q,within_cap=cpu<=18 and gpu<=2))+'\n')
  if cpu>18 or gpu>2:
   # Preserve in-flight states by pausing our entire Slurm step process groups.
   own=[line.split('|')[0] for line in q.splitlines() if '|r30e_' in line and '|RUNNING|' in line]
   subprocess.run(['scontrol','suspend',','.join(own[:max(cpu-18,1)])],check=True);write(HERE/'RESOURCE_EXCEPTION.json',dict(utc=utc(),cpu=cpu,gpu_shards=gpu,queue=q,action='suspended in-flight jobs without retry'))
  time.sleep(15)
 write(HERE/'RESOURCE_SUMMARY.json',dict(utc=utc(),peak_cpu=peak_cpu,peak_gpu_shards=peak_gpu,monitor_start='after pilot, during tuning; earlier submission/worker guards separately recorded',guard=guard()))
if __name__=='__main__':main()

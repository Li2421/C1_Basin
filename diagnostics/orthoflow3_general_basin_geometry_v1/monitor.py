#!/usr/bin/env python3
import subprocess, re, json, time
from pathlib import Path
HERE=Path(__file__).parent
def call(cmd):return subprocess.run(cmd,text=True,capture_output=True,check=False).stdout.strip()
def main():
 jobs=call(['squeue','-h','-o','%i|%u|%T|%j']).splitlines();gpu=call(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])
 processes=[]
 for line in gpu.splitlines():
  try:pid,mem=line.split(',');processes.append((int(pid),int(mem)))
  except ValueError:continue
 rss=0
 for pid,_ in processes:
  val=call(['ps','-p',str(pid),'-o','rss=']);rss+=int(val or 0)
 progress=[]
 for p in sorted((HERE/'logs').glob('877_*.out')):
  lines=p.read_text().splitlines();r=None
  for line in reversed(lines):
   try:z=json.loads(line)
   except ValueError:continue
   if 'phase_total' in z:r=z;break
  if r:progress.append(dict(shard=p.stem,**r))
 meminfo={a.split(':')[0]:int(a.split()[1]) for a in open('/proc/meminfo') if a.startswith(('MemAvailable:','MemTotal:'))}
 record={'timestamp':time.time(),'jobs':jobs,'gpu_process_count':len(processes),'gpu_memory_MiB':sum(m for _,m in processes),'gpu_process_RSS_MiB':rss/1024,'RAM_available_GiB':meminfo['MemAvailable']/2**20,'progress':progress}
 with open(HERE/'resource_snapshots.jsonl','a') as f:f.write(json.dumps(record)+'\n')
 total=sum(r['phase_total'] for r in progress);done=sum(r['new'] for r in progress)
 print(json.dumps({'active_jobs':jobs,'common_new_done':done,'common_new_total':total,'fraction':done/total if total else None,'gpu_MiB':record['gpu_memory_MiB'],'RSS_MiB':record['gpu_process_RSS_MiB']}))
if __name__=='__main__':main()

from common import *
import subprocess,time,re
G=guard()
def main():
 peak=0;gpupeak=0;violations=0
 while not (HERE/'STOP_RESOURCE_WATCH').exists():
  q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True);cpu=0;gpu=0
  for line in q.splitlines():
   job,name,status,cores,gres=line.split('|')
   if status not in ['RUNNING','COMPLETING','CONFIGURING']:continue
   if 'shard' in gres:
    m=re.search(r'shard(?::[^,:]+)?:(\d+)',gres);gpu+=int(m.group(1)) if m else 1
   else:cpu+=int(cores)
  peak=max(peak,cpu);gpupeak=max(gpupeak,gpu);violations+=int(cpu>18 or gpu>2)
  with (HERE/'RESOURCE_AUDIT.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),cpu=cpu,gpu_shards=gpu,squeue=q,within_cap=cpu<=18 and gpu<=2))+'\n')
  time.sleep(5)
 write(HERE/'RESOURCE_SUMMARY.json',dict(utc=utc(),peak_cpu=peak,peak_gpu_shards=gpupeak,violations=violations,start_scope='During supplement; replay/diagnostic protected by saved submission and entry guards.',guard=guard()))
if __name__=='__main__':main()

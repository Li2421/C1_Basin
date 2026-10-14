from core import *
import subprocess
from datetime import timedelta
def main(k):
 labels_allowed();assert read(HERE/'CONSISTENCY_GATE.json')['passed'];now=datetime.now().astimezone();assert now.date().isoformat()=='2026-10-14' and now.hour*60+now.minute<=450,'Wave cannot start after 07:30 UTC+8 on 2026-10-14'
 rows=read(HERE/f'wave{k}_MISSING_CASES.json');assert len(rows)<=14336
 sq=subprocess.run(['squeue','-h','-u','zhihan','-o','%i|%t|%C|%b|%j'],capture_output=True,text=True,check=True).stdout.splitlines()
 ps=subprocess.run(['ps','-eo','pid,pcpu,comm'],capture_output=True,text=True,check=True).stdout
 running=[x for x in sq if '|R|' in x];other_cpu=sum(int(x.split('|')[2]) for x in running if 'r15_wave' not in x and 'r15_guard' not in x);other_gpu=sum('shard:' in x for x in running)
 cap=24 if now.hour*60+now.minute<510 else 18
 assert other_cpu+18<=cap,(other_cpu,cap,running)
 assert other_gpu<= (6 if cap==24 else 2)
 forecast=None
 if k==4:
  prev=[read(HERE/f'wave{i}_VERIFY.json') for i in (1,2,3)];mean=sum(x['mean_seconds'] for x in prev)/3;forecast=mean*len(rows)/18+900
  assert now+timedelta(seconds=forecast)<now.replace(hour=10,minute=0,second=0,microsecond=0),'Wave4 forecast misses 10:00'
 d=HERE/f'wave{k}';d.mkdir(exist_ok=True)
 write(d/'START.receipt.json',dict(utc=utc(),wave=k,requests=len(rows),other_cpu_shards=other_cpu,other_gpu_shards=other_gpu,planned_cpu_shards=18,cpu_cap=cap,forecast_seconds=forecast,squeue=sq,top_processes=ps.splitlines()[:35],prereg_sha256=sha(HERE/'PREREGISTRATION.md'),r5_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),r6_receipt_sha256=sha(HERE/'R6_FROZEN.receipt.json')))
 print(k,len(rows),other_cpu,cap)
if __name__=='__main__':main(int(sys.argv[1]))

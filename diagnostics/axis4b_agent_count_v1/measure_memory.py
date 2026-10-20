from core import *
import subprocess,time
previous=read(HERE/'MEMORY_PEAK_SUMMARY.json') if (HERE/'MEMORY_PEAK_SUMMARY.json').exists() else {};started=previous.get('started_utc',utc());maximum=previous.get('peak_single_worker_rss_kib',0);peak_rows=previous.get('peak_sample',[]);samples=previous.get('samples',0)
while True:
 rows=[]
 for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,rss=,args='],text=True).splitlines():
  p=line.strip().split(None,2)
  if len(p)==3 and any(str(HERE/name)+' ' in p[2] for name in ['run.py','run_derive.py']) and 'python -B' in p[2] and 'bash -c' not in p[2]:rows.append(dict(pid=int(p[0]),rss_kib=int(p[1]),worker=p[2].rsplit(' ',2)[-2:]))
 nowmax=max([r['rss_kib'] for r in rows],default=0)
 if nowmax>maximum:maximum=nowmax;peak_rows=rows
 samples+=1;write(HERE/'MEMORY_PEAK_SUMMARY.json',dict(started_utc=started,utc=utc(),peak_single_worker_rss_kib=maximum,peak_sample=peak_rows,samples=samples,scope='Observed pilot/source/target/expert worker processes, includes cached models acrossN; not a perN memory benchmark'))
 if (HERE/'ALL_LABELS_COMPLETE.receipt.json').exists():seal('MEMORY_MONITOR_FINISHED.receipt.json',[HERE/'MEMORY_PEAK_SUMMARY.json'],sampling_seconds=10);break
 time.sleep(10)

#!/usr/bin/env python3
from core import *
import subprocess
def main():
 name,job=sys.argv[1:3];start=time.time();cost=json.load(open(H/f'rounds/{name}/cost_estimate.json'));last=-1
 while True:
  q=subprocess.run(['squeue','-h','-o','%i|%u|%T|%j'],capture_output=True,text=True,check=True).stdout.strip().splitlines();active=[s for s in q if s.split('|')[0].split('_')[0]==job]
  complete=list((H/f'raw/{name}').glob('shard*_runtime.json'));n=0
  for p in (H/f'runs/{name}').glob('shard*/raw/pilot_rollouts.jsonl'):
   with p.open() as f:n+=sum(1 for l in f if l.endswith('\n'))
  data=dict(time=time.time(),round=name,job_id=job,live_shards=len(active),completed_shards=len(complete),new_raw_records=n,requested_maximum=cost['maximum_new_continuations'],other_user_jobs=[s for s in q if '|zhihan|' not in s],elapsed_monitor_seconds=time.time()-start)
  dump('live_progress.json',data)
  if n!=last:
   with (H/'progress_samples.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
   last=n
  if data['other_user_jobs']:print(json.dumps(dict(status='OTHER_USER_JOB_DETECTED',detail=data)),flush=True);return
  if not active:
   print(json.dumps(dict(status='JOB_ENDED',detail=data)),flush=True);return
  if time.time()-start>3*cost['estimated_wall_seconds']:print(json.dumps(dict(status='COST_ANOMALY',detail=data)),flush=True);return
  time.sleep(20)
if __name__=='__main__':main()

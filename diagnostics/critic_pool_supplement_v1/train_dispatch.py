from common import *
from schedule import jobs,quota,pressure_limit,cancel,log
import subprocess

def run():
 if(HERE/'TRAIN_COMPLETE.json').exists():return
 tasks=read(HERE/'TRAIN_TASKS.json');retries=0;retry_jobs=set();seen={};limit=3600
 def finished(t):
  folder=HERE/'fits'/t['id']/f"seed{t['seed']}";return(folder/'FIT.receipt.json').exists()and(folder/'PREDICTIONS.receipt.json').exists()
 while True:
  done=sum(finished(t)for t in tasks)
  if done==len(tasks):break
  fatal=list((HERE/'TRAIN_FATAL').glob('*.json'))
  if fatal:raise RuntimeError(('TRAIN_IMPLEMENTATION_FAILURE',str(fatal[0])))
  rows=jobs();ours=[r for r in rows if r['name']=='xcs_train'];live={r['id']for r in rows};cap=min(14,pressure_limit(rows,'train',quota(rows)))
  for p in(HERE/'TRAIN_QUEUE').glob('*'):
   if not p.name.isdigit()or not(p/'OWNER.json').exists()or(p/'DONE.json').exists():continue
   owner=read(p/'OWNER.json')
   if owner['job']in live or finished(owner['task']):continue
   proc=subprocess.run(['scontrol','show','job',owner['job'],'-o'],capture_output=True,text=True)
   if owner['job']not in retry_jobs and any(x in proc.stdout for x in ['JobState=FAILED','JobState=NODE_FAIL','JobState=OUT_OF_MEMORY','JobState=TIMEOUT','JobState=PREEMPTED']):retries+=1;retry_jobs.add(owner['job'])
   p.rename(p.with_name(p.name+'_ended_'+str(time.time_ns())))
  if retries>3:raise RuntimeError('TRAIN_INFRASTRUCTURE_RETRIES_EXCEEDED')
  for r in ours:
   seen.setdefault(r['id'],time.time())
   if r['state']=='PENDING'and time.time()-seen[r['id']]>900 and sum(x['state']=='RUNNING'for x in ours)<cap:cancel(r['id']);limit=max(600,limit//2);log('train_pending_shorten',job=r['id'],seconds=limit)
  for r in [r for r in ours if r['state']=='RUNNING'][cap:]:cancel(r['id'],True)
  count=max(0,min(cap-len(ours),len(tasks)-done-len(ours)))
  if count:
   job=subprocess.check_output(['sbatch','--parsable','--job-name=xcs_train',f'--array=0-{count-1}',f'--time={limit//3600:02d}:{limit%3600//60:02d}:00',f'--export=ALL,XDT_TIME_LIMIT_S={limit}',str(HERE/'train_worker.sbatch')],text=True).strip().split(';')[0]
   observed=[]
   for _ in range(20):
    observed=[r for r in jobs()if r['id'].split('_')[0]==job]
    if observed and all(r['state']=='RUNNING'for r in observed):break
    time.sleep(1)
   log('training_submitted_RUNNING_readback',job=job,count=count,states=observed)
  write(HERE/'PROGRESS.json',dict(utc=utc(),stage='training',done=done,total=len(tasks),worker_cap=cap,retries=retries,validation_labels_read=False))
  (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{utc()}：critic训练与无标签预测{done}/{len(tasks)}，并发上限{cap}（≤14），18CPU总限额；未读取验证标签，无生成器。\n')
  time.sleep(10)
 for r in jobs():
  if r['name']=='xcs_train':cancel(r['id'],True)
 for _ in range(30):
  if not any(r['name']=='xcs_train'for r in jobs()):break
  time.sleep(1)
 assert not any(r['name']=='xcs_train'for r in jobs());write(HERE/'TRAIN_COMPLETE.json',dict(utc=utc(),status='COMPLETE',fits=len(tasks),infrastructure_retries=retries))
if __name__=='__main__':run()

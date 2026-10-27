from common import *
from schedule import jobs,quota,pressure_limit,cancel,log
import subprocess

def run(stage):
 if(HERE/f'{stage}_COMPLETE.json').exists():return
 assert(HERE/'HETEROGENEITY_CHECK.json').exists();tasks=read(HERE/('TRAIN_INNER_TASKS.json'if stage=='train_inner'else'TRAIN_OUTER_TASKS.json'));retries=0;starts={};limit=3600
 def finished(t):return(HERE/'folds'/t['fold']/t['mode']/f"seed{t['seed']}"/'FIT.receipt.json').exists()
 while True:
  done=sum(finished(t)for t in tasks)
  if done==len(tasks):break
  fatal=[p for p in(HERE/'TRAIN_FATAL').glob('*.json')if read(p)['stage']==stage]
  if fatal:raise RuntimeError(('TRAIN_IMPLEMENTATION_FAILURE',str(fatal[0])))
  rows=jobs();ours=[r for r in rows if r['name']=='xhp_'+stage];live={r['id']for r in rows};cap=pressure_limit(rows,stage,quota(rows))
  for p in(HERE/'TRAIN_QUEUE'/stage).glob('*'):
   if not p.name.isdigit()or not(p/'OWNER.json').exists()or(p/'DONE.json').exists():continue
   owner=read(p/'OWNER.json')
   if owner['job']in live or finished(owner['task']):continue
   out=subprocess.run(['scontrol','show','job',owner['job'],'-o'],capture_output=True,text=True)
   if any(x in out.stdout for x in ['JobState=FAILED','JobState=TIMEOUT','JobState=NODE_FAIL','JobState=OUT_OF_MEMORY','JobState=PREEMPTED']):retries+=1
   p.rename(p.with_name(p.name+'_ended_'+str(time.time_ns())))
  if retries>3:raise RuntimeError('TRAIN_INFRA_RETRY_LIMIT')
  for r in ours:starts.setdefault(r['id'],time.time())
  for r in ours:
   if r['state']=='PENDING'and time.time()-starts[r['id']]>900 and sum(x['state']=='RUNNING'for x in ours)<cap:
    cancel(r['id']);limit=max(600,limit//2);log('train_pending_shorten',stage=stage,job=r['id'],seconds=limit)
  for r in [r for r in ours if r['state']=='RUNNING'][cap:]:cancel(r['id'],True)
  free=max(0,min(cap-len(ours),len(tasks)-done-len(ours)))
  if free:
   job=subprocess.check_output(['sbatch','--parsable',f'--job-name=xhp_{stage}',f'--array=0-{free-1}',f'--time={limit//3600:02d}:{limit%3600//60:02d}:00',f'--export=ALL,XDT_STAGE={stage},XDT_TIME_LIMIT_S={limit}',str(HERE/'train_worker.sbatch')],text=True).strip().split(';')[0]
   observed=[]
   for _ in range(20):
    observed=[r for r in jobs()if r['id'].split('_')[0]==job]
    if observed and all(r['state']=='RUNNING'for r in observed):break
    time.sleep(1)
   log('training_submitted_RUNNING_readback',stage=stage,job=job,count=free,states=observed)
  write(HERE/'PROGRESS.json',dict(utc=utc(),stage=stage,done=done,total=len(tasks),worker_cap=cap,retries=retries))
  (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{utc()}：{stage}，完成{done}/{len(tasks)}个原v2小网络拟合；源场景内部验证与超参数不改。\n')
  time.sleep(10)
 for r in jobs():
  if r['name']=='xhp_'+stage:cancel(r['id'],True)
 for _ in range(30):
  if not any(r['name']=='xhp_'+stage for r in jobs()):break
  time.sleep(1)
 assert not any(r['name']=='xhp_'+stage for r in jobs())
 write(HERE/f'{stage}_COMPLETE.json',dict(utc=utc(),status='COMPLETE',fits=len(tasks),infrastructure_retries=retries))

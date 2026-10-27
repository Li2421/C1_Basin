from experiment import *
import ast,re,difflib
from atomic_write import write
policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json')
policy.update(utc=utc(),authority='Latest explicit user correction:24 CPU now,18 CPU Oct23 09:00 Asia/Singapore;2 GPU shares unchanged',before=dict(total_cpu_shares=24,total_gpu_shards=2),after=dict(total_cpu_shares=18,total_gpu_shards=2))
write(HERE/'RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json',policy)
request=read(HERE/'CPU_RELEASE_USER_SCHEDULED_20261023.json');request.update(utc=utc(),before_cpu=24,after_cpu=18,correction='User corrected previous mistaken18->12 interpretation; latest authority24->18')
write(HERE/'CPU_RELEASE_USER_SCHEDULED_20261023.json',request)
module=HERE/'resources_corrections_release_20261023.py'
s=module.read_text().replace('GPU_SHARD_LIMIT = 5','GPU_SHARD_LIMIT = 2')
s=s.replace("reduced=at >= datetime.fromisoformat(policy['graceful_drain_at_utc'])","reduced=at >= datetime.fromisoformat(policy['release_at_utc'])\n    draining=at >= datetime.fromisoformat(policy['graceful_drain_at_utc'])")
s=s.replace("release_at_utc=policy['release_at_utc'],reduced=reduced)","release_at_utc=policy['release_at_utc'],reduced=reduced,\n                admission_cpu_limit=policy['after']['total_cpu_shares'] if draining else selected['total_cpu_shares'],\n                graceful_drain_at_utc=policy['graceful_drain_at_utc'],\n                deadline_fallback_at_utc=policy['deadline_fallback_at_utc'])")
s=s.replace('cpu_limit-LOCAL_CPU_RESERVE-used_cpu',"quota['admission_cpu_limit']-LOCAL_CPU_RESERVE-used_cpu")
module.write_text(s)
mapping={
'resources_corrections_20261021':'resources_corrections_release_20261023',
'fine_dispatch_correction_recovery_20261021':'fine_dispatch_correction_release_20261023',
'coordinator_correction_recovery_20261021':'coordinator_correction_release_20261023',
'scheduler_watchdog_correction_owner_guard_20261021':'scheduler_watchdog_correction_release_20261023',
'progress_correction_20261021':'progress_correction_release_20261023',
'scene_learning_scheduler_revised_20261021':'scene_learning_scheduler_revised_release_20261023',
'critic_jobs_revised_20261021':'critic_jobs_revised_release_20261023',
'analysis_submit_cached':'analysis_submit_release_20261023',
'diagnosis_backfill':'diagnosis_backfill_release_20261023',
'heartbeat_corrections_20261021':'heartbeat_corrections_release_20261023',
'resource_guard_corrections_20261021':'resource_guard_corrections_release_20261023'}
pattern=re.compile(r'\b('+ '|'.join(map(re.escape,mapping)) + r')\b')
files=[module,HERE/'RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json',HERE/'CPU_RELEASE_USER_SCHEDULED_20261023.json',Path(__file__)]
diffs=[]
for old,new in mapping.items():
 if old=='resources_corrections_20261021':continue
 source=HERE/(old+'.py');target=HERE/(new+'.py');before=source.read_text();after=pattern.sub(lambda m:mapping[m.group()],before)
 if old in ('analysis_submit_cached','diagnosis_backfill'):
  after=after.replace('from resources_released import','from resources_corrections_release_20261023 import')
 if old=='progress_correction_20261021':
  after=after.replace('old_circle_workers_may_still_drain=True',"old_circle_workers_may_still_drain=not (HERE/'OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json').exists()")
 if old=='heartbeat_corrections_20261021':after=after.replace('RESOURCE_SCHEDULE_CORRECTED.json','RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json')
 if old=='scheduler_watchdog_correction_owner_guard_20261021':after=after.replace('CORRECTION_WATCHDOG_OWNER_GUARD_20261021.receipt.json','CPU_RELEASE_CODE_20261023.receipt.json')
 if old=='coordinator_correction_recovery_20261021':after=after.replace("def main():\n", "def main():\n verify_receipt('CPU_RELEASE_CODE_20261023.receipt.json')\n")
 if old=='resource_guard_corrections_20261021':
  after=after.replace('CORRECTION_RESOURCE_CODE_20261021.receipt.json','CPU_RELEASE_CODE_20261023.receipt.json')
  after=after.replace("state=snapshot();cpu=state['scheduled_cpus'];gpu=state['gpu_shards'];yielded=[]", "state=snapshot();cpu=state['scheduled_cpus'];gpu=state['gpu_shards'];yielded=[]\n        target=state['admission_cpu_limit']\n        deadline=datetime.now(timezone.utc)>=datetime.fromisoformat(state['deadline_fallback_at_utc'])")
  after=after.replace("cpu+1<=state['cpu_limit'] and gpu<=2", "cpu+1<=target and gpu<=state['gpu_shard_limit']")
  after=after.replace("if job['state']=='PENDING':args=", "if job['state']=='PENDING' or deadline:args=")
  after=after.replace("completed_episode_boundary=job['state']!='PENDING'", "completed_episode_boundary=job['state']!='PENDING' and not deadline,deadline_fallback=deadline")
  after=after.replace('hard_cancelled_running_rollouts=0',"hard_cancelled_running_rollouts=sum(r['deadline_fallback'] and r['job']['state']!='PENDING' for r in yielded)")
 ast.parse(after);target.write_text(after)
 diff=HERE/(new+'.py.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(source),tofile=str(target))))
 diffs.append(dict(source=str(source),source_sha256=sha(source),target=str(target),target_sha256=sha(target)))
 files.extend([target,diff])
write(HERE/'CPU_RELEASE_SOURCE_DERIVATIONS_20261023.json',diffs);files.append(HERE/'CPU_RELEASE_SOURCE_DERIVATIONS_20261023.json')
from resources_corrections_release_20261023 import limits
checks=[]
for when,expected,admit in [('2026-10-21T17:00:00+00:00',24,24),('2026-10-22T23:30:00+00:00',24,18),('2026-10-23T00:59:00+00:00',24,18),('2026-10-23T01:00:00+00:00',18,18)]:
 value=limits(datetime.fromisoformat(when));assert value['cpu_limit']==expected and value['admission_cpu_limit']==admit and value['gpu_shard_limit']==2,value
 checks.append(dict(at=when,**value))
write(HERE/'CPU_RELEASE_BOUNDARY_CHECKS_20261023.json',checks);files.append(HERE/'CPU_RELEASE_BOUNDARY_CHECKS_20261023.json')
seal('CPU_RELEASE_CODE_20261023.receipt.json',files,latest_user_authority='24->18 CPU Oct23 09:00 Asia/Singapore',gpu_shards=2,scientific_workers_unchanged=True,owner_lock_race_guard_preserved=True)
print('Prepared and sealed',len(files),'files',flush=True)

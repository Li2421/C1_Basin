"""Create immutable scheduling-only variants for the user's idle-CPU authorization."""
from pathlib import Path
import sys,os,json,re,difflib,ast
p=Path(__file__).resolve().parent
os.sched_setaffinity(0,{json.loads((p/'LOCAL_CPU_GUARD_STATUS.json').read_text())['local_cpu']})
sys.path.insert(0,str(p))
from experiment import *
from atomic_write import write
from registry import verify_receipt
verify_receipt('RESOURCE_RELEASE_CORRECTION.receipt.json')
policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTED.json')
policy.update(utc=utc(),authority='User: if nobody else is using CPU, use all CPU; latest authorization overrides CPU release only',supersedes_cpu_policy_sha256=sha(HERE/'RESOURCE_SCHEDULE_CORRECTED.json'),idle_total_cpu_shares=24,busy_total_cpu_shares=18,other_gpu_policy_unchanged=True)
policy['after']=dict(total_cpu_shares=24,total_gpu_shards=2)
write(HERE/'RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json',policy)
names=['coordinator_flat_static','downstream_flat_static','fine_dispatch_flat_static','scene_learning_scheduler_flat_static','scheduler_watchdog_flat_static','optional_delivery_flat_static','optional_fine_dispatch_flat_static','critic_jobs_released','diagnosis_backfill','analysis_submit_cached','optional_diagnosis_backfill','submit_active','status_released','heartbeat_released']
mapping={name:name+'_allcpu' for name in names}
mapping['resources_released']='resources_idle_cpu'
mapping['coordinator_pipelined']=mapping['coordinator_flat_static']
pattern=re.compile(r'\b('+'|'.join(sorted(mapping,key=len,reverse=True))+r')\b')
changes=[]
for name in names:
    original=HERE/(name+'.py');dest=HERE/(mapping[name]+'.py')
    source=original.read_text();updated=pattern.sub(lambda m:mapping[m.group()],source)
    if name=='status_released':
        updated=updated.replace('RESOURCE_SCHEDULE_CORRECTED.json','RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json')
        updated=updated.replace('已按用户纠正释放6 CPU和3 GPU份额：全用户总上限18 CPU/2 GPU份额，预留1个本地CPU，最多17个调度CPU；独立resource_guard_released.py执行新上限。','依用户最新授权，无其他用户计算作业时使用24 CPU（23调度+1本地），其他用户提交计算作业时恢复18 CPU；GPU仍为2份额。')
    if name=='heartbeat_released':updated=updated.replace('RESOURCE_SCHEDULE_CORRECTED.json','RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json')
    assert not dest.exists(),dest
    ast.parse(updated);dest.write_text(updated)
    diff=HERE/(dest.name+'.diff');diff.write_text(''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile=original.name,tofile=dest.name)))
    changes.append(dict(original=str(original),original_sha256=sha(original),new=str(dest),new_sha256=sha(dest),diff=str(diff)))
source=(HERE/'resources_released.py').read_text().replace('RESOURCE_SCHEDULE_CORRECTED.json','RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json')
needle="    quota=limits()\n    cpu_limit=quota['cpu_limit'];gpu_limit=quota['gpu_shard_limit']"
replacement="""    quota=limits()
    other_raw=subprocess.check_output(['squeue','-h','-r','-o','%u|%i|%T|%C'],text=True)
    other_jobs=[]
    for line in other_raw.splitlines():
        user,job,state,cpu=line.split('|')
        if user!=os.environ['USER']:
            other_jobs.append(dict(user=user,job_id=job,state=state,cpus=int(cpu)))
    quota['cpu_limit']=18 if other_jobs else 24
    quota['other_user_compute_jobs']=other_jobs
    quota['cpu_authority']='Latest user idle CPU authorization; GPU remains2 shares'
    cpu_limit=quota['cpu_limit'];gpu_limit=quota['gpu_shard_limit']"""
assert needle in source
source=source.replace(needle,replacement)
ast.parse(source);(HERE/'resources_idle_cpu.py').write_text(source)
write(HERE/'IDLE_CPU_AUTHORIZATION_SOURCE_CHANGES.json',dict(utc=utc(),changes=changes,worker_and_scientific_sources_unchanged=True,new_rollouts=0,resource_policy_only=True))
seal('IDLE_CPU_AUTHORIZATION_CODE.receipt.json',[HERE/'prepare_allcpu_authorization.py',HERE/'resources_idle_cpu.py',HERE/'RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json',HERE/'IDLE_CPU_AUTHORIZATION_SOURCE_CHANGES.json']+[Path(r[k]) for r in changes for k in ('new','diff')],previous_sealed_policy_preserved=True,science_unchanged=True,cpu_idle=24,cpu_other_users=18,gpu_shards=2)
print('Created14 scheduling variants; previous sealed resource policy preserved')

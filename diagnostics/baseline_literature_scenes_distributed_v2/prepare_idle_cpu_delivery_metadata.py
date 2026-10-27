"""Keep final handoff and local affinity status consistent with latest CPU authority."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import difflib,ast
verify_receipt('IDLE_CPU_AUTHORIZATION_CODE.receipt.json')
changes=[]
def add(old,new,replacements):
 source=(HERE/old).read_text();updated=source
 for a,b in sorted(replacements,key=lambda row:len(row[0]),reverse=True):
  assert a in updated,(old,a)
  updated=updated.replace(a,b)
 ast.parse(updated)
 if (HERE/new).exists():assert (HERE/new).read_text()==updated
 else:(HERE/new).write_text(updated)
 diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile=old,tofile=new)))
 changes.append(dict(original=old,original_sha256=sha(HERE/old),new=new,new_sha256=sha(HERE/new),diff=diff.name))
common=[('coordinator_flat_static_allcpu.py','coordinator_flat_static_allcpu_delivery.py'),('downstream_flat_static_allcpu.py','downstream_flat_static_allcpu_delivery.py')]
add('coordinator_flat_static_allcpu.py','coordinator_flat_static_allcpu_delivery.py',[common[1]])
add('downstream_flat_static_allcpu.py','downstream_flat_static_allcpu_delivery.py',[('optional_delivery_flat_static_allcpu.py','optional_delivery_flat_static_allcpu_delivery.py')])
add('optional_delivery_flat_static_allcpu.py','optional_delivery_flat_static_allcpu_delivery.py',[('final_reports_flat_static.py','final_reports_flat_static_allcpu_delivery.py')])
add('scheduler_watchdog_flat_static_allcpu.py','scheduler_watchdog_flat_static_allcpu_delivery.py',[(a.removesuffix('.py'),b.removesuffix('.py')) for a,b in common])
replacements=[('coordinator_flat_static.py','coordinator_flat_static_allcpu_delivery.py'),('downstream_flat_static.py','downstream_flat_static_allcpu_delivery.py'),('fine_dispatch_flat_static.py','fine_dispatch_flat_static_allcpu.py'),('optional_fine_dispatch_flat_static.py','optional_fine_dispatch_flat_static_allcpu.py'),('scene_learning_scheduler_flat_static.py','scene_learning_scheduler_flat_static_allcpu.py'),('scheduler_watchdog_flat_static.py','scheduler_watchdog_flat_static_allcpu_delivery.py'),('optional_delivery_flat_static.py','optional_delivery_flat_static_allcpu_delivery.py'),('final_reports_flat_static.py','final_reports_flat_static_allcpu_delivery.py')]
replacements.append(('用户最新资源覆盖 prompt 的旧限额：已纠正释放日期：2026-10-21起已释放6 CPU和3 GPU份额，总计18 CPU份额/2 GPU份额；RESOURCE_SCHEDULE_CORRECTED.json覆盖旧日程。GPU份额为单卡 shard；所有用户作业（含pending）计入，CPU预留1给本地协调/分析。resources_released.py 每次提交前检查；resource_guard_released.py 独立守护，只取消本轮 blv2_ 作业至限额，完整raw/journal保留，未完成基础设施重试。','用户最新资源授权覆盖此前CPU释放安排：其他用户没有计算作业时使用24 CPU份额（23调度+1本地），其他用户提交Slurm计算作业时恢复18 CPU份额；GPU仍保持已释放后的2份额，不恢复为5份额。RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json、IDLE_CPU_AUTHORIZATION_CODE.receipt.json和IDLE_CPU_AUTHORIZATION_DEPLOYED.json记录授权、证据与部署；原密封18/2日程保留。resources_idle_cpu.py每次提交前检查全用户作业（含pending）与其他用户Slurm任务；resource_guard_idle_cpu.py只让本轮多出的运行worker在完整episode落盘后退出，pending可直接取消，已完成raw/journal保留。local_cpu_guard_idle_cpu.py保持本地进程及其全部线程在Slurm未分配的单个物理CPU。'))
add('final_reports_flat_static.py','final_reports_flat_static_allcpu_delivery.py',replacements)
add('local_cpu_guard.py','local_cpu_guard_idle_cpu.py',[('total_user_cpu_cap=18',"total_user_cpu_cap=read(HERE/'RESOURCE_RELEASE_STATUS.json')['snapshot']['cpu_limit']"),('LOCAL_CPU_GUARD_CODE.receipt.json','IDLE_CPU_DELIVERY_METADATA_CODE.receipt.json')])
write(HERE/'IDLE_CPU_DELIVERY_METADATA_CHANGES.json',dict(utc=utc(),changes=changes,changes_only='Final handoff entrypoint/resource prose and local affinity status cap; scheduling entrypoint links',worker_science_unchanged=True))
seal('IDLE_CPU_DELIVERY_METADATA_CODE.receipt.json',[HERE/'prepare_idle_cpu_delivery_metadata.py',HERE/'IDLE_CPU_DELIVERY_METADATA_CHANGES.json']+[HERE/c[k] for c in changes for k in ('new','diff')],worker_science_unchanged=True)
print('Prepared final delivery entrypoints and dynamic resource metadata')

"""Correct the resource deadline without changing any sealed science source."""
from experiment import *
import ast
import difflib
import subprocess
import signal
import time

def main():
    old=read(HERE/'RESOURCE_SCHEDULE.json')
    policy=dict(old,utc=utc(),authority='User corrected the missed six-CPU release; apply 18 CPU and 2 GPU shares immediately, including all user jobs',release_at_local='2026-10-21T09:00:00+08:00',release_at_utc='2026-10-21T01:00:00+00:00',supersedes_original_policy_sha256=sha(HERE/'RESOURCE_SCHEDULE.json'),correction='Original October 22 calendar interpretation was wrong; intended October 21 release was missed')
    write(HERE/'RESOURCE_SCHEDULE_CORRECTED.json',policy)
    mappings={
        'resources.py':('resources_released.py',[("POLICY = HERE/'RESOURCE_SCHEDULE.json'","POLICY = HERE/'RESOURCE_SCHEDULE_CORRECTED.json'")]),
        'submit_reservation_v2.py':('submit_reservation_released.py',[('from resources import','from resources_released import')]),
        'analysis_jobs_current.py':('analysis_jobs_released.py',[('from resources import','from resources_released import')]),
        'critic_jobs.py':('critic_jobs_released.py',[('from resources import','from resources_released import')]),
        'coordinator_current.py':('coordinator_released.py',[
            ('from resources import','from resources_released import'),('from status import','from status_released import'),
            ('from submit_reservation_v2 import','from submit_reservation_released import'),('from analysis_jobs_current import submit','from analysis_jobs_released import submit'),
            ("'downstream_current.py'","'downstream_released.py'")]),
        'downstream_current.py':('downstream_released.py',[
            ('from resources import','from resources_released import'),('from status import','from status_released import'),
            ('from coordinator_current import','from coordinator_released import'),('from critic_jobs import','from critic_jobs_released import'),
            ('from critic_supervisor_current import','from critic_supervisor_released import'),("'optional_delivery.py'","'optional_delivery_released.py'")]),
        'critic_supervisor_current.py':('critic_supervisor_released.py',[
            ('from resources import','from resources_released import'),('from status import','from status_released import'),
            ('from coordinator import','from coordinator_released import'),('from critic_jobs import','from critic_jobs_released import')]),
        'optional_delivery.py':('optional_delivery_released.py',[
            ('from resources import','from resources_released import'),('from status import','from status_released import'),
            ('from coordinator_current import','from coordinator_released import'),("'final_reports.py'","'final_reports_released.py'")]),
        'status.py':('status_released.py',[
            ('from resources import','from resources_released import'),('RESOURCE_SCHEDULE.json','RESOURCE_SCHEDULE_CORRECTED.json'),
            ('资源为全用户总上限24 CPU/5 GPU份额，预留1个本地CPU；2026-10-22 09:00新加坡时间降为18 CPU/2 GPU份额，独立resource_guard.py已启动。','已按用户纠正释放6 CPU和3 GPU份额：全用户总上限18 CPU/2 GPU份额，预留1个本地CPU，最多17个调度CPU；独立resource_guard_released.py执行新上限。')]),
        'resource_guard.py':('resource_guard_released_core.py',[
            ('from resources import','from resources_released import'),('RESOURCE_SCHEDULE.json','RESOURCE_SCHEDULE_CORRECTED.json')]),
        'resource_guard_v2.py':('resource_guard_released.py',[
            ('from resource_guard import','from resource_guard_released_core import'),('RESOURCE_SCHEDULE.json','RESOURCE_SCHEDULE_CORRECTED.json'),("entrypoint='resource_guard_v2.py'","entrypoint='resource_guard_released.py'")]),
        'heartbeat.py':('heartbeat_released.py',[
            ('from resources import','from resources_released import'),('RESOURCE_SCHEDULE.json','RESOURCE_SCHEDULE_CORRECTED.json')]),
        'final_reports.py':('final_reports_released.py',[
            ('coordinator_current.py','coordinator_released.py'),('downstream_current.py','downstream_released.py'),
            ('submit_reservation_v2.py','submit_reservation_released.py'),('resources.py','resources_released.py'),
            ('resource_guard_v2.py','resource_guard_released.py'),('final_reports.py','final_reports_released.py'),
            ('2026-10-22 09:00 Asia/Singapore 前总计24 CPU份额/5 GPU份额，之后18/2。','已纠正释放日期：2026-10-21起已释放6 CPU和3 GPU份额，总计18 CPU份额/2 GPU份额；RESOURCE_SCHEDULE_CORRECTED.json覆盖旧日程。')])
    }
    derived=[]
    for original,(target,replacements) in mappings.items():
        source=(HERE/original).read_text();result=source
        for before,after in replacements:
            assert before in result,(original,before)
            result=result.replace(before,after)
        dest=HERE/target;assert not dest.exists(),dest
        ast.parse(result);dest.write_text(result)
        diff=HERE/(target+'.diff');diff.write_text(''.join(difflib.unified_diff(source.splitlines(True),result.splitlines(True),fromfile=original,tofile=target)))
        derived.append(dict(original=original,original_sha256=sha(HERE/original),target=target,target_sha256=sha(dest),diff=str(diff),diff_sha256=sha(diff)))
    write(HERE/'RESOURCE_RELEASE_DERIVATIONS.json',derived)
    from resources_released import snapshot,limits,require_capacity
    state=snapshot();assert state['cpu_limit']==18 and state['gpu_shard_limit']==2
    assert state['scheduled_cpus']+state['local_cpu_reserve']<=18,state
    denied=False
    try:require_capacity(0,3)
    except AssertionError:denied=True
    assert denied
    write(HERE/'RESOURCE_RELEASE_TESTS.json',dict(utc=utc(),snapshot=state,three_gpu_shards_rejected=True,source_semantics='Operational quotas and report wording only; all controller/physics/analysis science sources unchanged'))
    paths=[HERE/'RESOURCE_SCHEDULE_CORRECTED.json',HERE/'RESOURCE_RELEASE_DERIVATIONS.json',HERE/'RESOURCE_RELEASE_TESTS.json',HERE/'deploy_resource_release_20261021.py']
    paths += [HERE/r['target'] for r in derived]+[Path(r['diff']) for r in derived]
    seal('RESOURCE_RELEASE_CORRECTION.receipt.json',paths,user_correction_effective_immediately=True,total_cpu_limit=18,total_gpu_shards=2,local_cpu_reserve=1,scientific_cases_and_seeds_unchanged=True)
    # End only the old local submission/guard processes. All surviving workers stay.
    frozen=read(HERE/'IMMEDIATE_RESOURCE_RELEASE_20261021.json')['stopped_coordinator_tree']
    old_pids=frozen+[read(HERE/'RESOURCE_GUARD_PID.json')['pid'],read(HERE/'HEARTBEAT_PID.json')['pid']]
    for pid in old_pids:
        try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    time.sleep(1)
    for pid in old_pids:
        try:
            os.kill(pid,0)
            os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    launched={}
    for script,log in [('resource_guard_released.py','resource_guard_released.log'),('coordinator_released.py','coordinator.log'),('heartbeat_released.py','heartbeat.log')]:
        with (HERE/log).open('a') as stream:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        launched[script]=p.pid
    write(HERE/'RESOURCE_RELEASE_RESTART.json',dict(utc=utc(),old_local_processes_ended=old_pids,new_processes=launched,remaining_workers_untouched=True,completed_evidence_preserved=True))
    with (HERE/'ITERATION_LOG.md').open('a') as stream:
        stream.write('\n2026-10-21 用户纠正资源释放：此前把“明早9点”错误排到10月22日，导致未及时释放。立即取消6个本轮worker，改为总18 CPU（17调度+1本地预留）/2 GPU份额。旧错误日程及全部已封存代码保留，RESOURCE_SCHEDULE_CORRECTED.json与RESOURCE_RELEASE_CORRECTION.receipt.json生效。新增released操作入口统一检查新限额；控制器、物理、UID、seed、分析和训练规则不改。完整raw/journal保留，未完成部分按原身份恢复，不记unknown。\n')
    print(dict(snapshot=state,new_processes=launched),flush=True)

if __name__=='__main__':main()

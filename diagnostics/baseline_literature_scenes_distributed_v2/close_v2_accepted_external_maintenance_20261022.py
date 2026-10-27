"""Complete only the already computed task, with explicit accepted audit scope."""
from experiment import *
from registry import verify_receipt
from resources_corrections_release_20261023_quota18_now_20261022 import snapshot
import subprocess,fcntl

def action(entry,log):
    with (HERE/log).open('w') as stream:
        result=subprocess.run([sys.executable,'-B',str(HERE/entry),*(['end'] if entry.startswith('integrity_') else [])],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,(entry,log,result.returncode)

def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('ACCEPTED_EXTERNAL_MAINTENANCE_FINALIZATION_READY_20261022.receipt.json')
    verify_receipt('USER_ACCEPTED_EXTERNAL_MAINTENANCE_AUDIT_EXCEPTION_20261022.receipt.json')
    approval=read(HERE/'USER_ACCEPTED_EXTERNAL_MAINTENANCE_AUDIT_EXCEPTION_20261022.json');assert approval['exception_accepted']
    with (HERE/'CORRECTION_COORDINATOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (HERE/'V2_COMPLETE.json').exists()
        assert not any(j['name'].startswith('blv2_') for j in snapshot()['jobs'])
        write(HERE/'FINALIZATION_PID_20261022.json',dict(utc=utc(),pid=os.getpid(),entrypoint=Path(__file__).name))
        action('integrity_accepted_external_maintenance_20261022.py','FINAL_AUDIT_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.log')
        verify_receipt('INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.receipt.json')
        audit=read(HERE/'INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.json')
        assert audit['passed'] and not audit['strict_passed'] and audit['differences']==approval['accepted_differences'] and not audit['unexpected_differences']
        action('final_reports_accepted_external_maintenance_20261022.py','FINAL_REPORT_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.log')
        for name in ('FINAL_REPORTS.receipt.json','MIGRATION_NUMBERS.receipt.json','OPTIONAL_NUMBERS.receipt.json','DB_POSTFLIGHT.receipt.json','FIGURES.receipt.json'):verify_receipt(name)
        db=read(HERE/'DB_POSTFLIGHT.json');assert db['passed'] and db['own_quarantined_rows']==0 and db['unmerged_complete_journals']==0
        optional=read(HERE/'OPTIONAL_RESULTS.json')['results']
        assert all(v['status']=='complete' and v['counts']['planned']==v['counts']['completed'] for k,v in optional.items() if k.startswith('optional_'))
        assert not any(j['name'].startswith('blv2_') for j in snapshot()['jobs'])
        archived=HERE/'COORDINATOR_ACTION_ERROR_STRICT_MAINTENANCE_AUDIT_20261022.json'
        current=HERE/'COORDINATOR_ACTION_ERROR.json'
        assert current.exists() and not archived.exists()
        current.rename(archived)
        seal('RESOLVED_STRICT_MAINTENANCE_AUDIT_ACTION_ERROR_20261022.receipt.json',[archived,HERE/'USER_ACCEPTED_EXTERNAL_MAINTENANCE_AUDIT_EXCEPTION_20261022.receipt.json',HERE/'INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.receipt.json'],original_strict_failure_preserved=True,resolved_only_by_user_accepted_exception=True)
        state=read(HERE/'STATUS.json');state.update(utc=utc(),phase='complete_with_user_accepted_external_maintenance_exception',resources=snapshot(),legacy_stage_claim_counters_not_final_coverage_proof=True,finalization=dict(science_complete=True,statistics_complete=True,official_merge_complete=True,database_postflight_passed=True,figures_complete=True,strict_source_audit_passed=False,user_accepted_external_maintenance_exception=True,unexpected_source_differences=0,final_report=str(HERE/'REPORT.md')));write(HERE/'STATUS.json',state)
        from progress_supplement_20261022 import tick
        progress=tick();assert progress['all_v2_remaining']==0 and progress['main_remaining']==0
        entry=read(HERE/'ACTIVE_ENTRYPOINT.json');entry.update(utc=utc(),state=state['phase'],final_reports='final_reports_accepted_external_maintenance_20261022.py',finalization_entrypoint=Path(__file__).name,strict_source_audit_passed=False,user_accepted_external_maintenance_exception=True,all_scientific_jobs_ended=True);write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
        write(HERE/'V2_COMPLETE.json',dict(utc=utc(),main_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),source_audit=sha(HERE/'INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.json'),strict_source_audit_passed=False,original_failed_source_audit_sha256=sha(HERE/'INTEGRITY_END.json'),user_accepted_external_maintenance_exception=True,user_authorization_receipt_sha256=sha(HERE/'USER_ACCEPTED_EXTERNAL_MAINTENANCE_AUDIT_EXCEPTION_20261022.receipt.json'),accepted_source_audit_receipt_sha256=sha(HERE/'INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.receipt.json'),final_reports_receipt_sha256=sha(HERE/'FINAL_REPORTS.receipt.json'),v1_readonly=True,main_revised_logical_requests=progress['main_planned_logical_requests'],optional_logical_requests=sum(v['planned'] for v in progress['second_stage_and_other_optional']),all_v2_logical_requests=progress['all_v2_planned_logical_requests'],remaining_logical_requests=0,own_slurm_jobs=0,original_user_stopped_scopes_not_restored=True))
        seal('V2_COMPLETE.receipt.json',[HERE/n for n in ('V2_COMPLETE.json','FINAL_REPORTS.receipt.json','INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022.receipt.json','USER_ACCEPTED_EXTERNAL_MAINTENANCE_AUDIT_EXCEPTION_20261022.receipt.json','DB_POSTFLIGHT.receipt.json','FIGURES.receipt.json',Path(__file__).name)],strict_source_audit_passed=False,completion_with_explicit_user_accepted_exception=True)
        with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+'：用户实际回复“接受”，已按精确限定的存储维护文档/HEAD两项例外完整重新只读核查，其他差异0。原严格失败审计及全部pending报告不改；正式16交付生成，原动作错误无损归档。全部科学请求/诊断/统计/最终合库/核库/图/报告完成，写V2_COMPLETE及收据。未重跑rollout、训练、bootstrap或merge，未修改受保护源码/v1/git。\n')
        print('V2 complete with explicitly user-accepted maintenance exception; original strict failed audit preserved.',flush=True)

if __name__=='__main__':main()

"""Static finalization under the user's exact, disclosed audit exception."""
from experiment import *
from registry import verify_receipt
import ast

APPROVAL='USER_ACCEPTED_EXTERNAL_MAINTENANCE_AUDIT_EXCEPTION_20261022'
AUDIT='INTEGRITY_END_ACCEPTED_EXTERNAL_MAINTENANCE_20261022'
READY='ACCEPTED_EXTERNAL_MAINTENANCE_FINALIZATION_READY_20261022'


def once(text, old, new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def create(name, text):
    path=HERE/name;assert not path.exists(),path
    ast.parse(text);path.write_text(text)
    return path


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt(APPROVAL+'.receipt.json')
    assert read(HERE/(APPROVAL+'.json'))['exception_accepted']
    assert not (HERE/'V2_COMPLETE.json').exists()
    original=(HERE/'integrity.py').read_text()
    text=original.replace('INTEGRITY_END.json',AUDIT+'.json').replace('INTEGRITY_END.receipt.json',AUDIT+'.receipt.json')
    text=once(text,'import subprocess\nimport time','import subprocess\nimport time\nfrom registry import verify_receipt')
    text=once(text,"    start=read(HERE/'INTEGRITY_START.json')",f"    verify_receipt('{APPROVAL}.receipt.json')\n    approval=read(HERE/'{APPROVAL}.json');assert approval['exception_accepted']\n    allowed=approval['accepted_differences']\n    original_failed=read(HERE/'INTEGRITY_END.json');assert not original_failed['passed'] and original_failed['differences']==allowed\n    start=read(HERE/'INTEGRITY_START.json')")
    marker=f"    write(HERE/'{AUDIT}.json',dict(utc=utc(),passed=not differences,"
    replacement=f"""    accepted=(differences==allowed and repos[str(ROOT)]['head']==approval['closing_head'] and sha(ROOT/'docs/hdd_migration_20261020.md')==approval['document_sha256'])
    unexpected=[d for d in differences if d not in allowed]
    write(HERE/'{AUDIT}.json',dict(utc=utc(),passed=accepted,strict_passed=not differences,passed_with_user_accepted_exception=accepted and bool(differences),accepted_differences=allowed,unexpected_differences=unexpected,user_authorization_receipt_sha256=sha(HERE/'{APPROVAL}.receipt.json'),original_failed_audit_sha256=sha(HERE/'INTEGRITY_END.json'),"""
    text=once(text,marker,replacement)
    text=once(text,"passed=not differences)\n    assert not differences,differences[:10]","passed=accepted,strict_passed=not differences,only_user_accepted_maintenance_exception=True)\n    assert accepted,(differences[:10],unexpected[:10])")
    text=once(text,f"HERE/'{AUDIT}.json',HERE/'integrity.py'",f"HERE/'{AUDIT}.json',HERE/'integrity_accepted_external_maintenance_20261022.py',HERE/'{APPROVAL}.receipt.json',HERE/'INTEGRITY_END.receipt.json'")
    audit_path=create('integrity_accepted_external_maintenance_20261022.py',text)

    report=(HERE/'final_reports_protocol_order_20261022.py').read_text()
    report=once(report,"audit=read(HERE/'INTEGRITY_END.json');db=read(HERE/'DB_POSTFLIGHT.json');assert audit['passed'] and db['passed']",f"verify_receipt('{AUDIT}.receipt.json');verify_receipt('{APPROVAL}.receipt.json')\n audit=read(HERE/'{AUDIT}.json');db=read(HERE/'DB_POSTFLIGHT.json');assert audit['passed'] and db['passed'] and not audit['strict_passed'] and audit['passed_with_user_accepted_exception']")
    report=once(report,'结束完整性审计通过：检查','结束完整性审计按用户接受的限定外部存储维护例外完成：严格审计仍有迁移说明追加28行及Git HEAD前移两项差异；原失败审计保留，完整重新核查没有新增未授权差异。用户明确回复“接受”，不改变实验或统计。检查')
    report=once(report,'详见INTEGRITY_END.json、DB_POSTFLIGHT.json。',f'详见原INTEGRITY_END.json、{AUDIT}.json、{APPROVAL}.json及DB_POSTFLIGHT.json。')
    report=once(report,'源码只认INTEGRITY_END、交付认FINAL_REPORTS',f'原严格源码审计认INTEGRITY_END（两项维护差异失败原件），最终限定例外闭合同时认{AUDIT}与{APPROVAL}收据、交付认FINAL_REPORTS')
    report=once(report,"save('HANDOFF.md',handoff_text())",f"save('HANDOFF.md','本轮全部科学计算、诊断、统计、最终合库、数据库核查、图表和报告已完成；最终完成标志由专用close_v2_accepted_external_maintenance_20261022.py写入。原严格INTEGRITY_END仍为两项维护文档/HEAD差异失败，用户已明确接受该限定范围，完整重新核查和授权见{AUDIT}及{APPROVAL}。没有重新运行实验、训练、bootstrap或merge。旧协调器和调度守护已退出，不重启；旧审计动作错误归档为COORDINATOR_ACTION_ERROR_STRICT_MAINTENANCE_AUDIT_20261022.json。\\n\\n'+handoff_text())")
    report=once(report,"'final_reports_protocol_order_20261022.py','training_selected_report_tables_20261022.py']],source_audit_passed=True",f"'final_reports_accepted_external_maintenance_20261022.py','training_selected_report_tables_20261022.py','{AUDIT}.receipt.json','{APPROVAL}.receipt.json']],source_audit_passed=False,strict_source_audit_passed=False,source_audit_passed_with_user_accepted_exception=True")
    report_path=create('final_reports_accepted_external_maintenance_20261022.py',report)

    closure='''"""Complete only the already computed task, with explicit accepted audit scope."""
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
        with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\\n'+utc()+'：用户实际回复“接受”，已按精确限定的存储维护文档/HEAD两项例外完整重新只读核查，其他差异0。原严格失败审计及全部pending报告不改；正式16交付生成，原动作错误无损归档。全部科学请求/诊断/统计/最终合库/核库/图/报告完成，写V2_COMPLETE及收据。未重跑rollout、训练、bootstrap或merge，未修改受保护源码/v1/git。\\n')
        print('V2 complete with explicitly user-accepted maintenance exception; original strict failed audit preserved.',flush=True)

if __name__=='__main__':main()
'''
    closure_path=create('close_v2_accepted_external_maintenance_20261022.py',closure)
    write(HERE/(READY+'.json'),dict(utc=utc(),report_only_finalization=True,no_science_reexecution=True,original_integrity_sha256=sha(HERE/'integrity.py'),original_renderer_sha256=sha(HERE/'final_reports_protocol_order_20261022.py'),user_approval_receipt_sha256=sha(HERE/(APPROVAL+'.receipt.json')),scope='Complete full read-only audit allowing exactly the two user-accepted differences; render final reports and closure, preserving original strict failure'))
    seal(READY+'.receipt.json',[Path(__file__),audit_path,report_path,closure_path,HERE/(READY+'.json'),HERE/(APPROVAL+'.receipt.json')],user_authorized=True,no_new_rollouts=True,no_repeated_statistics=True)
    print('PASS accepted-scope finalization sources sealed',flush=True)


if __name__=='__main__':main()

"""Disclose strict audit failure and render complete reviewable pending reports."""
from experiment import *
from registry import verify_receipt
import ast
import subprocess


def create(path, text):
    assert not path.exists(), path
    path.write_text(text)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    for proof in ('INTEGRITY_END.receipt.json', 'DB_POSTFLIGHT.receipt.json',
                  'OPTIONAL_NUMBERS.receipt.json', 'FIGURES.receipt.json',
                  'FINAL_DELIVERY_PROTOCOL_ORDER_20261022.receipt.json'):
        verify_receipt(proof)
    audit = read(HERE/'INTEGRITY_END.json')
    opening = read(HERE/'INTEGRITY_START.json')
    assert not audit['passed'] and len(audit['differences']) == 2
    assert {d['kind'] for d in audit['differences']} == {'content', 'git_head'}
    doc = ROOT/'docs/hdd_migration_20261020.md'
    content = next(d for d in audit['differences'] if d['kind'] == 'content')
    assert content['path'] == str(doc) and content['closing'] == sha(doc)
    old = opening['repositories'][str(ROOT)]['head']
    new = audit['repositories'][str(ROOT)]['head']
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT,
            env={**os.environ, 'GIT_OPTIONAL_LOCKS':'0'})
    git('merge-base', '--is-ancestor', old, new)
    changed = git('diff', '--name-only', old, new).decode().splitlines()
    assert changed == ['docs/hdd_migration_20261020.md'], changed
    assert hashlib.sha256(git('show', new+':docs/hdd_migration_20261020.md')).hexdigest() == sha(doc)
    diff = git('diff', old, new, '--', changed[0]).decode()
    metadata = git('log', '--format=%H %aI %s', old+'..'+new).decode()
    prefix = 'EXTERNAL_MAINTENANCE_AUDIT_REVIEW_20261022'
    write(HERE/(prefix+'.json'), dict(
        utc=utc(), strict_audit_passed=False, strict_audit_receipt_sha256=sha(HERE/'INTEGRITY_END.receipt.json'),
        strict_differences=audit['differences'], checked_files=audit['checked_files'],
        opening_head=old, closing_head=new, changed_tracked_files=changed,
        commit_metadata=metadata, git_diff=diff, current_document_sha256=sha(doc),
        evidence_conclusion='Both strict differences correspond to one storage-maintenance documentation commit; no other protected content differences were reported by the complete audit.',
        exception_accepted=False, user_decision_required_for_exception=True,
        no_new_simulations=True, no_retraining=True, no_git_mutations=True,
        original_sealed_audit_preserved=True))
    md = ('严格结束审计未通过，原 INTEGRITY_END.json/.receipt 保留。'
          '542326 个受保护文件中仅已有迁移说明内容变化，加 Git HEAD 变化，共两项差异。'
          '只读提交核对确认同一存储维护提交只给该文档追加28行；其他受保护内容没有报告差异。'
          '这不是自动豁免。实验、诊断、统计、数据库核查与图表均已完成；最终完成标志尚未生成。\n\n'
          '原规范 docs/prompt_baseline_literature_scenes_distributed_v2_20261020.md 第零节第3条：'
          '“开始与结束必须一致”；第4条：“docs/ 中已有文件被修改或删除，仍按差异处理。”\n\n'
          '提交：\n\n'+metadata+'\n变更：\n\n```diff\n'+diff+'```\n')
    create(HERE/(prefix+'.md'), md)
    seal(prefix+'.receipt.json', [HERE/(prefix+'.json'), HERE/(prefix+'.md'),
         HERE/'INTEGRITY_END.receipt.json', Path(__file__)],
         strict_audit_passed=False, exception_accepted=False)

    output = HERE/'pending_external_audit_20261022'
    output.mkdir(exist_ok=False)
    source = HERE/'final_reports_protocol_order_20261022.py'
    original = source.read_text()
    base_path = HERE/'final_reports_flat_static_allcpu_delivery.py'
    base = base_path.read_text()
    base_tree = ast.parse(base)
    functions = '\n\n'.join(ast.get_source_segment(base, node)
        for node in base_tree.body if isinstance(node, ast.FunctionDef)
        and node.name in ('geometry', 'dynamics'))
    assert functions.count('def geometry(') == functions.count('def dynamics(') == 1
    report = original
    report = replace_once(report, ',pick,geometry,dynamics,comparison', ',pick,comparison')
    report = replace_once(report, 'import csv,collections',
        "import csv,collections\nOUTDIR=HERE/'pending_external_audit_20261022'\n\n"+functions)
    report = replace_once(report, 'from training_selected_report_tables_20261022 import primary_tables',
        'from training_selected_report_tables_pending_external_audit_20261022 import primary_tables')
    report = replace_once(report, 'path=HERE/name;', 'path=OUTDIR/name;')
    report = report.replace("(HERE/'APPENDIX_REFERENCE_AXIS.csv')", "(OUTDIR/'APPENDIX_REFERENCE_AXIS.csv')")
    report = report.replace("(HERE/'DYNAMICS_STAGE.md')", "(OUTDIR/'DYNAMICS_STAGE.md')")
    report = replace_once(report, "assert audit['passed'] and db['passed']",
        "assert not audit['passed'] and db['passed'];verify_receipt('EXTERNAL_MAINTENANCE_AUDIT_REVIEW_20261022.receipt.json');assert audit['differences']==read(HERE/'EXTERNAL_MAINTENANCE_AUDIT_REVIEW_20261022.json')['strict_differences']")
    old_lead = '结束完整性审计通过：检查'
    new_lead = ('严格结束完整性审计未通过，待用户决定是否接受已披露的外部存储维护例外；'
                '未生成V2_COMPLETE。仅docs/hdd_migration_20261020.md追加28行及对应Git HEAD变化两项差异；'
                '提交0d102abf949801641aba498733a03d8255fe6c39只改此文档。'
                '原严格失败审计保留，不自行豁免。检查')
    report = replace_once(report, old_lead, new_lead)
    report = replace_once(report, '；v1与受保护源码保持只读。共享数据库',
        '；v1与实验源码内容没有报告差异。完整复核见EXTERNAL_MAINTENANCE_AUDIT_REVIEW_20261022。共享数据库')
    report = replace_once(report, 'save(\'HANDOFF.md\',handoff_text())',
        "save('HANDOFF.md','状态：所有实验、诊断、统计、官方合库、DB核查和图表完成。严格源码结束审计有两项外部存储维护文档/HEAD差异，尚未获豁免，不能称审计通过或写V2_COMPLETE。原协调器已因审计退出，COORDINATOR_ACTION_ERROR保留阻止自动重启；不要重跑实验或重新merge。完整待审报告与16个交付文件在本目录；原审计和复核证据在父目录。\\n\\n'+handoff_text())")
    report = replace_once(report,
        "seal('FINAL_REPORTS.receipt.json',[HERE/n for n in [*outputs,'final_reports_protocol_order_20261022.py','training_selected_report_tables_20261022.py']],source_audit_passed=True",
        "seal('FINAL_REPORTS_PENDING_EXTERNAL_AUDIT_20261022.receipt.json',[*[OUTDIR/n for n in outputs],HERE/'final_reports_pending_external_audit_20261022.py',HERE/'training_selected_report_tables_pending_external_audit_20261022.py',HERE/'EXTERNAL_MAINTENANCE_AUDIT_REVIEW_20261022.receipt.json'],source_audit_passed=False,exception_accepted=False")
    helper = (HERE/'training_selected_report_tables_20261022.py').read_text()
    helper = replace_once(helper, "(HERE/'AXIS_TABLE.csv')",
                          "(HERE/'pending_external_audit_20261022'/'AXIS_TABLE.csv')")
    paths = [HERE/'final_reports_pending_external_audit_20261022.py',
             HERE/'training_selected_report_tables_pending_external_audit_20261022.py']
    for path, text in zip(paths, (report, helper)):
        ast.parse(text)
        create(path, text)
    # Only report output locations, the truthful audit disclosure and pending seal change.
    before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original).body
              if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(report).body
             if isinstance(n, ast.FunctionDef)}
    unchanged = ('decomposition', 'class_gain_tables', 'handoff_text')
    assert all(before[n] == after[n] for n in unchanged)
    write(HERE/'PENDING_EXTERNAL_AUDIT_DELIVERY_READY_20261022.json', dict(
        utc=utc(), report_only=True, original_renderer_sha256=sha(source),
        original_table_helper_sha256=sha(HERE/'training_selected_report_tables_20261022.py'),
        geometry_dynamics_source_sha256=sha(base_path), unchanged_function_asts=list(unchanged),
        output_directory=str(output), strict_audit_passed=False, exception_accepted=False,
        no_rollouts=True, no_statistics_recomputed=True, no_completed_marker=True))
    seal('PENDING_EXTERNAL_AUDIT_DELIVERY_READY_20261022.receipt.json',
         [Path(__file__), *paths, HERE/'PENDING_EXTERNAL_AUDIT_DELIVERY_READY_20261022.json',
          HERE/(prefix+'.receipt.json')], report_only=True, exception_accepted=False)
    print('PASS pending complete report renderer prepared; strict failed audit unchanged', flush=True)


if __name__ == '__main__':
    main()

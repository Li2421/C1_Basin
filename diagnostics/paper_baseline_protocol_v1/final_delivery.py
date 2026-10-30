"""Evidence-only final stream, integrity receipts and explicit remaining gaps."""
import guard
from guard import HERE,ROOT
import json,hashlib,subprocess,datetime
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_stream():
    temporary=HERE/'candidates.jsonl.tmp';count=0
    with temporary.open('w')as out:
        for path in sorted((HERE/'raw').glob('*.json')):
            value=json.loads(path.read_text());out.write(json.dumps(value,ensure_ascii=False)+'\n');count+=1
    temporary.replace(HERE/'candidates.jsonl')
    # Semantic aliases remain separate from the unique original-execution stream.
    aliases=0
    if(HERE/'REUSE_MANIFEST.json').exists():aliases=len(json.loads((HERE/'REUSE_MANIFEST.json').read_text())['records'])
    value=dict(unique_new_raw_records=count,semantic_aliases=aliases,
               raw_stream='candidates.jsonl',alias_manifest='REUSE_MANIFEST.json',
               logical_context='plans/*_planned.json and tuning/test_tables/oracles metadata preserve every candidate and purpose; UIDs may repeat across logical contexts.')
    (HERE/'CANDIDATE_STREAM.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def import_integrity():
    # Include the still-running coordinator, whose atexit receipt is not saved yet.
    guard.save_imports()
    observed={};receipts=0
    for receipt in (HERE/'import_manifests').glob('*.json'):
        receipts+=1
        for name,value in json.loads(receipt.read_text()).items():observed.setdefault(name,set()).add(value)
    release=json.loads((HERE/'EXECUTION_FROZEN.json').read_text())
    immutable={str(HERE/name):value for name,value in release['files'].items()}
    for name,value in json.loads((HERE/'INPUT_SNAPSHOT.json').read_text())['files'].items():
        expected=value.get('target_sha256',value.get('source_sha256'))
        if expected:immutable.setdefault(str(HERE/name),expected)
    rows={};changed_immutable={};external_changes={}
    for name,hashes in observed.items():
        path=Path(name);actual=sha(path)if path.is_file()else None
        expected=immutable.get(name)
        rows[name]=dict(observed_sha256=sorted(hashes),final_sha256=actual,immutable_reference=expected)
        if expected is not None and actual!=expected:changed_immutable[name]=rows[name]
        if not path.is_relative_to(HERE)and(actual not in hashes or len(hashes)>1):external_changes[name]=rows[name]
    result=dict(receipts=receipts,unique_imported_files=len(rows),files=rows,
        immutable_changes=changed_immutable,external_import_changes=external_changes,
        passed=not changed_immutable and not external_changes,
        interpretation='Administrative experiment files may have documented pre-test revisions. Immutable numerical/input files and external imports are checked separately; historical import hashes are retained rather than overwritten.')
    (HERE/'IMPORT_INTEGRITY_END.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def integrity():
    release=json.loads((HERE/'EXECUTION_FROZEN.json').read_text());changed={}
    for name,expected in release['files'].items():
        path=HERE/name;actual=sha(path)if path.exists()else None
        if actual!=expected:changed[name]=dict(expected=expected,actual=actual)
    copied=json.loads((HERE/'INPUT_SNAPSHOT.json').read_text())['files'];bad_inputs={}
    for name,value in copied.items():
        path=HERE/name
        expected=value.get('target_sha256',value.get('source_sha256'))
        if expected and(not path.exists()or sha(path)!=expected):bad_inputs[name]='Copied input differs from recorded snapshot'
    original=(HERE/'GIT_START_main_DIFF.patch').read_bytes()
    current=subprocess.check_output(['git','diff','--binary','--','.',
        ':(exclude)diagnostics/paper_baseline_protocol_v1',':(exclude)diagnostics/basis_reduction_within_four_v1'])
    external_same=hashlib.sha256(original).digest()==hashlib.sha256(current).digest()
    subprocess.run(['git','status','--porcelain','--untracked-files=no'],stdout=(HERE/'GIT_END_STATUS.txt').open('w'),check=True)
    (HERE/'GIT_END_DIFF.patch').write_bytes(subprocess.check_output(['git','diff','--binary']))
    imports=import_integrity()
    freeze_changes={}
    analysis=json.loads((HERE/'ANALYSIS_PLAN_FROZEN.json').read_text())
    for field,name in [('parameters_sha256','PARAMETERS_FROZEN.json'),('ours_fixed_sha256','OURS_FIXED_FROZEN.json'),
                       ('menus_sha256','PAPER_MENUS_FROZEN.json'),('execution_release_sha256','EXECUTION_FROZEN.json'),
                       ('splits_sha256','SPLITS_FROZEN.json'),('native_reproduction_sha256','NATIVE_REPRODUCTION_RESULTS.json'),
                       ('method_policy_sha256','METHOD_EXECUTION_POLICY.json')]:
        if sha(HERE/name)!=analysis[field]:freeze_changes[name]=dict(expected=analysis[field],actual=sha(HERE/name))
    if analysis.get('relaxed_scenes_sha256')!=sha(HERE/'RELAXED_SCENES_FROZEN.json'):
        freeze_changes['RELAXED_SCENES_FROZEN.json']='Historical sensitivity scene selection changed after analysis freeze.'
    for name,expected in analysis['analysis_sources_sha256'].items():
        if sha(HERE/name)!=expected:freeze_changes[name]=dict(expected=expected,actual=sha(HERE/name))
    passed=not changed and not bad_inputs and external_same and imports['passed']and not freeze_changes and not(HERE/'PROTECTION_VIOLATION.jsonl').exists()
    value=dict(passed=passed,numerical_source_changes=changed,copied_input_changes=bad_inputs,
               protected_external_git_diff_unchanged=external_same,copied_inputs_checked=len(copied),
               import_integrity_passed=imports['passed'],imported_files_checked=imports['unique_imported_files'],
               test_freeze_changes=freeze_changes,
               execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json'),
               head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               parallel_old_directory='Metadata-only throughout this study; prior-study commit and delivery receipt retained.',
               poc='Not a Git repository at opening audit; no clean-Git claim. Numerical imports confined to copied/private namespaces and official DB.')
    (HERE/'INTEGRITY_END.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def run():
    stream=raw_stream();audit=integrity();result=json.loads((HERE/'RESULTS.json').read_text());budget=json.loads((HERE/'BUDGET.json').read_text())
    gaps=[]
    if not audit['passed']:gaps.append('Final source/repository integrity audit did not pass; inspect INTEGRITY_END.json.')
    if not result.get('main_fixed_tests_complete'):gaps.append('Required fixed-policy tests or training selections incomplete; no optional Oracle claim.')
    if not(HERE/'TIME_BACKFILL_DECISION.json').exists():gaps.append('Conditional historical timing-supplement decision not completed.')
    if(HERE/'TIME_BACKFILL_DECISION.json').exists():
        timing=json.loads((HERE/'TIME_BACKFILL_DECISION.json').read_text())
        if not timing.get('executed'):gaps.append('Historical completion times remain missing under the user1-hour condition; no imputation of successful completion times.')
    # Video is non-mainline in the user protocol; the required shared renderer
    # entry documentation is absent. Never fabricate footage or rerun for it.
    if not(ROOT/'docs/batch_videos.md').exists():gaps.append('Non-mainline video delivery: docs/batch_videos.md shared entry documentation missing; existing true traces retained, no synthetic videos.')
    gaps.append('DAC exact journal figure/page unavailable in accessible primary full text; official pinned repository/demo and DOI verified, no invented figure attribution.')
    for path in sorted(HERE.glob('NONMAIN_*_ERROR.json')):gaps.append('Non-mainline supplement incomplete; see '+path.name)
    if(HERE/'RELAXED_SENSITIVITY_RESULTS.json').exists():
        relaxed=json.loads((HERE/'RELAXED_SENSITIVITY_RESULTS.json').read_text())
        if relaxed['unavailable_logical_rows']:gaps.append(str(relaxed['unavailable_logical_rows'])+'historical sensitivity logical rows lack compatible frozen admission passages/selection; unavailable, not rollout failures.')
    process=[]
    if(HERE/'PROCESS_GAPS.json').exists():process=json.loads((HERE/'PROCESS_GAPS.json').read_text()).get('events',[])
    numeric=bool(result.get('main_fixed_tests_complete'))
    value=dict(finished=True,complete=numeric and audit['passed']and not gaps and not process,
        main_numeric_work_complete=numeric,integrity_passed=audit['passed'],process_compliance_passed=not process,
        time=datetime.datetime.now().astimezone().isoformat(),actual_attempts=budget['actual_attempts'],hard_limit=50000,
        candidates=stream,gaps=gaps,process_gaps=process,report=str(HERE/'REPORT.md'))
    assert budget['actual_attempts']<=50000
    (HERE/'COMPLETE.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    registry=json.loads((HERE/'PAPER_REGISTRY.json').read_text())
    native=json.loads((HERE/'NATIVE_REPRODUCTION_RESULTS.json').read_text())['methods']
    for method,row in native.items():
        if row['reproducible']:
            registry['methods'][method]['status']='原环境复现通过；共同环境主线固定策略测试完成'if numeric else'原环境复现通过；共同环境主线有缺失，详见COMPLETE.json'
            registry['methods'][method]['final_report']='REPORT.md'
    registry['time']=value['time']
    (HERE/'PAPER_REGISTRY.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
    with(HERE/'GAPS.md').open('a')as out:
        for gap in gaps:out.write('\n- '+gap+'\n')
    with(HERE/'REPORT.md').open('a')as out:
        out.write('\n## 交付与完整性\n\n数值主线完成：'+str(numeric)+'；源码完整性：'+str(audit['passed'])+'；新增尝试'+str(budget['actual_attempts'])+'/50000。完整过程/非主线缺口见COMPLETE.json与GAPS.md；不把带缺口交付标成全协议无缺口完成。\n')
    (HERE/'STATUS.md').write_text('# 当前状态\n\n本轮结果与可恢复记录已交付。数值主线完成：'+str(numeric)+'；完整性：'+str(audit['passed'])+'。详见REPORT.md、COMPLETE.json与GAPS.md。\n')
    return value

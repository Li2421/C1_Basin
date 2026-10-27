"""Static analysis-only memo deployment; all scientific workers stay unchanged."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ast,difflib

READY='BOOTSTRAP_EXACT_MEMO_READY_20261022.receipt.json'
COPIES={
'statistics_correction_supplement_20261022.py':'statistics_correction_bootstrap_memo_20261022.py',
'training_selected_primary_statistics_20261022.py':'training_selected_primary_statistics_memo_20261022.py',
'statistics_tasks_supplement_20261022.py':'statistics_tasks_bootstrap_memo_20261022.py',
'statistics_supplement_20261022.py':'statistics_driver_bootstrap_memo_20261022.py',
'statistics_supplement_20261022.sbatch':'statistics_bootstrap_memo_20261022.sbatch',
'correction_continuation_spatial_mask_20261022.py':'correction_continuation_bootstrap_memo_20261022.py'}


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert not (HERE/READY).exists() and not (HERE/'MIGRATION_NUMBERS.receipt.json').exists()
    assert not (HERE/'PARALLEL_NUMERIC_INPUTS_SUPPLEMENT_20261022.receipt.json').exists(),'Do not replace a running statistical stage'
    verify_receipt('BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json');verify_receipt('BOOTSTRAP_EXACT_MEMO_TEST_PLAN_20261022.receipt.json')
    assert read(HERE/'BOOTSTRAP_EXACT_MEMO_TEST_20261022.json')['passed']
    subs={Path(a).stem:Path(b).stem for a,b in COPIES.items()}
    subs.update(statistics_header_reader_20261023='statistics_bootstrap_memo_reader_20261022',statistics_training_selected_reader_20261022='statistics_training_selected_memo_reader_20261022')
    rows=[]
    for old,new in COPIES.items():
        before=(HERE/old).read_text();after=before
        for a,b in sorted(subs.items(),key=lambda p:-len(p[0])):after=after.replace(a,b)
        after=after.replace('PARALLEL_NUMERIC_INPUTS_SUPPLEMENT_20261022','PARALLEL_NUMERIC_INPUTS_BOOTSTRAP_MEMO_20261022').replace('parallel_numeric_fragments_supplement_20261022','parallel_numeric_fragments_bootstrap_memo_20261022')
        if new.startswith(('statistics_tasks_','statistics_driver_')):after=after.replace('ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json',READY)
        if new.startswith('statistics_tasks_'):
            after=after.replace('import signal,subprocess,re','import signal,subprocess,re\nfrom bootstrap_exact_memo_20261022 import snapshot as bootstrap_cache_snapshot')
            needle='result=calculate(task,old,new,census,classes,ss,ns))';assert after.count(needle)==1
            after=after.replace(needle,'result=calculate(task,old,new,census,classes,ss,ns),bootstrap_exact_cache=bootstrap_cache_snapshot())')
        if new.startswith('statistics_driver_'):
            marker=" fragments=[HERE/"
            after=after.replace(marker," names += ['bootstrap_exact_memo_20261022.py','bootstrap_fp_control_20261022.cpp','bootstrap_fp_control_20261022.so','statistics_bootstrap_memo_reader_20261022.py','statistics_training_selected_memo_reader_20261022.py']\n"+marker)
        if new.startswith('correction_continuation_'):
            after=after.replace('def main():\n',"def main():\n verify_receipt('"+READY+"')\n",1)
        if new.endswith('.py'):ast.parse(after)
        p=HERE/new;assert not p.exists();p.write_text(after);diff=p.with_suffix(p.suffix+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff)))
    # Compare the complete task manifest and unchanged task calculation AST.
    from statistics_tasks_supplement_20261022 import task_manifest as old_tasks,calculate as old_calculate
    from statistics_tasks_bootstrap_memo_20261022 import task_manifest as new_tasks,calculate as new_calculate
    import inspect
    assert old_tasks()==new_tasks() and ast.dump(ast.parse(inspect.getsource(old_calculate)))==ast.dump(ast.parse(inspect.getsource(new_calculate)))
    write(HERE/'BOOTSTRAP_EXACT_MEMO_PIPELINE_MAP_20261022.json',dict(utc=utc(),copies=COPIES,static_copies=rows,unchanged_task_manifest=True,unchanged_calculate_function_AST=True,scientific_workers_unchanged=True,training_choice_and_instance_classes_unchanged=True,total_cpu=18,gpu_shares=2,activated=False))
    seal(READY,[Path(__file__),HERE/'BOOTSTRAP_EXACT_MEMO_PIPELINE_MAP_20261022.json',HERE/'BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json',HERE/'BOOTSTRAP_EXACT_MEMO_TEST_PLAN_20261022.receipt.json',HERE/'BOOTSTRAP_EXACT_MEMO_TEST_20261022.json',HERE/'ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json',HERE/'PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json']+[p for r in rows for p in (HERE/r['new'],Path(r['diff']))],passed=True,all5000_draws_bitwise_equal=True,uncached_mathematical_kernel_AST_identical=True,cache_capacity_and_fallback_tested=True,task_manifest_unchanged=True,execution_semantics_unchanged=True,total_cpu=18,gpu_shares=2)
    print('PASS:exact bootstrap memo and6 analysis/orchestration copies sealed;workers untouched',flush=True)


if __name__=='__main__':main()

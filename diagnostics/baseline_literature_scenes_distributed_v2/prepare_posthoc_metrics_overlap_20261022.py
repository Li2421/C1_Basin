"""Static orchestration overlap after main numbers;no science or metric edits."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ast,difflib


def main():
    from posthoc_metrics_overlap_20261022 import READY,TASKS
    assert not (HERE/READY).exists();verify_receipt('MIGRATION_NUMBERS.receipt.json')
    pairs={'optional_delivery_spatial_mask_20261022.py':'optional_delivery_posthoc_overlap_20261022.py','correction_continuation_bootstrap_memo_20261022.py':'correction_continuation_posthoc_overlap_20261022.py'};rows=[]
    for old,new in pairs.items():
        before=(HERE/old).read_text();after=before
        for a,b in pairs.items():after=after.replace(Path(a).stem,Path(b).stem)
        if new.startswith('optional_'):
            after=after.replace('import time\n','import time\nimport posthoc_metrics_overlap_20261022 as background\n',1)
            start=after.index("    if not (HERE/'MAIN_ADDITIONAL_METRICS.receipt.json').exists():")
            end=after.index('    from learning_label_support import main as old_support',start)
            after=after[:start]+'    background.ensure()\n'+after[end:]
            after=after.replace('    while True:\n        state=snapshot();mapping=jobs_by_stage(state)','    while True:\n        background.ensure()\n        state=snapshot();mapping=jobs_by_stage(state)',1)
            after=after.replace("    action('official_final_merge','db_merge.py')","    background.wait()\n    action('official_final_merge','db_merge.py')",1)
        signature='def main():\n';assert after.count(signature)==1;after=after.replace(signature,signature+('    ' if new.startswith('optional_') else ' ')+f"verify_receipt('{READY}')\n",1)
        ast.parse(after);p=HERE/new;assert not p.exists();p.write_text(after);diff=p.with_suffix('.py.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff)))
    write(HERE/'POSTHOC_METRICS_OVERLAP_MAP_20261022.json',dict(utc=utc(),copies=pairs,static_copies=rows,prerequisite='MIGRATION_NUMBERS sealed before any stage-two planning',retained_original_metric_sources=[dict(script=s,receipt=p,source_sha256=sha(HERE/s)) for s,p,m in TASKS.values()],effect='Only move independent saved-evidence posthoc tasks off the serial science-stage barrier;wait for all receipts and task exits before final merge/report/audit',statistical_and_scientific_functions_unchanged=True,total_cpu=18,gpu_shards=2))
    seal(READY,[Path(__file__),HERE/'POSTHOC_METRICS_OVERLAP_MAP_20261022.json',HERE/'posthoc_metrics_overlap_20261022.py',HERE/'posthoc_metrics_overlap_20261022.sbatch',HERE/'MIGRATION_NUMBERS.receipt.json',HERE/'PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json',HERE/'BOOTSTRAP_EXACT_MEMO_LAUNCHER_FIX_20261022.receipt.json']+[HERE/s for s,p,m in TASKS.values()]+[p for r in rows for p in (HERE/r['new'],Path(r['diff']))],scheduling_only=True,science_and_metrics_source_unchanged=True,no_new_research_requests=True,main_numbers_already_sealed=True,total_cpu=18,gpu_shards=2)
    print('Sealed scheduling-only posthoc overlap;original metric kernels untouched',flush=True)


if __name__=='__main__':main()

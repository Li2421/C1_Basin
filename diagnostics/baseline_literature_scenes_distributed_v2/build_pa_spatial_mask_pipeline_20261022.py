"""Stage exact spatial-mask compute cache on the latest18CPU analysis pipeline."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import ast,difflib

EQ='PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json'
CODE='PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json'
COPIES={
'worker_correction_pa_memo_label_20261022.py':'worker_spatial_mask_label_20261022.py',
'worker_correction_pa_memo_label_20261022.sbatch':'worker_spatial_mask_label_20261022.sbatch',
'worker_correction_pa_memo_20261022.py':'worker_spatial_mask_eval_20261022.py',
'worker_correction_pa_memo_20261022.sbatch':'worker_spatial_mask_eval_20261022.sbatch',
'fine_dispatch_pa_memo_20261022_quota18_now_20261022.py':'fine_dispatch_spatial_mask_20261022.py',
'scene_learning_scheduler_reuse_order_20261022.py':'scene_learning_scheduler_spatial_mask_20261022.py',
'coordinator_supplement_20261022.py':'coordinator_spatial_mask_20261022.py',
'correction_continuation_reuse_order_20261022.py':'correction_continuation_spatial_mask_20261022.py',
'scheduler_watchdog_supplement_20261022.py':'scheduler_watchdog_spatial_mask_20261022.py',
'optional_worker_pa_memo_20261022.py':'optional_worker_spatial_mask_20261022.py',
'optional_pa_memo_20261022.sbatch':'optional_spatial_mask_20261022.sbatch',
'optional_fine_dispatch_pa_memo_20261022_quota18_now_20261022.py':'optional_fine_dispatch_spatial_mask_20261022.py',
'optional_delivery_supplement_clarified_20261022.py':'optional_delivery_spatial_mask_20261022.py',
'final_reports_supplement_clarified_20261022.py':'final_reports_spatial_mask_20261022.py',
'dynamics_pa_memo_20261022.py':'dynamics_spatial_mask_20261022.py',
'runtime_local_progress_pa_memo_20261022.py':'runtime_local_progress_spatial_mask_20261022.py'}


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('PA_SPATIAL_MASK_CANDIDATE_20261022.receipt.json');verify_receipt('CIRCLE_EVALUATION_REUSE_ORDER_20261022.receipt.json')
    assert not (HERE/'PA_SPATIAL_MASK_PIPELINE_SOURCE_MAP_20261022.json').exists()
    subs={Path(a).stem:Path(b).stem for a,b in COPIES.items()}
    subs.update({Path(r['original']).stem:Path(r['new']).stem for r in read(HERE/'PA_SPATIAL_MASK_CANDIDATE_20261022.json')['static_copies']})
    rows=[]
    for old,new in COPIES.items():
        before=(HERE/old).read_text();after=before
        for a,b in sorted(subs.items(),key=lambda p:-len(p[0])):after=after.replace(a,b)
        worker=new.startswith(('worker_','optional_worker_')) and new.endswith('.py')
        if worker:
            helper="def spatial_mask_snapshot():\n    from pa_spatial_mask_stats_20261022 import snapshot\n    return snapshot()\n\n\n"
            assert after.count('def main(stage):')==1;after=after.replace('def main(stage):',helper+'def main(stage):',1)
            oldline='                result,arrays=run(c)\n';assert after.count(oldline)==1
            addition="                before_spatial=spatial_mask_snapshot() if before_pa is not None else None\n"+oldline
            addition+="                if before_spatial is not None:\n                    after_spatial=spatial_mask_snapshot()\n                    result['spatial_mask_cache']=dict(counter_deltas={k:after_spatial[k]-before_spatial[k] for k in after_spatial if k!='maximum_entries'},maximum_cells=after_spatial['maximum_entries'])\n"
            addition+=f"                    result['spatial_mask_cache_equivalence_receipt_sha256']=sha(HERE/'{EQ}')\n"
            after=after.replace(oldline,addition,1)
        if new.endswith('.py') and new.startswith(('worker_','optional_worker_','coordinator_','correction_continuation_','scheduler_watchdog_','optional_delivery_','final_reports_','fine_dispatch_','optional_fine_dispatch_')):
            signature='def main(stage):' if worker else 'def submit(stage):' if new.startswith(('fine_dispatch_','optional_fine_dispatch_')) else 'def main():'
            line=after.split(signature+'\n',1)[1].splitlines()[0];indent=line[:len(line)-len(line.lstrip())]
            after=after.replace(signature+'\n',signature+'\n'+indent+f"verify_receipt('{EQ}');verify_receipt('{CODE}')\n",1)
        if new.startswith('final_reports_'):
            needle='PA另启用有界精确查询内存缓存：';assert after.count(needle)==1
            after=after.replace(needle,f'PA低层原生查询另缓存同一查询内同一网格单元/移动的静态墙谓词和已静止预约的碰撞谓词；连续起点首步与目标/保持边仍走原运算，最多65536单元，超过容量走原运算，搜索顺序、预算和轨迹不变。658项原生路径、含实际容量回退、四场景20查询及两条完整2000步轨迹逐位验证见{EQ}。单例计时不代表全流程倍率。'+needle)
        if new.startswith('scene_learning_scheduler_'):
            needle="            originals += [evaluations[s] for s in active if s in evaluations and stage_done(evaluations[s])]"
            assert after.count(needle)==1
            after=after.replace(needle,"            originals += [s for s in fixed_evaluations if stage_done(s)]\n"+needle)
        if new.endswith('.py'):ast.parse(after)
        p=HERE/new;assert not p.exists();p.write_text(after);diff=p.with_suffix(p.suffix+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        row=dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff))
        if new.startswith(('dynamics_','runtime_local_progress_')):
            fs=lambda s:[ast.dump(n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            assert fs(before)==fs(after);row['physics_and_controller_function_AST_identical']=True
        rows.append(row)
    write(HERE/'PA_SPATIAL_MASK_PIPELINE_SOURCE_MAP_20261022.json',dict(utc=utc(),copies=COPIES,static_copies=rows,requires=[EQ,CODE],production_activated=False,total_cpu=18,gpu_shares=2,analysis_supplement_retained=True,circle_evaluation_reuse_order_retained=True,scientific_cases_and_budgets_unchanged=True,transition='USR1 completes current episode;replace as slots become available'))
    print('Staged16 latest-pipeline compute variants;not activated',flush=True)


if __name__=='__main__':main()

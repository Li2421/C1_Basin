"""Stage static variants only. This script never changes active processes."""
from experiment import *
from atomic_write import write
import ast,difflib

EQ='CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json'
CODE='CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json'
COPIES={
 'worker_correction_revised_20261021.py':'worker_correction_start_signature_20261022.py',
 'worker_correction_revised_20261021.sbatch':'worker_correction_start_signature_20261022.sbatch',
 'fine_dispatch_correction_release_20261023.py':'fine_dispatch_start_signature_20261022.py',
 'scene_learning_scheduler_revised_release_20261023.py':'scene_learning_scheduler_start_signature_20261022.py',
 'coordinator_correction_configured_20261023.py':'coordinator_start_signature_20261022.py',
 'correction_after_distributed_fixed_closed_scope_20261023.py':'correction_after_distributed_fixed_start_signature_20261022.py',
 'scheduler_watchdog_correction_configured_20261023.py':'scheduler_watchdog_start_signature_20261022.py',
 'optional_worker_correction_revised_20261021.py':'optional_worker_start_signature_20261022.py',
 'optional_correction_revised_20261021.sbatch':'optional_start_signature_20261022.sbatch',
 'optional_fine_dispatch_correction_revised_20261021.py':'optional_fine_dispatch_start_signature_20261022.py',
 'optional_delivery_correction_closed_scope_20261023.py':'optional_delivery_start_signature_20261022.py',
 'final_reports_correction_closed_scope_20261023.py':'final_reports_start_signature_20261022.py',
 'dynamics_correction_revised_20261021.py':'dynamics_start_signature_20261022.py',
 'runtime_local_progress_correction_revised_20261021.py':'runtime_local_progress_start_signature_20261022.py',
}


def add_guard(source,signature):
    assert source.count(signature+'\n')==1,signature
    following=source.split(signature+'\n',1)[1].splitlines()[0]
    indent=following[:len(following)-len(following.lstrip())]
    return source.replace(signature+'\n',signature+'\n'+indent+f"verify_receipt('{EQ}');verify_receipt('{CODE}')\n",1)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    replacements={Path(a).stem:Path(b).stem for a,b in COPIES.items()}
    replacements['runtime_correction_revised_20261021']='runtime_correction_start_signature_v2_candidate_20261022'
    replacements['reservation_certificate_v2_flat_static']='reservation_certificate_v2_start_signature_v2_20261022'
    rows=[]
    guarded=('worker_','optional_worker_','coordinator_','correction_after_','scheduler_watchdog_','optional_delivery_','final_reports_')
    for origin,target in COPIES.items():
        before=(HERE/origin).read_text();after=before
        for old,new in sorted(replacements.items(),key=lambda r:-len(r[0])):after=after.replace(old,new)
        if target.endswith('.py') and target.startswith(guarded):
            after=add_guard(after,'def main(stage):' if target.startswith(('worker_','optional_worker_')) else 'def main():')
        if target.startswith(('fine_dispatch_','optional_fine_dispatch_')):after=add_guard(after,'def submit(stage):')
        if target.endswith('.py') and target.startswith('worker_'):
            old="result['correction_revision_receipt_sha256']=sha(HERE/'CORRECTION_REVISION.receipt.json')"
            new=f"result['computational_equivalence_receipt_sha256']=sha(HERE/'{EQ}') if run is distributed_run else None\n                "+old
            assert after.count(old)==1;after=after.replace(old,new)
        if target.endswith('.py') and target.startswith('optional_worker_'):
            old="result['pure_geometry_cache_equivalence_receipt_sha256']="
            assert after.count(old)==1;after=after.replace(old,f"result['computational_equivalence_receipt_sha256']=sha(HERE/'{EQ}')\n                "+old)
        if target.startswith('final_reports_'):
            old='性能加速为已验证静态必要条件证书和原生精确bit-key几何缓存，科学语义/RNG/事件不变；'
            new=f'性能加速另含低层A*起点网格/完整初始边判定签名的精确缓存，证明及完整2000步逐位回放见{EQ}与CBS_START_SIGNATURE_THEORY_20261022.md。该优化只复用数学上相同的查询输出；路径第一点保留本次真实坐标，原700时域、CBS节点预算、物理与统计定义不变，不构成第三套修正语义。'+old
            assert after.count(old)==1;after=after.replace(old,new)
        if target.endswith('.py'):ast.parse(after)
        path=HERE/target
        if path.exists():assert path.read_text()==after,target
        else:path.write_text(after)
        diff=path.with_suffix(path.suffix+'.diff');delta=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=origin,tofile=target))
        if diff.exists():assert diff.read_text()==delta,str(diff)
        else:diff.write_text(delta)
        row=dict(original=origin,staged=target,original_sha256=sha(HERE/origin),staged_sha256=sha(path),diff=str(diff))
        if target.startswith(('dynamics_','runtime_local_progress_')):
            def bodies(s):return [ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            assert bodies(before)==bodies(after),target;row['all_physics_and_control_function_AST_identical']=True
        rows.append(row)
    write(HERE/'CBS_START_SIGNATURE_PIPELINE_SOURCE_MAP_20261022.json',dict(utc=utc(),static_copies=rows,activated=False,no_worker_signals=True,no_runtime_patching=True,scientific_case_factories_unchanged=True,resources='Existing dated24->18CPU,2GPU policy unchanged',required_before_activation=[EQ,CODE]))
    print('Staged',len(rows),'static orchestration/worker variants; no activation',flush=True)


if __name__=='__main__':main()

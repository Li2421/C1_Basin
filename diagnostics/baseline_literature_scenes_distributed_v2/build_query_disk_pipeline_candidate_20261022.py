"""Stage future eval/optional acceleration; keep all label workers unchanged."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from build_start_signature_pipeline_20261022 import COPIES as PREVIOUS
import ast,difflib

EQ='CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json'
CODE='CBS_QUERY_DISK_PIPELINE_CODE_20261022.receipt.json'
COPIES={value:value.replace('start_signature','query_disk') for value in PREVIOUS.values()}
LABEL_BATCH='worker_correction_start_signature_20261022.sbatch'
HELPER='''def query_disk_snapshot():
    import ctypes
    from tiled_planning_query_disk_candidate_20261022 import CBS,QUERY_CACHE_NAMESPACE
    values=(ctypes.c_uint64*7)()
    CBS.query_disk_cache_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    CBS.query_disk_cache_stats.restype=None
    CBS.query_disk_cache_stats(values)
    return dict(namespace=QUERY_CACHE_NAMESPACE,values=list(values))


'''


def guard(source,signature):
    assert source.count(signature+'\n')==1
    line=source.split(signature+'\n',1)[1].splitlines()[0];indent=line[:len(line)-len(line.lstrip())]
    return source.replace(signature+'\n',signature+'\n'+indent+f"verify_receipt('{EQ}');verify_receipt('{CODE}')\n",1)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    if (HERE/CODE).exists():verify_receipt(CODE);print('Already sealed; no files rewritten');return
    replacements={Path(a).stem:Path(b).stem for a,b in COPIES.items()}
    replacements['runtime_correction_start_signature_v2_candidate_20261022']='runtime_correction_query_disk_candidate_20261022'
    replacements['reservation_certificate_v2_start_signature_v2_20261022']='reservation_certificate_v2_query_disk_candidate_20261022'
    rows=[]
    for origin,target in COPIES.items():
        before=(HERE/origin).read_text();after=before
        for a,b in sorted(replacements.items(),key=lambda item:-len(item[0])):after=after.replace(a,b)
        worker=target.endswith('.py') and target.startswith(('worker_','optional_worker_'))
        if target.endswith('.py') and target.startswith(('worker_','optional_worker_','coordinator_','correction_after_','scheduler_watchdog_','optional_delivery_','final_reports_')):
            after=guard(after,'def main(stage):' if worker else 'def main():')
        if target.startswith(('fine_dispatch_','optional_fine_dispatch_')):after=guard(after,'def submit(stage):')
        if worker:
            signature='def main(stage):';assert after.count(signature)==1;after=after.replace(signature,HELPER+signature,1)
            old='                result,arrays=run(c)\n';assert after.count(old)==1
            condition='run is distributed_run' if target.startswith('worker_') else 'True'
            replacement=f'                before_query=query_disk_snapshot() if {condition} else None\n'+old
            replacement+="                if before_query is not None:\n                    after_query=query_disk_snapshot()\n                    result['native_query_cache']=dict(namespace=after_query['namespace'],counter_deltas=[b-a for a,b in zip(before_query['values'],after_query['values'])],counter_order=['reads','hits','collisions','invalid','writes','write_errors','skipped'])\n"
            replacement+=f"                    result['query_disk_cache_equivalence_receipt_sha256']=sha(HERE/'{EQ}')\n"
            after=after.replace(old,replacement)
        if target.startswith('fine_dispatch_'):
            # No draining/restarting scientific label workers. Even future
            # replacement label jobs retain the current memory-only backend.
            selected="str(HERE/'worker_correction_query_disk_20261022.sbatch')";assert after.count(selected)==1
            after=after.replace(selected,"str(HERE/('"+LABEL_BATCH+"' if stage.startswith('learning_labels_revised_20261021_') else 'worker_correction_query_disk_20261022.sbatch'))")
        if target.startswith('final_reports_'):
            old='性能加速另含低层A*';assert after.count(old)==1
            paragraph=f'后续评估和第二阶段/变体另使用进程间低层A*精确查询缓存，完整键和SHA256校验通过才复用；未命中、碰撞或I/O失败执行原搜索。可替换的缓存槽只保存纯函数计算结果，最多65536槽、每槽16KiB，不是研究raw、轨迹或数据库。当前标签worker保留内存缓存版本，不为此中断正在跑的局；两版共享同一科学执行身份和预注册请求。证据见{EQ}。这不是全流程提速倍数保证。'+old
            after=after.replace(old,paragraph)
        if target.endswith('.py'):ast.parse(after)
        path=HERE/target;delta=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=origin,tofile=target))
        if path.exists():assert path.read_text()==after,target
        else:path.write_text(after)
        d=path.with_suffix(path.suffix+'.diff')
        if d.exists():assert d.read_text()==delta,str(d)
        else:d.write_text(delta)
        row=dict(original=origin,staged=target,original_sha256=sha(HERE/origin),staged_sha256=sha(path),diff=str(d))
        if target.startswith(('dynamics_','runtime_local_progress_')):
            def bodies(s):return [ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            assert bodies(before)==bodies(after),target;row['physics_and_control_function_AST_identical']=True
        rows.append(row)
    result=dict(static_copies=rows,activated=False,requires=[EQ,CODE],current_label_batch_preserved=LABEL_BATCH,new_worker_signals=0,scientific_case_factories_and_queues_unchanged=True,resource_policy_unchanged=True)
    p=HERE/'CBS_QUERY_DISK_PIPELINE_SOURCE_MAP_20261022.json'
    if p.exists():assert all(read(p)[k]==v for k,v in result.items())
    else:write(p,dict(utc=utc(),**result))
    print('Staged14static variants; labels keep current batch; no active changes',flush=True)


if __name__=='__main__':main()

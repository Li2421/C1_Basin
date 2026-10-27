"""Stage static PA acceleration on the current immediate18CPU pipeline."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from build_query_disk_pipeline_candidate_20261022 import COPIES as DISK_COPIES
import ast
import difflib

EQ='PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json'
CODE='PA_QUERY_MEMO_PIPELINE_CODE_20261022.receipt.json'
HELPER='''def pa_query_snapshot():
    from pa_query_memo_candidate_20261022 import snapshot
    return snapshot()


'''


def instrument_worker(text,condition):
    signature='def main(stage):'
    assert text.count(signature)==1
    text=text.replace(signature,HELPER+signature,1)
    old='                result,arrays=run(c)\n'
    assert text.count(old)==1
    new=f'                before_pa=pa_query_snapshot() if {condition} else None\n'+old
    new+="                if before_pa is not None:\n                    after_pa=pa_query_snapshot()\n                    names=('queries','hits','negative_hits','misses','stored','evictions','bypass','oversize','allocation_fallback')\n                    result['pa_query_cache']=dict(counter_deltas={k:after_pa.get(k,0)-before_pa.get(k,0) for k in names},state_after=after_pa)\n"
    new+=f"                    result['pa_query_cache_equivalence_receipt_sha256']=sha(HERE/'{EQ}')\n"
    return text.replace(old,new)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('CPU_LIMIT_IMMEDIATE_18CPU_CODE_20261022.receipt.json')
    verify_receipt('PA_QUERY_MEMO_CANDIDATE_CODE_20261022.receipt.json')
    assert not (HERE/'PA_QUERY_MEMO_PIPELINE_SOURCE_MAP_20261022.json').exists()
    quota=read(HERE/'CPU_LIMIT_IMMEDIATE_18CPU_SOURCE_MAP_20261022.json')['mapping']
    copies={quota.get(v,v):quota.get(v,v).replace('query_disk','pa_memo') for v in DISK_COPIES.values()}
    copies.update({'worker_correction_start_signature_20261022.py':'worker_correction_pa_memo_label_20261022.py','worker_correction_start_signature_20261022.sbatch':'worker_correction_pa_memo_label_20261022.sbatch'})
    substitutions={Path(a).stem:Path(b).stem for a,b in copies.items()}
    substitutions.update(tiled_planning_query_disk_candidate_20261022='tiled_planning_pa_memo_query_disk_candidate_20261022',runtime_correction_query_disk_candidate_20261022='runtime_correction_pa_memo_query_disk_candidate_20261022',reservation_certificate_v2_query_disk_candidate_20261022='reservation_certificate_v2_pa_memo_query_disk_candidate_20261022',runtime_correction_start_signature_v2_candidate_20261022='runtime_correction_pa_memo_candidate_20261022')
    rows=[]
    for old,new in copies.items():
        before=(HERE/old).read_text();after=before
        for src,dst in sorted(substitutions.items(),key=lambda item:-len(item[0])):after=after.replace(src,dst)
        worker=new.endswith('.py') and new.startswith(('worker_','optional_worker_'))
        if worker:
            after=instrument_worker(after,'run is distributed_run' if new.startswith('worker_') else 'True')
        if new.endswith('.py') and new.startswith(('worker_','optional_worker_','coordinator_','correction_after_','scheduler_watchdog_','optional_delivery_','final_reports_')):
            signature='def main(stage):' if worker else 'def main():'
            line=after.split(signature+'\n',1)[1].splitlines()[0];indent=line[:len(line)-len(line.lstrip())]
            after=after.replace(signature+'\n',signature+'\n'+indent+f"verify_receipt('{EQ}');verify_receipt('{CODE}')\n",1)
        if new.startswith(('fine_dispatch_','optional_fine_dispatch_')):
            signature='def submit(stage):'
            line=after.split(signature+'\n',1)[1].splitlines()[0];indent=line[:len(line)-len(line.lstrip())]
            after=after.replace(signature+'\n',signature+'\n'+indent+f"verify_receipt('{EQ}');verify_receipt('{CODE}')\n",1)
        if new.startswith('final_reports_'):
            needle='性能加速另含低层A*';assert after.count(needle)==1
            after=after.replace(needle,f'PA另启用有界精确查询内存缓存：保留全部预约后缀与十个初始判定位，恢复真实连续起点，未命中仍执行原始A*。每进程最多8192项和32MiB估计内容；完整两个2000步保存episode回放与有限网格、四场景查询均须逐位通过。证据见{EQ}；单例计时不代表全流程倍数。当前资源按用户最新指令固定为18CPU（含1管理核）和2GPU份额。'+needle)
        if new.endswith('.py'):ast.parse(after)
        path=HERE/new
        if path.exists():assert path.read_text()==after,path
        else:path.write_text(after)
        delta=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new))
        diff=path.with_suffix(path.suffix+'.diff')
        if diff.exists():assert diff.read_text()==delta,diff
        else:diff.write_text(delta)
        row=dict(original=old,staged=new,original_sha256=sha(HERE/old),staged_sha256=sha(path),diff=str(diff))
        if new.startswith(('dynamics_','runtime_local_progress_')):
            bodies=lambda t:[ast.dump(n) for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            assert bodies(before)==bodies(after);row['physics_and_control_function_AST_identical']=True
        rows.append(row)
    write(HERE/'PA_QUERY_MEMO_PIPELINE_SOURCE_MAP_20261022.json',dict(utc=utc(),copies=copies,static_copies=rows,activated=False,requires=[EQ,CODE],quota_code='CPU_LIMIT_IMMEDIATE_18CPU_CODE_20261022.receipt.json',quota_policy='RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json',total_cpu=18,gpu_shares=2,old_circle_semantics_closed=True,scientific_cases_and_budgets_unchanged=True,label_transition='Finish current episode before switching workers',new_research_rollouts=0))
    print('Staged',len(rows),'static variants; immediate18CPU quota retained; not activated',flush=True)


if __name__=='__main__':main()

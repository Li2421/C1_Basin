"""Use validated alias-only diagnosis reuse in future posthoc jobs."""
from experiment import *
from registry import verify_receipt
import ast
import difflib
import subprocess
import signal
import time

def main():
    tests=read(HERE/'DIAGNOSIS_ALIAS_CACHE_TESTS.json');assert tests['passed'] and tests['independent_existing_diagnoses_compared']>=21
    for name in ('DOWNSTREAM_READY.receipt.json','DYNAMICS_DIAGNOSTICS.receipt.json','WORKER_FULL_CERTIFIED_DEPLOYMENT.receipt.json'):verify_receipt(name)
    seal('DIAGNOSIS_ALIAS_CACHE.receipt.json',[HERE/n for n in ('diagnosis_alias_cache.py','test_diagnosis_alias_cache.py','DIAGNOSIS_ALIAS_CACHE_TESTS.json','failure_analysis.py','dynamics_diagnostics.py','DOWNSTREAM_READY.receipt.json','DYNAMICS_DIAGNOSTICS.receipt.json')],independent_saved_twin_diagnoses=tests['independent_existing_diagnoses_compared'],semantic_mismatch_rejection=True,new_physical_steps=0,statistical_seed_multiplicity_preserved=True)
    base='''            with np.load(raw['trace_path']) as z:result=summary(raw,z)
            write(dest,dict(utc=utc(),rollout_uid=row['rollout_uid'],source=row,trace_sha256=raw['trace_sha256'],**result))'''
    cached='''            twin=reuse(raw)
            if twin is None:
                with np.load(raw['trace_path']) as z:result=summary(raw,z)
                reuse_info={}
            else:result=twin['result'];reuse_info=dict(alias_diagnosis_reuse=twin['provenance'])
            write(dest,dict(utc=utc(),rollout_uid=row['rollout_uid'],source=row,trace_sha256=raw['trace_sha256'],**reuse_info,**result))'''
    mappings={
        'analysis_jobs_current.py':('analysis_jobs_cached.py',[
            ('from atomic_write import write','from atomic_write import write\nfrom diagnosis_alias_cache import reuse'),
            ('from resources import','from resources_released import'),
            ("verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');index=read(HERE/stage/'GROUP_INDEX.json')","verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json');verify_receipt('DOWNSTREAM_READY.receipt.json');index=read(HERE/stage/'GROUP_INDEX.json')"),
            (base,cached),('analysis_jobs_current.py','analysis_jobs_cached.py')]),
        'analysis_current.sbatch':('analysis_cached.sbatch',[('analysis_jobs_current.py','analysis_jobs_cached.py')]),
        'analysis_jobs_released.py':('analysis_submit_cached.py',[("str(HERE/'analysis_current.sbatch')","str(HERE/'analysis_cached.sbatch')"),
            ("verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')","verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json')")]),
        'optional_analysis_worker.py':('optional_analysis_worker_cached.py',[
            ('from atomic_write import write','from atomic_write import write\nfrom diagnosis_alias_cache import reuse'),
            ("verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json')","verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json');verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json');verify_receipt('DOWNSTREAM_READY.receipt.json');verify_receipt('DYNAMICS_DIAGNOSTICS.receipt.json')"),
            (base,cached),('optional_analysis_worker.py','optional_analysis_worker_cached.py')]),
        'optional_analysis.sbatch':('optional_analysis_cached.sbatch',[('optional_analysis_worker.py','optional_analysis_worker_cached.py')]),
        'coordinator_active.py':('coordinator_fast.py',[
            ('from analysis_jobs_released import submit','from analysis_submit_cached import submit'),("'downstream_active.py'","'downstream_fast.py'")]),
        'downstream_active.py':('downstream_fast.py',[
            ('from coordinator_active import','from coordinator_fast import'),("'optional_delivery_active.py'","'optional_delivery_fast.py'")]),
        'optional_delivery_active.py':('optional_delivery_fast.py',[
            ('from coordinator_active import','from coordinator_fast import'),('optional_analysis.sbatch','optional_analysis_cached.sbatch'),("'final_reports_active.py'","'final_reports_fast.py'")]),
        'final_reports_active.py':('final_reports_fast.py',[
            ('coordinator_active.py','coordinator_fast.py'),('downstream_active.py','downstream_fast.py'),
            ('optional_delivery_active.py','optional_delivery_fast.py'),('final_reports_active.py','final_reports_fast.py'),
            ('后验诊断 analysis_jobs_current.py 用 atomic_write.py','后验诊断 analysis_jobs_cached.py/optional_analysis_worker_cached.py 用 atomic_write.py'),
            ('不能改已密封规范 experiment.write。','不能改已密封规范 experiment.write。diagnosis_alias_cache.py仅复用trace SHA相同、case除seed身份外一致、完整raw除seed/计时元数据外一致的确定性副本；23组独立既有诊断核对通过，语义不匹配就正常重算。每个seed仍有独立UID/诊断入口并保留统计权重。')])
    }
    records=[]
    for source,(target,replacements) in mappings.items():
        before=(HERE/source).read_text();after=before
        for old,new in replacements:
            assert old in after,(source,old)
            after=after.replace(old,new)
        if target.endswith('.py'):ast.parse(after)
        dest=HERE/target;assert not dest.exists();dest.write_text(after)
        diff=HERE/(target+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=source,tofile=target)))
        records.append(dict(source=source,source_sha256=sha(HERE/source),target=target,target_sha256=sha(dest),diff=str(diff),diff_sha256=sha(diff)))
    write(HERE/'POSTHOC_CACHE_DERIVATIONS.json',records)
    seal('POSTHOC_CACHE_DEPLOYMENT.receipt.json',[HERE/'DIAGNOSIS_ALIAS_CACHE.receipt.json',HERE/'POSTHOC_CACHE_DERIVATIONS.json',HERE/'deploy_posthoc_cache.py',*[HERE/r['target'] for r in records],*[Path(r['diff']) for r in records]],source_case_and_raw_must_match_exactly_except_seed_bookkeeping=True,no_new_rollouts=True,statistical_rules_unchanged=True,resource_limit_18_cpu_2_gpu=True)
    # No posthoc workers are running yet. Only restart the local orchestration.
    rows=[line.strip().split(None,2) for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
    targets={read(HERE/'COORDINATOR_PID.json')['pid']}
    while True:
        new={int(r[0]) for r in rows if int(r[1]) in targets}
        if new<=targets:break
        targets|=new
    for pid in targets:
        try:os.kill(pid,signal.SIGSTOP)
        except ProcessLookupError:pass
    for pid in targets:
        try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    time.sleep(1)
    for pid in targets:
        try:os.kill(pid,0);os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    with (HERE/'coordinator.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator_fast.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'ACTIVE_ENTRYPOINT.json',dict(utc=utc(),coordinator='coordinator_fast.py',pid=process.pid,worker='worker_active.py',submit='submit_active.py',analysis='analysis_jobs_cached.py',downstream='downstream_fast.py',optional='optional_delivery_fast.py',final_reports='final_reports_fast.py',resources='resources_released.py',total_cpu=18,total_gpu_shards=2))
    with (HERE/'ITERATION_LOG.md').open('a') as stream:
        stream.write('\n2026-10-21 用户授权在不影响效果前提下尽量加速：用原生单进程merger提前合并已完成批次的239080份journal，后续官方合并仍由同一锁串行补齐。新增确定性副本后验诊断缓存，仅在trace SHA、case（除seed身份）及完整raw（除seed/计时元数据）逐值一致时复用已有诊断；23组独立计算的既有双种子诊断完全一致，篡改trace/责任系数/η均拒绝缓存。每个逻辑seed仍有独立诊断入口与原统计权重，不新增轨迹、不改失败/成功定义；物理worker不动，资源18 CPU/2 GPU不变。当前coordinator_fast/downstream_fast，来源diff与缓存收据封存于POSTHOC_CACHE_DEPLOYMENT.receipt.json。\n')
    print('Cached analysis deployment ready; coordinator',process.pid,flush=True)

if __name__=='__main__':main()

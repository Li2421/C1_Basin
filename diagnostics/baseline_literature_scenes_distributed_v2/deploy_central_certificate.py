"""Select a proven-equivalent central backend; keep current episodes intact."""
from experiment import *
from registry import verify_receipt
import ast
import difflib
import subprocess
import signal
import time

def main():
    verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json');verify_receipt('RESOURCE_RELEASE_CORRECTION.receipt.json')
    replay=read(HERE/'CENTRAL_FIXED_CERTIFICATE_FULL_REPLAY.json');assert replay['passed']
    assert all(replay['arrays'].values()) and all(replay['terminal'].values())
    seal('CENTRAL_FIXED_CERTIFICATE_REPLAY.receipt.json',[HERE/n for n in ('CENTRAL_CIRCLE_COST_PROFILE.json','CENTRAL_FIXED_CERTIFICATE_SNAPSHOT_TEST.json','CENTRAL_FIXED_CERTIFICATE_FULL_REPLAY.json','test_central_fixed_certificate.py','central_reservation_reference_v2.py','RESERVATION_V2_EQUIVALENCE.receipt.json')],new_research_rollouts=0,original_seconds=replay['original_seconds'],equivalent_backend_seconds=replay['seconds'],native_joint_cone_and_unknown_semantics_preserved=True)
    mappings={
        'worker_reservation_v2.py':('worker_active.py',[
            ('from central_tiled_runtime import run as historical_run','from central_reservation_reference_v2 import run as historical_run'),
            ("verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json')","verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json');verify_receipt('WORKER_FULL_CERTIFIED_DEPLOYMENT.receipt.json')"),
            ("actual='central_tiled_runtime.py'","actual='central_reservation_reference_v2.py'"),
            ('worker_reservation_v2.py','worker_active.py'),
            ('result["reservation_equivalence_receipt_sha256"]=sha(HERE/"RESERVATION_V2_EQUIVALENCE.receipt.json");save(stage,c,result,arrays)',
             'result["reservation_equivalence_receipt_sha256"]=sha(HERE/"RESERVATION_V2_EQUIVALENCE.receipt.json");result["central_fixed_equivalence_receipt_sha256"]=sha(HERE/"CENTRAL_FIXED_CERTIFICATE_REPLAY.receipt.json");save(stage,c,result,arrays)')]),
        'worker_reservation_v2.sbatch':('worker_active.sbatch',[('worker_reservation_v2.py','worker_active.py')]),
        'submit_reservation_released.py':('submit_active.py',[
            ('worker_reservation_v2.sbatch','worker_active.sbatch'),
            ("verify_receipt('RESERVATION_V2_WORKER_DEPLOYMENT.receipt.json')","verify_receipt('RESERVATION_V2_WORKER_DEPLOYMENT.receipt.json');verify_receipt('WORKER_FULL_CERTIFIED_DEPLOYMENT.receipt.json')")]),
        'infrastructure_recovery_current.py':('infrastructure_recovery_active.py',[
            ("'worker_reservation' in review['slurm']['details']","('worker_reservation' in review['slurm']['details'] or 'worker_active' in review['slurm']['details'])")]),
        'coordinator_released.py':('coordinator_active.py',[
            ('from infrastructure_recovery_current import','from infrastructure_recovery_active import'),
            ('from submit_reservation_released import','from submit_active import'),("'downstream_released.py'","'downstream_active.py'")]),
        'downstream_released.py':('downstream_active.py',[
            ('from coordinator_released import','from coordinator_active import'),("'optional_delivery_released.py'","'optional_delivery_active.py'")]),
        'optional_delivery_released.py':('optional_delivery_active.py',[
            ('from coordinator_released import','from coordinator_active import'),
            ('from infrastructure_recovery_current import','from infrastructure_recovery_active import'),("'final_reports_released.py'","'final_reports_active.py'")]),
        'final_reports_released.py':('final_reports_active.py',[
            ('coordinator_released.py','coordinator_active.py'),('downstream_released.py','downstream_active.py'),
            ('submit_reservation_released.py','submit_active.py'),('worker_reservation_v2.py','worker_active.py'),
            ('optional_delivery.py','optional_delivery_active.py'),('final_reports_released.py','final_reports_active.py'),
            ('集中式非拥挤场景使用复制的旧执行链；仅按精确身份复用。','集中式非拥挤场景使用 central_reservation_reference_v2.py：保留原联合投影、锥诊断及unknown语义，仅加入原搜索必无解的必要条件证书。CENTRAL_FIXED_CERTIFICATE_REPLAY.receipt.json封存一个原耗时3592秒的2000步固定修正案例回放，全部物理数组及联合锥/起作用行/结果逐位一致，耗时9.85秒。只按精确身份复用。')])
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
    write(HERE/'CENTRAL_CERTIFICATE_OPERATIONAL_DERIVATIONS.json',records)
    seal('WORKER_FULL_CERTIFIED_DEPLOYMENT.receipt.json',[HERE/'CENTRAL_FIXED_CERTIFICATE_REPLAY.receipt.json',HERE/'CENTRAL_CERTIFICATE_OPERATIONAL_DERIVATIONS.json',HERE/'deploy_central_certificate.py',*[HERE/r['target'] for r in records],*[Path(r['diff']) for r in records]],old_science_sources_unchanged=True,canonical_cases_and_seeds_unchanged=True,resource_policy='resources_released.py; total 18 CPU and 2 GPU shares',preserve_current_episodes=True)
    # Quiesce only the two local submission processes, not Slurm workers.
    rows=[line.strip().split(None,2) for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
    root=read(HERE/'COORDINATOR_PID.json')['pid'];targets={root}
    while True:
        new={int(r[0]) for r in rows if int(r[1]) in targets}
        if new<=targets:break
        targets|=new
    for pid in targets:
        try:os.kill(pid,signal.SIGSTOP)
        except ProcessLookupError:pass
    from resources_released import snapshot
    from coordinator_released import jobs_by_stage
    state=snapshot();mapping=jobs_by_stage(state)
    # Workers finish the in-flight episode, persist it, then exit. No cancellation.
    graceful=[]
    for job,stage in mapping.items():
        if stage=='central_fixed_lit_circle_five':
            subprocess.run(['scancel','--signal=USR1','--full',job],check=True);graceful.append(job)
    for pid in targets:
        try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    time.sleep(1)
    for pid in targets:
        try:os.kill(pid,0);os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    with (HERE/'coordinator.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator_active.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'ACTIVE_ENTRYPOINT.json',dict(utc=utc(),coordinator='coordinator_active.py',pid=process.pid,worker='worker_active.py',submit='submit_active.py',downstream='downstream_active.py',optional='optional_delivery_active.py',final_reports='final_reports_active.py',resources='resources_released.py',total_cpu=18,total_gpu_shards=2))
    write(HERE/'CENTRAL_CERTIFICATE_TRANSITION.json',dict(utc=utc(),old_coordinator_tree=sorted(targets),new_coordinator_pid=process.pid,gracefully_retiring_jobs=graceful,cancelled_jobs=[],inflight_episodes_preserved=True,completed_raw_and_unknown_never_rerun=True))
    with (HERE/'ITERATION_LOG.md').open('a') as stream:
        stream.write('\n2026-10-21 13:00后（新加坡）：只读剖析圆形集中式优先级规划实例0、state8、固定η13的已完成3592秒轨迹，99.7%时间用于每10步的原路径搜索。原已封存的reservation_certificate_v2和central_reservation_reference_v2保留原联合投影/NNLS/unknown语义。一个慢重规划点输出逐位一致，26.58秒降至0.09秒；整局2000步全部位置、速度、动作、名义、安全速度、清距、联合锥、改写标志、起作用行和终止结果逐位一致，3592.02秒降至9.85秒，147个必要条件证书。该回放只验证已有证据，不进入研究样本/数据库。新增worker_active与操作入口，旧封存文件不改；四个当前圆形集中式worker仅收USR1，完成当前局并落盘后退出，其他worker不动；后续缺失请求使用等价后端，UID/seed不变，上限仍18 CPU/2 GPU。\n')
    print(dict(coordinator_pid=process.pid,graceful_jobs=graceful,resource_snapshot=state),flush=True)

if __name__=='__main__':main()

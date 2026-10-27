"""Replace local schedulers; scientific workers drain only after healthy restart."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023 import snapshot
from coordinator_flat_static import jobs_by_stage
import signal,subprocess,time,fcntl

EQ='CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json'
CODE='CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json'
PLAN='CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.json'


def process(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.read_bytes().replace(b'\0',b' ').decode().strip() if p.exists() else ''


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt(EQ);verify_receipt(CODE)
    assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
    assert read(HERE/'STATUS.json')['phase']=='scene_learning_pipeline'
    assert not (HERE/PLAN).exists(),'Activation already planned; inspect status before resume'
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
    old=[(read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'],entry['watchdog']), (read(HERE/'COORDINATOR_PID.json')['pid'],entry['coordinator']), (read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid'],entry['downstream'])]
    for pid,name in old:assert process(pid)==sys.executable+' -B '+str(HERE/name),(pid,process(pid))
    state=snapshot();mapping=jobs_by_stage(state)
    ids=sorted(j for j,s in mapping.items() if s=='learning_labels_revised_20261021_lit_circle_five')
    assert ids and len(ids)<=23 and state['scheduled_cpus']<=23,state
    evidence=[]
    for p in sorted((HERE/'learning_labels_revised_20261021_lit_circle_five'/'raw').glob('*.json')):
        r=read(p);evidence.append(dict(raw=str(p),raw_sha256=sha(p),trace=r.get('trace_path'),trace_sha256=r.get('trace_sha256')))
    write(HERE/'CBS_START_SIGNATURE_PRE_SWITCH_EVIDENCE_20261022.json',dict(utc=utc(),rows=evidence,raw_count=len(evidence),existing_files_will_not_be_overwritten=True))
    write(HERE/'CBS_START_SIGNATURE_PRE_SWITCH_ENTRY_20261022.json',entry)
    write(HERE/PLAN,dict(utc=utc(),old_managers=old,old_worker_jobs=ids,resource_snapshot=state,worker_signal='USR1 only after new local coordinator/continuation healthy',no_hard_worker_cancellation=True,existing_raw_count=len(evidence)))
    seal('CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.receipt.json',[HERE/PLAN,HERE/'CBS_START_SIGNATURE_PRE_SWITCH_EVIDENCE_20261022.json',HERE/'CBS_START_SIGNATURE_PRE_SWITCH_ENTRY_20261022.json',Path(__file__)],no_scientific_case_changes=True)
    for pid,name in old:
        assert process(pid)==sys.executable+' -B '+str(HERE/name)
        os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+15
    while any(process(pid)==sys.executable+' -B '+str(HERE/name) for pid,name in old):
        assert time.monotonic()<deadline,'Old local managers did not terminate; no workers signalled'
        time.sleep(.25)
    for name in ('CORRECTION_COORDINATOR.lock','CORRECTION_CONTINUATION.lock','SCHEDULER_WATCHDOG.lock'):
        with (HERE/name).open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from build_start_signature_pipeline_20261022 import COPIES
    names={a:b for a,b in COPIES.items()}
    new=dict(entry)
    for key in ('coordinator','worker','submit','downstream','optional','final_reports','watchdog','optional_worker','scene_learning_scheduler','optional_submit'):
        new[key]=names[entry[key]]
    new.update(utc=utc(),downstream_receipt=CODE,optional_receipt=CODE,native_backend='tiled_planning_start_signature_v2_20261022.py',computational_equivalence_receipt=EQ,computation_acceleration='Exact low-level A* start-cell and first-edge-predicate-signature cache; original search unchanged on cache miss',state='Restarting exact-computation variant; unchanged scientific identities')
    write(HERE/'ACTIVE_ENTRYPOINT.json',new)
    with (HERE/'coordinator_correction_20261021.log').open('a') as out:
        root=subprocess.Popen([sys.executable,'-B',str(HERE/new['coordinator'])],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    new['pid']=root.pid;write(HERE/'ACTIVE_ENTRYPOINT.json',new)
    write(HERE/'CBS_START_SIGNATURE_SWITCH_STATUS_20261022.json',dict(utc=utc(),coordinator_pid=root.pid,old_worker_jobs=ids,scientific_workers_signalled=False,phase='waiting_new_coordinator_and_circle_continuation'))
    print('New root',root.pid,'started; all existing scientific workers continue untouched',flush=True)


if __name__=='__main__':main()

"""Replace local managers after sealed gates; never signal scientific workers."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023_quota18_now_20261022 import snapshot
import subprocess,time,signal,fcntl

EQ='PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json'
CODE='ANALYSIS_SUPPLEMENT_DELIVERY_CLARIFIED_20261022.receipt.json'
PREFIX='ANALYSIS_SUPPLEMENT_DELIVERY_DEPLOYMENT_20261022'


def cmd(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.read_bytes().replace(b'\0',b' ').decode().strip() if p.exists() else ''


def start(script,log):
    with (HERE/log).open('a') as out:return subprocess.Popen([sys.executable,'-B',str(HERE/script)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)


def stop(pid,script):
    expected=sys.executable+' -B '+str(HERE/script)
    if cmd(pid)==expected:os.kill(pid,signal.SIGTERM)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt(EQ);verify_receipt(CODE)
    assert not (HERE/(PREFIX+'.receipt.json')).exists(),'Already deployed'
    assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
    assert read(HERE/'STATUS.json')['phase']=='scene_learning_pipeline','Wait for a stable learning scheduling loop'
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
    old=[(read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'],entry['watchdog']),(read(HERE/'COORDINATOR_PID.json')['pid'],entry['coordinator']),(read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid'],entry['downstream'])]
    for pid,name in old:assert cmd(pid)==sys.executable+' -B '+str(HERE/name),(pid,cmd(pid))
    # A current action process can own the official merger or be sealing a
    # model. Do not interrupt that boundary or create a second writer.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            words=proc.joinpath('stat').read_text().split(') ',1)[1].split();parent=int(words[1])
            args=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if parent==old[2][0] and len(args)>2 and args[1]==b'-B':raise AssertionError(('Current continuation action active; retry after completion',int(proc.name)))
            if len(args)>2 and args[2]==str(HERE/'db_merge.py').encode():raise AssertionError('Official merger active; wait before manager switch')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    pre=snapshot();rows=[]
    stage=HERE/'learning_labels_revised_20261021_lit_circle_five'
    for p in sorted((stage/'raw').glob('*.json')):
        r=read(p);rows.append(dict(raw=str(p),raw_sha256=sha(p),trace=r['trace_path'],trace_sha256=r['trace_sha256']))
    write(HERE/(PREFIX+'_PRE_ENTRY.json'),entry)
    write(HERE/(PREFIX+'_PRE_EVIDENCE.json'),dict(utc=utc(),rows=rows))
    write(HERE/(PREFIX+'_PLAN.json'),dict(utc=utc(),old_managers=old,pre_resources=pre,scientific_worker_signals=0,label_workers_and_their_caches_preserved=True))
    seal(PREFIX+'_PLAN.receipt.json',[HERE/(PREFIX+'_PRE_ENTRY.json'),HERE/(PREFIX+'_PRE_EVIDENCE.json'),HERE/(PREFIX+'_PLAN.json'),Path(__file__)],scientific_worker_signals=0)
    COPIES=read(HERE/'ANALYSIS_SUPPLEMENT_DELIVERY_MAP_20261022.json')['copies']
    changed={a:b for a,b in COPIES.items()};new=dict(entry);known=[]
    try:
        for pid,name in old:stop(pid,name)
        deadline=time.monotonic()+15
        while any(cmd(pid)==sys.executable+' -B '+str(HERE/name) for pid,name in old):
            assert time.monotonic()<deadline,'Old local manager has not exited';time.sleep(.25)
        for name in ('CORRECTION_COORDINATOR.lock','CORRECTION_CONTINUATION.lock','SCHEDULER_WATCHDOG.lock'):
            with (HERE/name).open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in ('coordinator','worker','submit','downstream','optional','final_reports','watchdog','optional_worker','scene_learning_scheduler','optional_submit'):new[key]=changed.get(entry[key],entry[key])
        new.update(utc=utc(),downstream_receipt=CODE,optional_receipt=CODE,native_backend='tiled_planning_pa_memo_query_disk_candidate_20261022.py',label_native_backend='tiled_planning_pa_memo_candidate_20261022.py',label_worker='worker_correction_pa_memo_label_20261022.py',computational_equivalence_receipt=EQ,previous_computational_equivalence_receipt=entry.get('computational_equivalence_receipt'),state='Training-selected primary comparator,combined safety-global taxonomy,primary matched-evaluation figures;18CPU total',computation_acceleration='Sealed exact PA memo and CBS shared caches retained')
        write(HERE/'ACTIVE_ENTRYPOINT.json',new)
        root=start(new['coordinator'],'coordinator_correction_20261021.log');known.append((root.pid,new['coordinator']));new['pid']=root.pid;write(HERE/'ACTIVE_ENTRYPOINT.json',new)
        deadline=time.monotonic()+300;watcher=None
        while True:
            assert root.poll() is None,'New coordinator exited'
            assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
            if read(HERE/'COORDINATOR_PID.json')['pid']==root.pid and watcher is None:
                watcher=start(new['watchdog'],'scheduler_watchdog_query_disk_20261022.log');known.append((watcher.pid,new['watchdog']))
            if watcher is not None:assert watcher.poll() is None,'New watchdog exited'
            child=read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid']
            duty=read(HERE/'SCHEDULER_DUTY.json')
            if cmd(child)==sys.executable+' -B '+str(HERE/new['downstream']) and read(HERE/'STATUS.json')['phase']=='scene_learning_pipeline' and duty['utc']>read(HERE/(PREFIX+'_PLAN.json'))['utc']:
                known.append((child,new['downstream']));break
            assert time.monotonic()<deadline,'New continuation startup timeout'
            time.sleep(1)
        # Existing research raw and traces stay immutable. New finished cases
        # during the switch are allowed and are handled by the same queue.
        for r in rows:assert sha(r['raw'])==r['raw_sha256'] and sha(r['trace'])==r['trace_sha256'],r['raw']
        verify_receipt(EQ);verify_receipt(CODE)
        write(HERE/(PREFIX+'.json'),dict(utc=utc(),passed=True,coordinator_pid=root.pid,continuation_pid=child,watchdog_pid=watcher.pid,scientific_worker_signals=0,preexisting_raw_and_trace_hashes_preserved=len(rows),post_resources=snapshot(),active_entry_snapshot=read(HERE/'ACTIVE_ENTRYPOINT.json'),current_label_batch_unchanged=True))
        seal(PREFIX+'.receipt.json',[HERE/(PREFIX+'.json'),HERE/(PREFIX+'_PLAN.receipt.json'),HERE/EQ,HERE/CODE,Path(__file__)],passed=True,no_scientific_workers_interrupted=True)
        with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+'：仅澄清交付文本和主图：全局失败分类不自动算死锁证据，HANDOFF加入训练最佳固定定义，主图改用匹配评估结果，旧固定图标附图。无worker信号，无数值或物理变化。见'+PREFIX+'.receipt.json。\n')
        print('PASS: final prose and figure clarification active;workers untouched',flush=True)
    except BaseException as exc:
        for pid,name in reversed(known):stop(pid,name)
        child=read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid']
        if new.get('downstream')!=entry['downstream']:stop(child,new['downstream'])
        # Startup can be in one of the idempotent official merge actions.
        # Stop that local action before restoring a manager which can merge.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                if cmd(int(proc.name))==sys.executable+' -B '+str(HERE/'db_merge.py'):os.kill(int(proc.name),signal.SIGTERM)
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        time.sleep(1)
        failure=HERE/'COORDINATOR_ACTION_ERROR.json'
        if failure.exists():failure.rename(HERE/(PREFIX+'_ACTION_ERROR_'+str(time.time_ns())+'.json'))
        write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
        restored=start(entry['coordinator'],'coordinator_correction_20261021.log');entry['pid']=restored.pid;entry['utc']=utc();write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
        for _ in range(15):
            if read(HERE/'COORDINATOR_PID.json')['pid']==restored.pid:break
            time.sleep(1)
        guard=start(entry['watchdog'],'scheduler_watchdog_query_disk_rollback_20261022.log')
        write(HERE/(PREFIX+'_ROLLBACK.json'),dict(utc=utc(),error=repr(exc),restored_coordinator_pid=restored.pid,restored_watchdog_pid=guard.pid,scientific_worker_signals=0))
        raise


if __name__=='__main__':main()

"""Finish the sealed switch after coordinator/continuation prove healthy."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023 import snapshot
import subprocess,time


def cmd(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.read_bytes().replace(b'\0',b' ').decode().strip() if p.exists() else ''


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.receipt.json')
    verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
    assert not (HERE/'CBS_START_SIGNATURE_DRAIN_STARTED_20261022.receipt.json').exists()
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json');root=read(HERE/'COORDINATOR_PID.json')['pid']
    assert cmd(root)==sys.executable+' -B '+str(HERE/entry['coordinator'])
    with (HERE/'scheduler_watchdog_start_signature_20261022.log').open('a') as log:
        watcher=subprocess.Popen([sys.executable,'-B',str(HERE/entry['watchdog'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+300
    while True:
        assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
        assert cmd(root)==sys.executable+' -B '+str(HERE/entry['coordinator'])
        assert watcher.poll() is None,'New watchdog failed'
        child=read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid']
        duty=read(HERE/'SCHEDULER_DUTY.json')
        if cmd(child)==sys.executable+' -B '+str(HERE/entry['downstream']) and read(HERE/'STATUS.json')['phase']=='scene_learning_pipeline' and 'learning_labels_revised_20261021_lit_circle_five' in duty.get('pending_rollout_stages',[]) and duty['utc']>read(HERE/'CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.json')['utc']:break
        assert time.monotonic()<deadline,'Continuation not healthy in300s; no scientific workers signalled'
        time.sleep(1)
    plan=read(HERE/'CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.json');state=snapshot();live={j['job_id'] for j in state['jobs']}
    signalled=[]
    for job in plan['old_worker_jobs']:
        if job not in live:continue
        r=subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
        assert r.returncode==0,(job,r.stderr);signalled.append(job)
    write(HERE/'CBS_START_SIGNATURE_POST_SWITCH_ENTRY_20261022.json',read(HERE/'ACTIVE_ENTRYPOINT.json'))
    write(HERE/'CBS_START_SIGNATURE_DRAIN_STARTED_20261022.json',dict(utc=utc(),coordinator_pid=root,continuation_pid=child,watchdog_pid=watcher.pid,signalled_old_workers=signalled,signal='USR1',hard_cancellations=0,resource_snapshot=state,policy_unchanged=True))
    seal('CBS_START_SIGNATURE_DRAIN_STARTED_20261022.receipt.json',[HERE/'CBS_START_SIGNATURE_DRAIN_STARTED_20261022.json',HERE/'CBS_START_SIGNATURE_POST_SWITCH_ENTRY_20261022.json',Path(__file__)],healthy_new_local_managers=True,old_workers_finish_current_episode=True)
    write(HERE/'CBS_START_SIGNATURE_SWITCH_STATUS_20261022.json',dict(utc=utc(),phase='old_workers_draining_new_workers_admitted_as_capacity_frees',coordinator_pid=root,continuation_pid=child,watchdog_pid=watcher.pid,old_worker_jobs=plan['old_worker_jobs'],scientific_workers_signalled=True))
    print('Healthy new coordinator+continuation+watchdog;',len(signalled),'old workers instructed to finish current episode; no hard cancellation',flush=True)


if __name__=='__main__':main()

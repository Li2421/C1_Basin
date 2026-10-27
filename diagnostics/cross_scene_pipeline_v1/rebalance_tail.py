"""Release head-of-line blocking without changing or repeating a scientific request."""
from common import *
import signal,subprocess
from schedule_resume import jobs

def is_worker(pid):
    try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return False
    return 'diagnostics/cross_scene_pipeline_v1/worker_resume.py main'in cmd

def main():
    assert not(HERE/'TAIL_REBALANCE.receipt.json').exists()
    # Terminate only our sleeping coordinator. Existing workers keep their exact
    # environment/actor/history; no rollout is cancelled or started again.
    out=subprocess.check_output(['ps','-u','zhihan','-o','pid=,args='],text=True)
    coordinators=[]
    for line in out.splitlines():
        pid,cmd=line.strip().split(None,1)
        if cmd.startswith('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_pipeline_v1/evaluate_resume.py'):
            coordinators.append(int(pid))
    assert len(coordinators)==1,coordinators
    os.kill(coordinators[0],signal.SIGTERM)
    q=HERE/'QUEUES/main';index=read(q/'INDEX.json');live={r['id']:r for r in jobs()if r['name']=='xpl_main'};guards=[]
    for entry in index:
        if(q/'done'/entry['group']).exists():continue
        p=q/'claims'/entry['group']/'OWNER.json'
        if not p.exists():continue
        owner=read(p)
        if owner['job']not in live or not is_worker(owner['pid']):continue
        assert sha(entry['path'])==entry['sha256'];group=read(entry['path'])
        todo=[c for c in group if not(HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()]
        if not todo:continue
        # All PA/new-scene groups remain owned by the original process through
        # completion. Only legacy circle workers are asked to finish the current
        # rollout then return; their numerical runtime ignores the PAUSE flag.
        circle_only=all(c['state']['scene']=='lit_circle_five'for c in todo)
        current_file=HERE/'worker_status'/f"{owner['job']}.json"
        if circle_only:
            os.kill(owner['pid'],signal.SIGSTOP)
            try:
                for _ in range(100):
                    state=Path(f"/proc/{owner['pid']}/status").read_text()
                    if '\nState:\tT'in state or '\nState:\tt'in state:break
                    time.sleep(.01)
                else:raise RuntimeError(('WORKER_NOT_STOPPED',owner))
                current=read(current_file)['task']['rollout_uid'];ids=[c['rollout_uid']for c in group]
                if current in ids:
                    at=ids.index(current);protected=ids[at:at+2]
                else:protected=ids[:2]
                # Protect one additional request for the Python signal-delivery
                # boundary between STOP check and status-file replacement.
                os.kill(owner['pid'],signal.SIGUSR1)
                guard=dict(owner=owner,group=entry['group'],protected=protected,
                    mode='circle_finish_current_then_exit',no_running_rollout_restart=True)
            finally:os.kill(owner['pid'],signal.SIGCONT)
        else:
            guard=dict(owner=owner,group=entry['group'],protected=[c['rollout_uid']for c in group],
                mode='original_group_continues_unchanged',no_signal_sent=True)
        guards.append(guard)
    missing=[c for c in read(HERE/'main_MISSING_CASES.json')if not(HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()]
    assert missing
    write(HERE/'TAIL_CASES.json',missing)
    plan=HERE/'TAIL_PLANNED_ROLLOUTS.json'
    write(plan,dict(requests=[{k:c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key')}for c in missing]))
    subprocess.run([sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(plan),'--output',str(HERE/'TAIL_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    pf=read(HERE/'TAIL_OFFICIAL_PREFLIGHT.json');assert pf['summary']['ambiguous']==0
    write(HERE/'TAIL_REBALANCE.receipt.json',dict(utc=utc(),reason='User requested acceleration; idle CPUs caused by unstarted requests behind long running requests in coarse groups.',
        old_coordinator=coordinators[0],original_guards=guards,remaining_requests=len(missing),cases_sha256=sha(HERE/'TAIL_CASES.json'),
        preflight_sha256=sha(HERE/'TAIL_OFFICIAL_PREFLIGHT.json'),no_models_proposals_choices_or_parameters_changed=True,
        no_running_rollout_restarted=True,no_no_result_age_cancel_rule=True,old_group_files_unchanged=True))
    print('Tail rebalance prepared:',len(missing),'requests;',len(guards),'original group guards.',flush=True)

if __name__=='__main__':main()

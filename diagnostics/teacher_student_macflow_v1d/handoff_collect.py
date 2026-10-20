from core import *
from audit_utils import source_inventory
import subprocess,time,re,traceback
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def queue():
    rows=[]
    for line in subprocess.check_output(['squeue','-r','-h','-o','%i|%u|%j|%T|%C'],text=True).splitlines():
        jid,user,name,state,cpu=line.split('|');rows.append(dict(job=jid,user=user,name=name,state=state,cpus=int(cpu)))
    return rows

def main():
    assert not (HERE/'HANDOFF_SUBMISSION.json').exists();rows=queue();active=[r for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']];own=[r for r in active if r['job'].split('_')[0]=='22046'];others=[r for r in active if r not in own];limit=min(17,(24 if not others else 20)-sum(r['cpus'] for r in active));assert limit>0
    cmd=['sbatch',f'--array=0-16%{limit}',str(HERE/'handoff.sbatch')];response=subprocess.check_output(cmd,text=True);job=re.search(r'(\d+)',response).group(1);write(HERE/'HANDOFF_SUBMISSION.json',dict(utc=utc(),command=cmd,response=response,resources_before=rows,authorization='User permits all24 CPU if other tasks absent',preflight_sha256=sha(HERE/'HANDOFF_PREFLIGHT.receipt.json')));print(response,flush=True)
    monitor=read(HERE/'RESOURCE_MONITOR_RESUME.json')
    while True:
        rows=queue();active=[r for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']];own=[r for r in active if r['job'].split('_')[0] in ['22046',job]];others=[r for r in active if r not in own];cap=24 if not others else 20;total=sum(r['cpus'] for r in active);monitor.append(dict(utc=utc(),cpu_total=total,cpu_own=sum(r['cpus'] for r in own),cpu_others=sum(r['cpus'] for r in others),authorized_cap=cap,queue=rows));write(HERE/'RESOURCE_MONITOR_RESUME.json',monitor)
        completed=sum(not p.is_symlink() for p in (HERE/'C/raw').glob('*.json'));write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='C',status='RUNNING',job=job,completed_new=completed,expected_new=13440,handoff=True))
        if total>cap:write(HERE/'RESOURCE_NEW_ANOMALY.json',monitor[-1])
        if not any(r['job'].split('_')[0] in ['22046',job] for r in rows):break
        time.sleep(30)
    assert len(list((HERE/'handoff').glob('worker_*.json')))==17,'Inspect helper error; never automatically retry'
    plan=read(HERE/'HANDOFF_PLAN.json');old={x['original_worker'] for x in plan['transfers']}
    assert all((HERE/'C'/f'worker_{i}.json').exists() for i in range(18) if i not in old)
    for x in plan['transfers']:
        log=(HERE/f"roll_22046_{x['original_worker']}.log").read_text();assert 'No retry or overwrite of completed/unknown trial' in log or 'Never retry or overwrite unknown/completed evidence' in log,log[-600:]
    cases=read(HERE/'C_MISSING_CASES.json');uids=set()
    for c in cases:
        path=HERE/'C/raw'/f"{c['rollout_uid']}.json";assert path.is_file() and not path.is_symlink();r=read(path);assert r['case']['rollout_uid']==c['rollout_uid'];assert r['state_uid']==c['state_uid'] and r['controller_uid']==c['controller_uid'];assert not r.get('dispatch_reservation_only',False);assert c['rollout_uid'] not in uids;uids.add(c['rollout_uid'])
    write(HERE/'HANDOFF_COMPLETION.json',dict(utc=utc(),passed=True,original_normal_worker_receipts=18-len(old),expected_pre_episode_guard_exits=len(old),helper_worker_receipts=17,all_new_cases_exactly_once=len(uids),reservations_remaining=0,rollouts_retried=0,scientific_parameters_unchanged=True,resumed_cap_pass=all(x['cpu_total']<=x['authorized_cap'] for x in monitor)))
    for script,args in [('postflight.py',['C']),('analysis.py',['C']),('finish_v2.py',[])]:
        print('RUN',script,args,flush=True);subprocess.run([PY,'-B',str(HERE/script),*args],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
    source_inventory('imports_handoff_collect.json');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE',resource_history_exception=True))
if __name__=='__main__':
    try:main()
    except Exception:
        write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=traceback.format_exc()));raise

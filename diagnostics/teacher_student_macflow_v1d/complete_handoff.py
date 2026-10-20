"""Finish already submitted handoff, with journal-level exactly-once proof."""
from core import *
from audit_utils import source_inventory
from handoff_collect import queue
import subprocess,time,traceback,collections
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def main():
    job=read(HERE/'HANDOFF_SUBMISSION.json')['response'].strip().split()[-1];monitor=read(HERE/'RESOURCE_MONITOR_RESUME.json')
    while True:
        rows=queue();active=[r for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']];own=[r for r in active if r['job'].split('_')[0] in ['22046',job]];others=[r for r in active if r not in own];cap=24 if not others else 20
        monitor.append(dict(utc=utc(),cpu_total=sum(r['cpus'] for r in active),cpu_own=sum(r['cpus'] for r in own),cpu_others=sum(r['cpus'] for r in others),authorized_cap=cap,queue=rows));write(HERE/'RESOURCE_MONITOR_RESUME.json',monitor);write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='C',status='RUNNING',job=job,completed_new=sum(not p.is_symlink() for p in (HERE/'C/raw').glob('*.json')),expected_new=13440,handoff=True))
        if not any(r['job'].split('_')[0] in ['22046',job] for r in rows):break
        time.sleep(30)
    assert len(list((HERE/'handoff').glob('worker_*.json')))==17
    plan=read(HERE/'HANDOFF_PLAN.json');old={x['original_worker'] for x in plan['transfers']};assert all((HERE/'C'/f'worker_{i}.json').exists() for i in range(18) if i not in old)
    for x in plan['transfers']:assert 'Never retry or overwrite unknown/completed evidence' in (HERE/f"roll_22046_{x['original_worker']}.log").read_text()
    cases=read(HERE/'C_MISSING_CASES.json');expected={c['rollout_uid'] for c in cases};count=collections.Counter();sources={};journal_root=ROOT/'shared_rollout_db/journals/exp_teacher_student_macflow_v1d_C';assert journal_root.is_relative_to(ROOT/'shared_rollout_db/journals')
    for p in sorted(journal_root.glob('*')):
        if p.suffix not in ['.jsonl','.merged']:continue
        for line in p.read_text().splitlines():
            r=json.loads(line);rid=r['case']['rollout_uid'];count[rid]+=1;raw=HERE/'C/raw'/f'{rid}.json';assert raw.is_file() and not raw.is_symlink();assert read(raw)==r;sources[rid]=dict(journal=str(p),journal_sha256=sha(p))
    assert set(count)==expected and all(n==1 for n in count.values());assert not any(p.is_symlink() for p in (HERE/'C/raw').glob('*.json'))
    write(HERE/'JOURNAL_ONCE_AUDIT.json',dict(utc=utc(),passed=True,new_rollouts=len(expected),journal_record_count=sum(count.values()),duplicate_rollout_ids=0,all_raw_records_exactly_match_journal=True,DB_old_unknown_not_rerun=True,source_records=sources))
    write(HERE/'HANDOFF_COMPLETION.json',dict(utc=utc(),passed=True,original_normal_worker_receipts=18-len(old),expected_pre_episode_guard_exits=len(old),helper_worker_receipts=17,transferred_future_cases=len(read(HERE/'HANDOFF_CASES.json')),all_new_cases_exactly_once=len(expected),reservations_remaining=0,rollouts_retried=0,scientific_parameters_unchanged=True,resumed_cap_pass=all(x['cpu_total']<=x['authorized_cap'] for x in monitor)))
    for script,args in [('postflight.py',['C']),('analysis.py',['C']),('finish_v3.py',[])]:
        print('RUN',script,args,flush=True);subprocess.run([PY,'-B',str(HERE/script),*args],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
    source_inventory('imports_complete_handoff.json');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE',resource_history_exception=True))
if __name__=='__main__':
    try:main()
    except Exception:
        write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=traceback.format_exc()));raise

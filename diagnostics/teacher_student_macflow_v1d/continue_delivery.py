"""Observe existing C array; never resubmit or rerun any rollout."""
from core import *
from audit_utils import source_inventory
import subprocess,time,traceback
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def main():
    assert (HERE/'INTEGRITY_RESUME.json').exists();a=read(HERE/'INTEGRITY_START.json');b=read(HERE/'INTEGRITY_RESUME.json');assert all(a[k]==b[k] for k in ['official','poc','round30','round30b','round30c']);assert all(a['basin'][k]==b['basin'][k] for k in ['head','diff_sha256'])
    write(HERE/'RESUME_20261016.json',dict(utc=utc(),previous_status=read(HERE/'RUN_STATUS.json'),action='Observe existing job22046 only; no rollout resubmissions. Prior resource breach retained; scientific pipeline unchanged.',resource_authorization='2026-10-16 user permits all CPU if no other tasks use them. Existing7 workers continue unchanged.',protected_resume_audit_pass=True))
    monitor=[]
    while True:
        lines=subprocess.check_output(['squeue','-r','-h','-o','%i|%u|%j|%T|%C'],text=True).splitlines();rows=[]
        for line in lines:
            jid,user,name,state,cpu=line.split('|');rows.append(dict(job=jid,user=user,name=name,state=state,cpus=int(cpu)))
        running=[r for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']];own=[r for r in running if r['job'].split('_')[0]=='22046'];others=[r for r in running if r not in own];cap=24 if not others else 20;total=sum(r['cpus'] for r in running)
        monitor.append(dict(utc=utc(),cpu_total=total,cpu_own=sum(r['cpus'] for r in own),cpu_others=sum(r['cpus'] for r in others),authorized_cap=cap,queue=rows));write(HERE/'RESOURCE_MONITOR_RESUME.json',monitor);write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='RUNNING' if own else 'VERIFYING',phase='C',job='22046',completed_new=len(list((HERE/'C/raw').glob('*.json'))),expected_new=13440))
        if total>cap:
            # No new submissions. Explicitly retain resource anomaly for audit.
            write(HERE/'RESOURCE_NEW_ANOMALY.json',monitor[-1])
        if not any(r['job'].split('_')[0]=='22046' for r in rows):break
        time.sleep(30)
    assert len(list((HERE/'C').glob('worker_*.json')))==18,'Do not rerun incomplete/unknown trials; inspect failed workers'
    for script,args in [('postflight.py',['C']),('analysis.py',['C']),('finish_v2.py',[])]:
        print('RUN',script,args,flush=True);subprocess.run([PY,'-B',str(HERE/script),*args],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
    source_inventory('imports_continue_delivery.json');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE',resource_history_exception=True))
if __name__=='__main__':
    try:main()
    except Exception:
        write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=traceback.format_exc()));raise

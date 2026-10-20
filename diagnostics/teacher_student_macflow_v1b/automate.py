"""One-shot phase coordinator; no retries of experiments or unknowns."""
from core import *
import subprocess,time,traceback,re
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
PLOTS='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'
monitor=[]
def call(script,*args,python=PYTHON):
    cmd=[python,'-B',str(HERE/script),*args];print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
def wait_phase(phase):
    receipt=read(HERE/f'{phase}_SUBMISSION.json');job=re.search(r'Submitted batch job (\d+)',receipt['result']).group(1);throttle=receipt['max_concurrency'];write(HERE/'AUTOMATION_STATUS.json',dict(utc=utc(),phase=phase,job=job,status='RUNNING'))
    while True:
        lines=subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%C'],text=True).splitlines();active=[];present=False
        for line in lines:
            jid,name,status,cpus=line.split('|');isown=jid.split('_')[0]==job;present=present or isown
            if status in ['RUNNING','COMPLETING','CONFIGURING']:active.append(dict(job=jid,name=name,status=status,cpus=int(cpus),this_phase=isown))
        other=sum(r['cpus'] for r in active if not r['this_phase']);target=max(1,min(20,20-other));total=sum(r['cpus'] for r in active);snapshot=dict(utc=utc(),phase=phase,job=job,active_cpu=total,other_cpu=other,desired_array_throttle=target,active=active);monitor.append(snapshot);write(HERE/'RESOURCE_MONITOR.json',monitor)
        if total>20:raise RuntimeError('User total active CPUs exceeds20; no further submission; inspect concurrent resource changes')
        if not present:break
        if target!=throttle:
            subprocess.run(['scontrol','update','JobId='+job,'ArrayTaskThrottle='+str(target)],check=True);throttle=target;print('THROTTLE',phase,target,flush=True)
        time.sleep(30)
    if phase=='probe':
        assert len(list((HERE/'probe/raw').glob('*.json')))==len(read(HERE/'probe_CASES.json'))
        assert all((HERE/f'imports_probe_{i}.json').exists() for i in range(20))
    else:assert all((HERE/phase/f'worker_{i}.json').exists() for i in range(20)),('Worker failed; no retry',phase)

def main():
    assert not (HERE/'AUTOMATION_STATUS.json').exists(),'Do not restart phases or overwrite receipts'
    wait_phase('pilot');call('stage.py','probe')
    if read(HERE/'CAPABILITY_GATE.json')['eligible']:
        wait_phase('probe');call('stage.py','mainphase');wait_phase('main');call('postflight.py','main')
    call('analysis.py');call('plots.py',python=PLOTS);call('validate.py');call('audit_extra.py');call('integrity.py','end');call('deliver.py');write(HERE/'AUTOMATION_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE'))
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True);write(HERE/'AUTOMATION_STATUS.json',dict(utc=utc(),status='STOPPED',error=error));(HERE/'STOPPED.md').write_text('Coordinator stopped without retrying or changing any existing source.\n\n```\n'+error+'\n```\n');raise

"""Compact immutable group files and recovery of infrastructure-abandoned claims."""
from experiment import *
import subprocess
import re

def build(stage):
    cases=read(HERE/f'{stage}_MISSING_CASES.json');groups={}
    for c in cases:groups.setdefault(uid('group',[c['state_uid'],c['controller_uid'],c['eta_uid']]),[]).append(c)
    queue=HERE/stage/'queue';queue.mkdir(parents=True,exist_ok=True);index=[]
    for group,values in groups.items():
        path=queue/f'{group}.json'
        if not path.exists():write(path,sorted(values,key=lambda c:c['seed']))
        index.append(dict(group=group,file=str(path),sha256=sha(path),rollout_uids=[c['rollout_uid'] for c in values]))
    write(HERE/stage/'GROUP_INDEX.json',index)
    seal(f'{stage}/STREAM_QUEUE.receipt.json',[HERE/stage/'GROUP_INDEX.json',HERE/f'{stage}_MISSING_CASES.json'],request_semantics_unchanged=True)
    print(stage,'groups',len(groups),flush=True)

def recover(stage):
    recovered=[];claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True)
    for path in claims.iterdir():
        if not path.is_dir() or not (path/'OWNER.json').exists():continue
        owner=read(path/'OWNER.json')
        if all((HERE/stage/'raw'/f'{rid}.json').exists() for rid in owner['rollout_uids']):continue
        out=subprocess.run(['scontrol','show','job',str(owner['job']),'-o'],capture_output=True,text=True)
        m=re.search(r'JobState=(\S+)',out.stdout);state=m.group(1) if m else 'not_in_slurm'
        if state in ('RUNNING','COMPLETING','PENDING','SUSPENDED','CONFIGURING'):continue
        if state=='not_in_slurm':
            process=Path(f"/proc/{owner['pid']}/cmdline")
            if process.exists() and stage.encode() in process.read_bytes():continue
        archive=HERE/stage/'claims_abandoned';archive.mkdir(exist_ok=True);target=archive/(path.name+'_'+str(time_ns()))
        path.rename(target);recovered.append(dict(group=path.name,owner=owner,slurm_state=state,archive=str(target),only_unfinished_resumed=True))
    write(HERE/stage/f'INFRA_RECOVERY_{time_ns()}.json',dict(utc=utc(),claims=recovered));print(stage,'released',len(recovered),flush=True)

def time_ns():return __import__('time').time_ns()
if __name__=='__main__':build(sys.argv[2]) if sys.argv[1]=='build' else recover(sys.argv[2])

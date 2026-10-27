"""Small immutable chunks, atomic claims and infrastructure-only recovery."""
from experiment import *
from registry import verify_receipt
import subprocess
import re
import time

def build(stage):
    verify_receipt(f'{stage}_PREFLIGHT.receipt.json')
    cases=read(HERE/f'{stage}_MISSING_CASES.json');queue=HERE/stage/'queue';queue.mkdir(parents=True,exist_ok=True)
    entries=[]
    for offset in range(0,len(cases),32):
        values=cases[offset:offset+32];group=f'{offset//32:06d}';path=queue/f'{group}.json'
        assert not path.exists(),path
        write(path,values)
        entries.append(dict(group=group,file=str(path),sha256=sha(path),rollout_uids=[c['rollout_uid'] for c in values]))
    write(HERE/stage/'GROUP_INDEX.json',entries)
    seal(f'{stage}/WORK_QUEUE.receipt.json',[HERE/stage/'GROUP_INDEX.json',HERE/f'{stage}_MISSING_CASES.json'],bounded_requests_per_chunk=32,request_semantics_unchanged=True)
    print(stage,'queue groups',len(entries),flush=True)

def recover(stage,live_jobs):
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);recovered=[]
    for path in claims.iterdir():
        if not path.is_dir() or (path/'COMPLETE.json').exists():continue
        owner=path/'OWNER.json'
        if owner.exists():
            info=read(owner)
            if str(info['job']) in live_jobs:continue
        else:
            if time.time()-path.stat().st_mtime<120:continue
            info=dict(reason='Claim interrupted before owner receipt')
        archive=HERE/stage/'claims_abandoned';archive.mkdir(exist_ok=True)
        dest=archive/f'{path.name}_{time.time_ns()}';path.rename(dest)
        recovered.append(dict(group=path.name,owner=info,archive=str(dest)))
    if recovered:write(HERE/stage/f'INFRA_RECOVERY_{time.time_ns()}.json',dict(utc=utc(),recovered=recovered,completed_raw_preserved=True,unfinished_not_unknown=True))
    return recovered

if __name__=='__main__':build(sys.argv[1])

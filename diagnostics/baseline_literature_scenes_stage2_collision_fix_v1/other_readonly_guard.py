"""Metadata-only scan of other protected roots; never opens research evidence."""
from audit import HERE,ROOT,V1,V2,write
from pathlib import Path
from datetime import datetime
import os,json,sys,time

def check(label):
    start=json.loads((HERE/'INTEGRITY_RESUME_START_20261023.json').read_text());cutoff=datetime.fromisoformat(start['utc']).timestamp()*1e9
    skip={HERE,V1,V2,ROOT/'.git',ROOT/'shared_rollout_db/journals',ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1'}
    changes=[];seen=0
    todo=[str(ROOT),str(ROOT.parent/'Basin_C1_flow_field_poc_20261004/poc'),str(ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')]
    while todo:
        folder=todo.pop()
        try:entries=os.scandir(folder)
        except FileNotFoundError:continue
        with entries:
            for e in entries:
                p=Path(e.path)
                if p in skip:continue
                if p.parent==ROOT/'shared_rollout_db' and p.name.startswith('rollout.sqlite'):continue
                try:s=e.stat(follow_symlinks=False)
                except FileNotFoundError:continue
                seen+=1
                directory=e.is_dir(follow_symlinks=False)
                if directory:todo.append(e.path)
                if s.st_mtime_ns<cutoff:continue
                if p in (ROOT/'docs',ROOT/'diagnostics',ROOT/'shared_rollout_db'):continue
                if p.parent==ROOT/'docs' and p.name.startswith('prompt_') and str(p) not in start['protected']:continue
                changes.append(dict(path=e.path,size=s.st_size,mtime_ns=s.st_mtime_ns))
    out=HERE/'READONLY_BATCH_CHECKS'/f'{time.time_ns()}_other_{label}.json';out.parent.mkdir(exist_ok=True)
    write(out,dict(passed=not changes,metadata_entries_checked=seen,changes=changes))
    assert not changes,('OTHER_READONLY_CHANGED',str(out),changes[:5])
    print('Other read-only guard passed:',label,seen,flush=True)
if __name__=='__main__':check(sys.argv[1])

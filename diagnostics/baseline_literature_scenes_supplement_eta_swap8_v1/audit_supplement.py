"""Independent opening audit and metadata-only batch boundary checks."""
from audit_base import *
import time
H=ROOT/'diagnostics/baseline_literature_scenes_stage2_collision_fix_v1'
SKIP_PREFIX=('decentralized_','teacher_student_','multiscene_')
def evidence(p):
    p=Path(p).resolve();h=sha(p)
    with (HERE/'EVIDENCE_INPUTS.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(p),sha256=h))+'\n')
    return h
def opening():
    assert not (HERE/'INTEGRITY_START.json').exists()
    assert json.loads((H/'COMPLETE.json').read_text())['protocol_execution_complete']
    seed=json.loads((H/'INTEGRITY_RESUME_START_20261023.json').read_text())
    paths={Path(p) for p in seed['protected']}
    for base in (V1,V2,H):
        for ext in ('*.py','*.md','*.hpp','*.cpp','*.so','*MANIFEST*.json','*SPLIT*.json','*QUALIFICATION*.json','*STATES*.json','*PARAMETER*.json','*MENU*.json','*COMPLETE*.json','*FIXED*.json'):
            paths.update(base.glob(ext))
    paths.update((H/'resume_backend').glob('*.py'));paths.update((H/'native_resume').glob('*'))
    paths.update((ROOT/'diagnostics/controller_loco_real_flow_v1').glob('protocol.json'))
    cutoff=time.time()-86400;parallel={}
    for p in (ROOT/'diagnostics').iterdir():
        if p in (HERE,H,V1,V2) or not p.is_dir():continue
        st=p.stat()
        if st.st_mtime>=cutoff or p.name.startswith(SKIP_PREFIX):parallel[str(p)]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
    obj=dict(utc=datetime.now(timezone.utc).isoformat(),part1_complete_sha256=sha(H/'COMPLETE.json'),protected={str(p.resolve()):sha(p) for p in sorted(paths) if p.is_file()},git={str(p):git_snapshot(p) for p in (ROOT,ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')},parallel_stat_only=parallel)
    write(HERE/'INTEGRITY_START.json',obj);print('Opening hash audit',len(obj['protected']),flush=True)
def check(label):
    start=json.loads((HERE/'INTEGRITY_START.json').read_text());cutoff=datetime.fromisoformat(start['utc']).timestamp()*1e9
    parallel={Path(p) for p in start['parallel_stat_only']}
    skip={HERE,ROOT/'.git',ROOT/'shared_rollout_db/journals'}
    changes=[];active=[];seen=0
    todo=[str(ROOT),str(ROOT.parent/'Basin_C1_flow_field_poc_20261004/poc'),str(ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')]
    while todo:
        folder=todo.pop()
        try:entries=os.scandir(folder)
        except FileNotFoundError:continue
        with entries:
            for e in entries:
                p=Path(e.path)
                if p in skip or p.name=='.git':continue
                if p.parent==ROOT/'shared_rollout_db' and p.name.startswith('rollout.sqlite'):continue
                try:s=e.stat(follow_symlinks=False)
                except FileNotFoundError:continue
                seen+=1;directory=e.is_dir(follow_symlinks=False)
                isparallel=any(p==b or b in p.parents for b in parallel)
                if directory:todo.append(e.path)
                if s.st_mtime_ns<cutoff:continue
                if p in (ROOT/'docs',ROOT/'diagnostics',ROOT/'shared_rollout_db'):continue
                if p.parent==ROOT/'docs' and p.name.startswith('prompt_') and str(p) not in start['protected']:continue
                (active if isparallel else changes).append(dict(path=e.path,size=s.st_size,mtime_ns=s.st_mtime_ns))
    out=HERE/'READONLY_BATCH_CHECKS'/f'{time.time_ns()}_{label}.json';out.parent.mkdir(exist_ok=True)
    write(out,dict(passed=not changes,metadata_entries_checked=seen,changes=changes,parallel_stat_changes=active))
    assert not changes,('READONLY_CHANGED',str(out),changes[:5]);print('Read-only stat passed',label,seen,flush=True)
if __name__=='__main__':opening() if sys.argv[1]=='start' else check(sys.argv[1])

"""Stat-only guard; prune parallel roots without reading their contents."""
from audit_supplement import *
def check(label):
    start=json.loads((HERE/'INTEGRITY_START.json').read_text());cutoff=datetime.fromisoformat(start['utc']).timestamp()*1e9
    parallel=set(start['parallel_stat_only']);skip={str(HERE),str(ROOT/'.git'),str(ROOT/'shared_rollout_db/journals')};changes=[];active=[];seen=0
    todo=[(str(ROOT),False),(str(ROOT.parent/'Basin_C1_flow_field_poc_20261004/poc'),False),(str(ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official'),False)]
    dbprefix=str(ROOT/'shared_rollout_db/rollout.sqlite');exceptions={str(ROOT/'docs'),str(ROOT/'diagnostics'),str(ROOT/'shared_rollout_db')};docprefix=str(ROOT/'docs/prompt_')
    while todo:
        folder,isparallel=todo.pop()
        try:entries=os.scandir(folder)
        except FileNotFoundError:continue
        with entries:
            for e in entries:
                p=e.path
                if p in skip or e.name=='.git' or p.startswith(dbprefix):continue
                try:s=e.stat(follow_symlinks=False)
                except FileNotFoundError:continue
                seen+=1;activebranch=isparallel or p in parallel
                if e.is_dir(follow_symlinks=False):todo.append((p,activebranch))
                if s.st_mtime_ns<cutoff or p in exceptions:continue
                if p.startswith(docprefix) and p not in start['protected']:continue
                (active if activebranch else changes).append(dict(path=p,size=s.st_size,mtime_ns=s.st_mtime_ns))
    out=HERE/'READONLY_BATCH_CHECKS'/f'{time.time_ns()}_{label}.json';out.parent.mkdir(exist_ok=True)
    write(out,dict(passed=not changes,metadata_entries_checked=seen,changes=changes,parallel_stat_changes=active))
    assert not changes,('READONLY_CHANGED',str(out),changes[:5]);print('Stat check PASSED',label,seen,flush=True)
if __name__=='__main__':check(sys.argv[1])

"""Resume baseline and stat-only batch guards; old incident records are immutable."""
from audit import HERE,ROOT,V1,V2,sha,write,git_snapshot
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,sys

ACCEPTED=V2/'runtime_cbs_query_memo_candidate_20261022'
SUPPLEMENT=ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1'
def metadata(root):
    result={}
    for folder,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in files:
            p=Path(folder)/name
            try:s=p.stat()
            except FileNotFoundError:continue
            result[str(p)]=[s.st_size,s.st_mtime_ns]
    return result

def opening():
    assert not (HERE/'INTEGRITY_RESUME_START_20261023.json').exists()
    old=json.loads((HERE/'INTEGRITY_START.json').read_text());cutoff=datetime.fromisoformat(old['utc']).timestamp()*1e9
    v1=metadata(V1);v2=metadata(V2)
    changed=[dict(path=p,stat=s) for p,s in v2.items() if s[1]>=cutoff and not Path(p).is_relative_to(ACCEPTED)]
    write(HERE/'RESUME_READONLY_VERIFICATION.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        accepted_exception=str(ACCEPTED),outside_accepted_cache_changed_since_original_opening=changed,
        v1_files=len(v1),v2_files=len(v2),passed=not changed))
    assert not changed,('v2 changed outside accepted cache',changed[:10])
    # The new accepted cache baseline detects ANY subsequent write, including
    # within the exception: acceptance applies only to the previous incident.
    write(HERE/'READONLY_RESUME_BASELINE.json',dict(v1=v1,v2=v2))
    protected={p:sha(p) for p in old['protected'] if Path(p).is_file()}
    for p in (ROOT/'docs/prompt_baseline_literature_scenes_resume_and_supplement_20261023.md',
              ROOT/'diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py'):
        protected[str(p)]=sha(p)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),protected=protected,
        readonly_baseline_sha256=sha(HERE/'READONLY_RESUME_BASELINE.json'),
        git={p:git_snapshot(p) for p in old['git']},
        old_failed_audit_sha256=sha(HERE/'INTEGRITY_END.json'),
        old_stop_sha256=sha(HERE/'READONLY_BOUNDARY_STOP.json'),
        accepted_exception='User accepts prior pure-query-cache writes only; no deletion, rollback or further writes',
        instruction_sha256=sha(ROOT/'docs/prompt_baseline_literature_scenes_resume_and_supplement_20261023.md'))
    write(HERE/'INTEGRITY_RESUME_START_20261023.json',result)
    print('Resume audit:',len(protected),'hashes;',len(v1)+len(v2),'read-only metadata records',flush=True)

def check(label):
    start=json.loads((HERE/'INTEGRITY_RESUME_START_20261023.json').read_text())
    cutoff=datetime.fromisoformat(start['utc']).timestamp()*1e9;changes=[];checked=0
    # No baseline JSON load is needed for a batch mtime guard. Direct scandir
    # metadata checks also catch directory deletion/creation. Final audit uses
    # the full opening inventory and content hashes independently.
    def visit(root):
        nonlocal checked
        todo=[str(root)]
        while todo:
            folder=todo.pop()
            try:entries=os.scandir(folder)
            except FileNotFoundError:
                changes.append(dict(path=folder,missing=True));continue
            with entries:
                for e in entries:
                    p=Path(e.path)
                    if p in (HERE,SUPPLEMENT,ROOT/'shared_rollout_db/journals') or p.is_relative_to(ROOT/'.git'):continue
                    if p.parent==ROOT/'shared_rollout_db' and p.name.startswith('rollout.sqlite'):continue
                    try:s=e.stat(follow_symlinks=False)
                    except FileNotFoundError:continue
                    checked+=1
                    if s.st_mtime_ns>=cutoff:
                        # New user prompt files are metadata-only exceptions.
                        if not (p.parent==ROOT/'docs' and p.name.startswith('prompt_') and str(p) not in start['protected']):
                            changes.append(dict(path=e.path,size=s.st_size,mtime_ns=s.st_mtime_ns))
                    if e.is_dir(follow_symlinks=False):todo.append(e.path)
    # Repository root includes v1, v2 and all other read-only diagnostics.
    # Parent directory mtimes change when this permitted work directory is
    # created; exclude the already-existing diagnostics container itself.
    for root in (V1,V2):visit(root)
    p=HERE/'READONLY_BATCH_CHECKS'/f'{time.time_ns()}_{label}.json'
    p.parent.mkdir(exist_ok=True)
    write(p,dict(utc=datetime.now(timezone.utc).isoformat(),passed=not changes,changes=changes,metadata_entries_checked=checked))
    assert not changes,('READONLY_BOUNDARY_CHANGED',str(p),changes[:5])
    print('Read-only guard passed:',label,flush=True)
    return p
if __name__=='__main__':opening() if sys.argv[1]=='start' else check(sys.argv[1])

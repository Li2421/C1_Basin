"""Opening/closing protected-source audit, with explicit stat-only pruning."""
from experiment import *
import subprocess
import time

FORBIDDEN = ('decentralized_', 'teacher_student_', 'multiscene_', 'baseline1_')
SKIP = {'.git','__pycache__','.pytest_cache','.mypy_cache','node_modules',
        '.venv','.venv-c1','.venv-c1-plots','.venv-ext','venv','env'}
V1_META_ONLY = {'cache','cuda_cache','tmp','mplconfig','logs','claims','groups'}
MAC = ROOT.parent / '01_MACFlow_Baseline_Reproduction/MACFlow_Official'
POC = ROOT.parent / 'Basin_C1_flow_field_poc_20261004/poc'

def stat(path):
    try:
        s = Path(path).stat()
        return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    except FileNotFoundError:
        return dict(missing=True)

def diagnostics_allowed(path, parallel):
    try: rel = Path(path).relative_to(ROOT/'diagnostics')
    except ValueError: return True
    if not rel.parts: return True
    name = rel.parts[0]
    if name == HERE.name: return False
    if str(ROOT/'diagnostics'/name) in parallel: return False
    if name.startswith(FORBIDDEN) and name != ROUND28.name: return False
    return True

def walk(root, parallel, v1_full=False):
    for folder, dirs, files in os.walk(root,followlinks=False):
        parent = Path(folder)
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.venv')
                         and diagnostics_allowed(parent/d,parallel))
        if not v1_full:
            dirs[:] = [d for d in dirs if d not in {'raw','traces','journals','datasets','archive','checkpoints'}]
        for name in sorted(files):
            p = parent/name
            if p.is_file() and not p.is_symlink(): yield p

def git_snapshot(root, parallel):
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    changed = git('diff','--name-only').decode().splitlines()
    restricted = [n for n in changed if root==ROOT and not diagnostics_allowed(ROOT/n,parallel)
                  and not (ROOT/n).is_relative_to(HERE)]
    args = ['diff','--','.']+[f':(exclude){n}' for n in restricted]
    digest = hashlib.sha256(git(*args)).hexdigest()
    return dict(head=git('rev-parse','HEAD').decode().strip(),
                status=git('status','--porcelain').decode(),
                diff_sha256=None if restricted else digest,
                protected_diff_sha256=digest,
                excluded_changed_paths_stat_only={n:stat(ROOT/n) for n in restricted})

def opening():
    assert not (HERE/'INTEGRITY_START.json').exists()
    assert not subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%j']).decode().strip()
    freeze = read(V1/'V1_FREEZE.receipt.json')
    for p,h in freeze['files'].items(): assert sha(p)==h,p
    cutoff = time.time()-86400
    parallel = {}
    for p in (ROOT/'diagnostics').iterdir():
        if p in (HERE,V1,ROUND28): continue
        if p.stat().st_mtime>cutoff:
            parallel[str(p)] = stat(p)
    restricted = {str(p):stat(p) for p in (ROOT/'diagnostics').iterdir()
                  if p.name.startswith(FORBIDDEN) and p != ROUND28}
    repos = {str(r):git_snapshot(r,parallel) for r in (ROOT,MAC)}
    readonly_dirs = {}
    for root in (V1,ROUND28,POC):
        for folder,dirs,_ in os.walk(root,followlinks=False):
            readonly_dirs[str(Path(folder))]=stat(folder)
            dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith('.venv')]
    manifest = HERE/'PROTECTED_INPUTS_START.jsonl'
    seen = set()
    counts = {}
    def entry(stream,p,kind,hash_content=True):
        p=p.resolve()
        if str(p) in seen:return
        assert diagnostics_allowed(p,parallel),p
        seen.add(str(p))
        r=dict(path=str(p),kind=kind,**stat(p))
        if hash_content:r['sha256']=sha(p)
        stream.write(canonical(r)+'\n')
        counts[kind]=counts.get(kind,0)+1
    with manifest.open('w') as stream:
        # Full frozen v1 evidence is covered before any v2 reuse. Runtime
        # caches/logs are stat-only, but are still checked for forbidden writes.
        for p in walk(V1,parallel,v1_full=True):
            rel=p.relative_to(V1)
            meta=any(part in V1_META_ONLY for part in rel.parts)
            entry(stream,p,'v1_runtime_metadata' if meta else 'v1_frozen_input',not meta)
        for p in walk(POC,parallel):
            if p.suffix=='.py':entry(stream,p,'poc_python')
        for p in walk(ROOT,parallel):
            if p.is_relative_to(V1) or p.is_relative_to(HERE):continue
            if p.suffix=='.py':entry(stream,p,'repository_python')
        for p in (ROOT/'docs').iterdir():
            if p.is_file():entry(stream,p,'existing_document')
        for p in (ROOT/'AGENTS.md',ROOT/'shared_rollout_db/README.md'):
            entry(stream,p,'required_instruction')
        for p in walk(ROUND28,parallel):
            if p.suffix in ('.py','.md','.json','.csv'):
                entry(stream,p,'round28_readonly_exception')
        for p in walk(MAC,parallel):
            if p.suffix in ('.py','.md','.json','.yaml','.yml','.csv','.so'):
                entry(stream,p,'official_macflow_input')
    write(HERE/'INTEGRITY_START.json',dict(
        utc=utc(),repositories=repos,manifest=str(manifest),manifest_sha256=sha(manifest),
        counts=counts,parallel_stat_only=parallel,forbidden_stat_only=restricted,
        readonly_directories=readonly_dirs,
        parallel_policy='Conservatively exclude all other diagnostics with top-level mtime in the last 24h, including possible interactive tasks; no active Slurm experiment remained at opening.',
        v1_frozen=True,v1_freeze_sha256=sha(V1/'V1_FREEZE.receipt.json'),
        required_inputs_full_file_hashes_at_opening=True,
        own_directory_excluded=str(HERE),resource_override='Latest user instruction: total 24 CPU shares and 5 GPU shards when no other use; count all user jobs before each submission.'))
    seal('INTEGRITY_START.receipt.json',[HERE/'INTEGRITY_START.json',manifest,HERE/'integrity.py'],opening_before_source_reuse=True)
    print('Opening audit complete',counts,flush=True)

def closing():
    assert not (HERE/'INTEGRITY_END.json').exists()
    start=read(HERE/'INTEGRITY_START.json')
    assert sha(start['manifest'])==start['manifest_sha256']
    differences=[];checked=0;opening_python=set()
    with Path(start['manifest']).open() as stream:
        for line in stream:
            r=json.loads(line);p=Path(r['path']);now=stat(p)
            if p.suffix=='.py':opening_python.add(str(p))
            if 'sha256' in r:
                actual=None if now.get('missing') else sha(p)
                if actual!=r['sha256']:differences.append(dict(path=str(p),kind='content',opening=r['sha256'],closing=actual))
            elif now!={k:r[k] for k in ('size','mtime_ns')}:
                differences.append(dict(path=str(p),kind='v1_metadata_write',opening=r,closing=now))
            checked+=1
    parallel=start['parallel_stat_only']
    for name,old in start['readonly_directories'].items():
        now=stat(name)
        if now!=old:differences.append(dict(kind='readonly_directory_metadata',path=name,opening=old,closing=now))
    for root in (ROOT,POC,MAC):
        for p in walk(root,parallel):
            if p.suffix=='.py' and str(p.resolve()) not in opening_python:
                differences.append(dict(kind='new_protected_python',path=str(p)))
    repos={str(r):git_snapshot(r,parallel) for r in (ROOT,MAC)}
    original_docs=set()
    with Path(start['manifest']).open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['kind']=='existing_document':original_docs.add(r['path'])
    new_prompts={}
    for p in (ROOT/'docs').iterdir():
        if str(p) in original_docs:continue
        if p.name.startswith('prompt_'):new_prompts[str(p)]=stat(p)
        else:differences.append(dict(kind='new_nonprompt_document',path=str(p),stat=stat(p)))
    def normalized_status(status,root):
        rows=[]
        for line in status.splitlines():
            p=root/line[3:].split(' -> ')[-1].strip('"').rstrip('/')
            if p.is_relative_to(HERE) or str(p) in new_prompts:continue
            if root==ROOT and not diagnostics_allowed(p,parallel):continue
            rows.append(line)
        return sorted(rows)
    for name,now in repos.items():
        old=start['repositories'][name];root=Path(name)
        if now['head']!=old['head']:differences.append(dict(kind='git_head',root=name))
        if normalized_status(now['status'],root)!=normalized_status(old['status'],root):differences.append(dict(kind='git_status',root=name))
        if now['protected_diff_sha256']!=old['protected_diff_sha256']:differences.append(dict(kind='git_diff',root=name))
    write(HERE/'INTEGRITY_END.json',dict(utc=utc(),passed=not differences,
        checked_files=checked,differences=differences,repositories=repos,
        parallel_stat_only={p:stat(p) for p in parallel},new_prompts_stat_only=new_prompts,
        v1_freeze_sha256=sha(V1/'V1_FREEZE.receipt.json')))
    seal('INTEGRITY_END.receipt.json',[HERE/'INTEGRITY_START.json',HERE/'PROTECTED_INPUTS_START.jsonl',HERE/'INTEGRITY_END.json',HERE/'integrity.py'],passed=not differences)
    assert not differences,differences[:10]

if __name__=='__main__':
    {'start':opening,'end':closing}[sys.argv[1]]()

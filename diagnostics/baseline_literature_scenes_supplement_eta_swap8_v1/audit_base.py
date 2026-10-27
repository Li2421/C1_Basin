"""Read-only opening/closing audit; no imports from historical entrypoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / 'diagnostics/baseline_literature_scenes_distributed_v2'
V1 = ROOT / 'diagnostics/baseline2_literature_scenes_v1'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def git_snapshot(repo):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def run(*args):return subprocess.check_output(['git','-C',str(repo),*args],env=env)
    status=run('status','--porcelain').decode()
    status='\n'.join(s for s in status.splitlines() if 'diagnostics/'+HERE.name not in s)
    return dict(head=run('rev-parse','HEAD').decode().strip(),status=status,
                status_sha256=hashlib.sha256(status.encode()).hexdigest(),
                diff_sha256=hashlib.sha256(run('diff','--no-ext-diff','--binary')).hexdigest())
def main(end=False):
    if end:
        start=json.loads((HERE/'INTEGRITY_START.json').read_text())
        hashes={p:sha(p) if Path(p).is_file() else None for p in start['protected']}
        git={p:git_snapshot(p) for p in start['git']}
        changes=[p for p,h in hashes.items() if h!=start['protected'][p]]
        result=dict(utc=datetime.now(timezone.utc).isoformat(),protected=hashes,git=git,
                    changed_files=changes,git_matches=git==start['git'])
        result['passed']=not changes and result['git_matches']
        write(HERE/'INTEGRITY_END.json',result)
        return result
    assert not (HERE/'INTEGRITY_START.json').exists()
    paths=set()
    # Source and documentation before first reuse. Large raw evidence gets a
    # separate pre-read inventory and sha256 journal, never copied or changed.
    for base in (V1,V2):
        paths.update(base.glob('*.py')); paths.update(base.glob('*.md'))
        for pat in ('*MANIFEST*.json','FLOW_STATES*.json','GEOMETRY_VALIDATION*.json',
                    '*QUALIFICATION.json','FIXED_ETA.json','*SPLIT*.json','V2_COMPLETE.json',
                    'PREREGISTRATION*.json','CORRECTION_REVISION*.json'):
            paths.update(base.glob(pat))
    for base in ('shared_control','shared_rollout_db','bottleneck_family','single_integrator','new_benchmark_common'):
        paths.update((ROOT/base).rglob('*.py'))
    paths.update((ROOT/'docs').glob('*'));paths.add(ROOT/'AGENTS.md')
    poc=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc')
    paths.update(poc.rglob('*.py'))
    # Determine active boundaries using metadata only, never opening other tasks.
    cutoff=__import__('time').time()-86400; parallel={}
    for p in (ROOT/'diagnostics').iterdir():
        if p in (HERE,V1,V2) or not p.is_dir():continue
        st=p.stat()
        if st.st_mtime>=cutoff or p.name.startswith(('decentralized_','teacher_student_','multiscene_','baseline1_')):
            parallel[str(p)]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),parallel_stat_only=parallel,
        protected={str(p.resolve()):sha(p) for p in sorted(paths) if p.is_file()},
        git={str(p):git_snapshot(p) for p in (ROOT,ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')},
        evidence_policy='Hash every raw/config before reading it; record in EVIDENCE_INPUTS.jsonl. Never use parallel task contents.')
    write(HERE/'INTEGRITY_START.json',result)
    print(json.dumps(dict(protected_files=len(result['protected']),parallel_dirs=len(parallel))))
if __name__=='__main__':main('--end' in sys.argv)

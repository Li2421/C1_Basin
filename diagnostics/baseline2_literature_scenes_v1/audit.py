"""Read-only protected-source audit; all output stays in this experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/zhihan/research/Basin_C1')
OUT = Path(__file__).resolve().parent
MAC = ROOT.parent / '01_MACFlow_Baseline_Reproduction/MACFlow_Official'
POC = ROOT.parent / 'Basin_C1_flow_field_poc_20261004'
FORBIDDEN = ('decentralized_', 'teacher_student_', 'multiscene_', 'baseline1_')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_snapshot(root):
    def run(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    return dict(head=run('rev-parse', 'HEAD').decode().strip(),
                status=run('status', '--porcelain').decode(),
                diff_sha256=hashlib.sha256(run('diff')).hexdigest())

def safe_walk(root):
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(FORBIDDEN) and d not in {'.git', '__pycache__', 'node_modules'} and not d.startswith('.venv')]
        for name in files:
            path = Path(base) / name
            if not path.is_symlink():
                yield path

def snapshot():
    now = time.time()
    parallel = {}
    for p in (ROOT / 'diagnostics').iterdir():
        if p == OUT:
            continue
        st = p.stat()
        if p.name.startswith(FORBIDDEN) or st.st_mtime >= now - 86400:
            parallel[str(p.relative_to(ROOT))] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    # Capture all permitted existing Python source once, before imports/copies.
    sources = {}
    for folder in ('bottleneck_family', 'shared_control', 'shared_rollout_db', 'new_benchmark_common', 'single_integrator'):
        for p in safe_walk(ROOT / folder):
            if OUT in p.parents or p.suffix != '.py':
                continue
            sources[str(p.relative_to(ROOT))] = sha(p)
    # Only Python files at diagnostic package roots and tools, not millions of
    # trajectory/cache descendants. This covers the explicitly requested reuse.
    for d in (ROOT / 'diagnostics').iterdir():
        if d == OUT or d.name.startswith(FORBIDDEN) or not d.is_dir():
            continue
        for p in list(d.glob('*.py')) + list((d / 'tools').glob('*.py')):
            if not p.is_symlink():
                sources[str(p.relative_to(ROOT))] = sha(p)
    start_path = OUT / 'INTEGRITY_START.json'
    old_docs = json.loads(start_path.read_text())['docs'] if start_path.exists() else None
    docs, new_prompts = {}, {}
    for p in (ROOT / 'docs').iterdir():
        rel = str(p.relative_to(ROOT))
        if old_docs is not None and rel not in old_docs and p.name.startswith('prompt_'):
            st = p.stat()
            new_prompts[rel] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
        elif p.is_file():
            docs[rel] = sha(p)
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                repositories={str(r): git_snapshot(r) for r in (ROOT, MAC)},
                poc_python={str(p): sha(p) for p in safe_walk(POC / 'poc') if p.suffix == '.py'},
                sources=sources,
                docs=docs, user_new_prompts_stat_only=new_prompts,
                parallel_stat_only=parallel)

if __name__ == '__main__':
    label = sys.argv[1]
    result = snapshot()
    (OUT / f'INTEGRITY_{label}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({'label': label, 'source_count': len(result['sources']), 'parallel_directories': list(result['parallel_stat_only']), 'poc_count': len(result['poc_python'])}, ensure_ascii=False))

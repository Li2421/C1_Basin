"""Read-only protected-source snapshot. Forbidden/active trees: stat only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/zhihan/research/Basin_C1')
HERE = Path(__file__).resolve().parent
POC = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
OFFICIAL = Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
PREFIXES = ('decentralized_', 'teacher_student_', 'multiscene_', 'crosstype2_')

REQUIRED_HISTORY = {'crosstype1_learned_zoo_selector_v1','orthoflow3_generator_critic_v1','orthoflow3_generator_critic_frozen_test_v1','orthoflow3_ring_k16_diagnostic_v1','orthoflow3_c1_generalization_v1','ring_controller_critic_v1','axis4a_geometry_horizon_v1','blackbox_families_v1'}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def stat_tree(path):
    rows = {}
    for directory, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
        for name in files:
            p = Path(directory) / name
            s = p.lstat()
            rows[str(p)] = [s.st_size, s.st_mtime_ns]
    return rows

def snapshot(known_parallel=()):
    now = time.time()
    parallel = {}
    forbidden = {}
    for p in sorted((ROOT / 'diagnostics').iterdir()):
        if not p.is_dir() or p == HERE:
            continue
        # Determine activity using metadata only, before any source read.
        rows = stat_tree(p)
        if p.name.startswith(PREFIXES):
            forbidden[str(p)] = rows
        if str(p) in known_parallel or (p.name not in REQUIRED_HISTORY and any(v[1] / 1e9 > now - 86400 for v in rows.values())):
            parallel[str(p)] = rows
    excluded = set(parallel) | set(forbidden)
    hashes = {}
    for root in (ROOT, POC, OFFICIAL):
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'node_modules')
                       and not d.startswith('.venv')
                       and str(Path(directory) / d) not in excluded
                       and Path(directory) / d != HERE]
            for name in files:
                p = Path(directory) / name
                if p.suffix == '.py' and not p.is_symlink():
                    hashes[str(p)] = digest(p.read_bytes())
    repos = {}
    for root in (ROOT, OFFICIAL):
        def git(*args):
            return subprocess.check_output(['git', '-C', str(root), *args])
        exclusions = [f':(exclude){Path(p).relative_to(ROOT)}' for p in sorted(excluded)] if root == ROOT else []
        if root == ROOT:
            exclusions.append(f':(exclude){HERE.relative_to(ROOT)}')
        diff = git('diff', 'HEAD', '--', '.', *exclusions)
        repos[str(root)] = dict(head=git('rev-parse', 'HEAD').decode().strip(),
                               status=git('status', '--porcelain').decode(),
                               diff_sha256=digest(diff), diff_scope='HEAD; active/forbidden trees excluded',
                               exclusions=exclusions)
    sq = subprocess.run(['squeue', '-o', '%.18i %.12u %.60j %.9P %.10T %.6C %b'], capture_output=True, text=True)
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                repositories=repos, protected_python_sha256=hashes,
                parallel_metadata=parallel, forbidden_metadata=forbidden,
                squeue=sq.stdout, squeue_returncode=sq.returncode,
                boundary_note='Forbidden prefixes pruned before content reads; named, completed historical sources are explicitly required read-only inputs. Other recently changed diagnostic trees conservatively metadata-only.')

if __name__ == '__main__':
    phase = sys.argv[1]
    target = HERE / f'INTEGRITY_{phase.upper()}.json'
    if target.exists():
        raise SystemExit(f'Refusing to replace {target}')
    # Keep opening parallel classifications even after their timestamps age
    # beyond 24 hours; do not start hashing a previously protected boundary.
    opening=HERE/'INTEGRITY_START.json'
    known_parallel=tuple(json.loads(opening.read_text())['parallel_metadata']) if phase.lower()!='start' and opening.exists() else ()
    result = snapshot(known_parallel)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(path=str(target), protected_files=len(result['protected_python_sha256']),
                          parallel=list(result['parallel_metadata']), forbidden=list(result['forbidden_metadata'])), indent=2))

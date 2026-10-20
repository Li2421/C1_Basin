"""Read-only source audit; never traverse the parallel round30 directories."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POC = ROOT.parent / 'Basin_C1_flow_field_poc_20261004'
MAC = ROOT.parent / '01_MACFlow_Baseline_Reproduction/MACFlow_Official'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True)

def sources(root):
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in
                         {'.git', '__pycache__', '.venv', 'node_modules'}
                         and not d.startswith('.venv')
                         and d != 'decentralized_blackbox_planners_v1'
                         and not d.startswith('teacher_student_macflow_v1')
                         and (Path(base) / d).resolve() != HERE)
        for name in sorted(files):
            path = Path(base) / name
            if name.endswith('.py') and not path.is_symlink():
                result[str(path)] = sha(path)
    return result

def capture():
    diff_args = ['git', 'diff', '--', '.',
                 ':(exclude)diagnostics/decentralized_blackbox_planners_v1/**',
                 ':(exclude)diagnostics/teacher_student_macflow_v1*/**']
    return dict(utc=datetime.now(timezone.utc).isoformat(),
                Basin=dict(status=command(['git','status','--porcelain']),
                           diff_sha256=hashlib.sha256(command(diff_args).encode()).hexdigest(),
                           diff_excludes='Forbidden parallel directories excluded before reading'),
                MAC=dict(head=command(['git','rev-parse','HEAD'], MAC).strip(),
                         status=command(['git','status','--porcelain'], MAC),
                         diff_sha256=hashlib.sha256(command(['git','diff'], MAC).encode()).hexdigest()),
                sources={**sources(ROOT), **sources(POC), **sources(MAC)},
                queue=command(['squeue','-u',str(os.getuid()),'-o','%.18i %.9P %.50j %.8T %.6C %.10b %.20R']))

if __name__ == '__main__':
    target = HERE / ('INTEGRITY_' + sys.argv[1] + '.json')
    assert not target.exists()
    value = capture()
    target.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(path=str(target), sources=len(value['sources']), utc=value['utc'])))

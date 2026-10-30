"""Install before importing research code; reject writes outside this experiment.

SQLite access to the official DB is permitted only for a coordinator. Workers
may write only the experiment's official journal and their experiment outputs.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
JOURNAL=ROOT/'shared_rollout_db/journals/exp_paper_baseline_protocol_v1_20261029'
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
for key,folder in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cuda_cache')]:
    path=HERE/folder;path.mkdir(exist_ok=True);os.environ[key]=str(path)

def allowed(path):
    if isinstance(path,int): return True
    p=Path(os.fsdecode(path)).resolve()
    return p.is_relative_to(HERE) or p.is_relative_to(JOURNAL) or p==Path('/dev/null')

def check(path,event):
    if path is not None and not allowed(path):
        with (HERE/'PROTECTION_VIOLATION.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event,path=str(path),pid=os.getpid()))+'\n')
        raise PermissionError(f'EXPERIMENT_HARD_STOP: {event}: {path}')

def audit(event,args):
    if event=='open':
        path,mode,flags=args
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND): check(path,event)
    elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.chown','os.utime','os.truncate'):
        # Official writer creates journals parents with exist_ok=True.
        if event=='os.mkdir' and Path(os.fsdecode(args[0])).resolve() in (JOURNAL.parent,JOURNAL.parent.parent) and Path(args[0]).is_dir(): return
        check(args[0],event)
    elif event in ('os.rename','os.link','os.symlink'):
        check(args[0],event);check(args[1],event)
    elif event=='sqlite3.connect':
        value=str(args[0]);official=str(ROOT/'shared_rollout_db/rollout.sqlite')
        readonly=value.startswith('file:'+official) and 'mode=ro' in value
        if not readonly and not (os.environ.get('STB_COORDINATOR')=='1' and value==official): check(value,event)

_base_audit = audit
_parallel = [Path(x['path']).resolve() for x in json.loads((HERE/'PARALLEL_TASKS.json').read_text())['parallel_directories']]
def audit(event,args):
    if event == 'open' and not isinstance(args[0],int):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(p) for p in _parallel):
            with (HERE/'PROTECTION_VIOLATION.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='parallel_content_access',path=str(path),pid=os.getpid()))+'\n')
            raise PermissionError('EXPERIMENT_HARD_STOP: parallel-task contents are metadata-only: '+str(path))
    _base_audit(event,args)
sys.addaudithook(audit)

def save_imports():
    paths=set()
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value and Path(value).is_file(): paths.add(str(Path(value).resolve()))
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}
    folder=HERE/'import_manifests';folder.mkdir(exist_ok=True)
    (folder/f'{os.getpid()}.json').write_text(json.dumps(hashes,indent=2)+'\n')

atexit.register(save_imports)

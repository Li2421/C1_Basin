"""Record local evidence locations and software versions without copying large data."""
import hashlib,json,platform,sys
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
OUT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    packages={}
    for name in ('numpy','scipy','jax','jaxlib','clarabel'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]='not installed in this interpreter'
    files={}
    for name in ('candidates.jsonl','legacy_candidates.jsonl','legacy_prior_candidates.jsonl','candidates_verified.jsonl'):
        p=OUT/name;files[name]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    traces={}
    for name in ('traces','legacy_traces'):
        pp=list((OUT/name).glob('*.npz'))
        traces[name]=dict(path=str(OUT/name),count=len(pp),bytes=sum(p.stat().st_size for p in pp),sha256='not computed for all NPZ; exact UID filenames and presence checked by verify.py')
    x=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
           raw_files=files,trace_directories=traces,cpu_limit=18,gpu_used=0,
           backup='Local evidence only; no external data backup performed.')
    (OUT/'artifact_manifest.json').write_text(json.dumps(x,indent=2)+'\n')
    print('Recorded evidence manifest')

if __name__=='__main__':main()

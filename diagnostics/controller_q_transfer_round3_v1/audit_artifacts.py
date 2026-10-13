"""Final read-only content inventory; no historical assets are modified."""
import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    names=['REPORT.md','HANDOFF.md','results.json','ENVIRONMENT.json','RUN_COMMANDS.md',
           'P0_RESULTS.json','P1_RESULTS.json','P1_POSTFLIGHT.json','P1_Q_DECOMPOSITION.json',
           'P2PRIME_PROTOCOL.json','P2PRIME_CASES.json','P2PRIME_CACHE_PREFLIGHT.json',
           'P2PRIME_POSTFLIGHT.json','P2PRIME_LABELS.npz','P2PRIME_RESULTS.json',
           'P2PRIME_SOURCE_COUNT_ANALYSIS.json','P2PRIME_R3_ALIASING_DIAGNOSTIC.json',
           'P2PRIME_RESULTS_UNCOUPLED_UNKNOWN.json','P3_RESULTS.json',
           'P2PRIME_MERGE_REGISTRATION_FIX.md','P2PRIME_PAIRWISE_UNKNOWN_FIX.md']
    names += [p.name for p in HERE.glob('PREREGISTRATION*.md')]
    names += [p.name for p in HERE.glob('*RECEIPT*.json')]
    keyfiles={name:dict(sha256=digest(HERE/name),bytes=(HERE/name).stat().st_size) for name in sorted(set(names)) if (HERE/name).is_file()}
    folders={}
    for name in ('p1_raw','p2prime/raw','p0_fits','p1_features','p2prime_features','p2prime_fits','p3_features'):
        d=HERE/name
        if d.exists():
            files=[p for p in d.rglob('*') if p.is_file()]
            folders[name]=dict(files=len(files),bytes=sum(p.stat().st_size for p in files))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),keyfiles=keyfiles,folders=folders,
                note='Key-file SHA256 and folder size/count inventory; raw individual provenance is in exact UID cases, per-record journal and shared rollout DB source_file.')
    (HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:v['files'] for k,v in folders.items()})


if __name__=='__main__':main()

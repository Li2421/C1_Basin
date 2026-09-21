"""Execute a frozen list of commands with separate logs; never overwrite a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    args=p.parse_args()
    raw=args.spec.read_bytes(); spec=json.loads(raw)
    fingerprint=hashlib.sha256(raw).hexdigest()
    folder=args.spec.parent/args.spec.stem
    folder.mkdir(exist_ok=True)
    for i,job in enumerate(spec['jobs']):
        marker=folder/f'{i:02d}.complete.json'
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['spec_sha256']!=fingerprint or old['argv']!=job:
                raise ValueError('Completed queue job changed')
            continue
        with (folder/f'{i:02d}.log').open('x') as log:
            print('Start',i,job,flush=True)
            result=subprocess.run(job,cwd=spec['cwd'],
                env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (folder/f'{i:02d}.failed.json').write_text(json.dumps(dict(argv=job,returncode=result.returncode)))
            raise SystemExit(result.returncode)
        marker.write_text(json.dumps(dict(argv=job,spec_sha256=fingerprint),indent=2))
        print('Complete',i,flush=True)
    (folder/'complete.json').write_text(json.dumps(dict(jobs=len(spec['jobs']),spec_sha256=fingerprint)))


if __name__=='__main__':main()

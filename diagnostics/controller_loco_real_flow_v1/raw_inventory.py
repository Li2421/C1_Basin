"""Content hashes for every raw episode, without inspecting outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
BASE=Path(__file__).resolve().parent

def row(path):
    p=Path(path);blob=p.read_bytes()
    return dict(path=str(p.relative_to(BASE)),rollout_uid=p.stem,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())

def main(stage):
    folders=[BASE/'raw',BASE/'stage_b/raw'] if stage=='ab' else [BASE/'stage_c/raw']
    paths=[str(p) for folder in folders for p in sorted(folder.glob('*.json'))]
    expected=114688 if stage=='ab' else 8192
    assert len(paths)==expected
    dest=BASE/f'RAW_CONTENT_INDEX_STAGE_{stage.upper()}.jsonl';assert not dest.exists()
    digest=hashlib.sha256()
    with ProcessPoolExecutor(max_workers=5) as pool, dest.open('w') as handle:
        for r in pool.map(row,paths,chunksize=128):
            line=json.dumps(r,separators=(',',':'))+'\n';handle.write(line);digest.update(line.encode())
    receipt=dict(status='COMPLETE',stage=stage,episodes=expected,index_sha256=digest.hexdigest(),outcomes_read=False)
    dest.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['ab','c']);main(ap.parse_args().stage)

"""Final source diffs before B training; originals remain read-only."""
from common import *
import difflib

def main():
    rows=[]
    for r in read(HERE/'B_CRITIC_COPY_SOURCES.json'):
        source=Path(r['source']);copy=Path(r['copy']);old=source.read_text();new=copy.read_text();assert sha(source)==r['source_sha256'];rows.append(dict(source=str(source),source_sha256=sha(source),copy=str(copy),copy_sha256=sha(copy),final_diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))))
    write(HERE/'B_IMPLEMENTATION_FINAL.json',dict(utc=utc(),files=rows,architecture_only_change='action input dimension10to13; cohort/source-set sizes and candidate masks adapted; source-only selection/hyperparameters unchanged'))
    seal('B_IMPLEMENTATION_FINAL.receipt.json',[HERE/'B_IMPLEMENTATION_FINAL.json',HERE/'implementation_manifest.py',*[Path(r['copy']) for r in rows]],before_real_critic_training=True)
if __name__=='__main__':main()

"""Hash local outputs and identify immutable reused sources."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
TRANSFER=HARD/'transfer_mechanism_v2'
PREV=ROOT/'diagnostics/part2_closed_loop_sensitivity_validation_20261012'

def info(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    source=[HARD/'confirm_q16_tasks.json',HARD/'confirmation_manifest.json',HARD/'screening_manifest.json',
            TRANSFER/'state_eta_chain_features.csv',TRANSFER/'final_report.txt',
            PREV/'b15_by_candidate.csv',PREV/'gain_loss_audit.csv',PREV/'final_report.md']
    out=dict(source_files={p.name:info(p) for p in source},result_roots=[str(HARD/'results'),str(TRANSFER/'results')],
             local_files={})
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='cache_index.json' and '__pycache__' not in p.parts:
            out['local_files'][str(p.relative_to(HERE))]=info(p)
    (HERE/'cache_index.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('indexed',len(out['local_files']),'local files')

if __name__=='__main__':main()

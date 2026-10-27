"""Evidence-only diagnostic backfill, 5000 paired statistics, DB and source audits."""
from pathlib import Path
import os,sys,json,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def call(script,*args):
    p=HERE/'finish_actions'/f'{time.time_ns()}_{script}.log';p.parent.mkdir(exist_ok=True)
    with p.open('w') as log:r=subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((script,r.returncode,str(p),p.read_text()[-2500:]))
def main():
    assert (HERE/'COMPUTATION_READY.json').exists()
    call('backfill_diagnostics.py');call('guard_fast.py','before_statistics')
    # Sequential CPU evidence-only jobs stay within one management CPU share.
    call('statistics_supplement_user_fix_20261023.py','eta');call('statistics_supplement_user_fix_20261023.py','swap')
    call('guard_fast.py','after_statistics');call('postflight.py');call('finalize_supplement_user_fix_20261023.py')
if __name__=='__main__':main()

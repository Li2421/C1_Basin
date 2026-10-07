"""Completion-driven orchestration. sbatch --wait; no scheduler/log polling."""
from __future__ import annotations
import argparse,subprocess,sys,time
from .design import ROOT,MAIN,read,write,sha
from . import cache


def collect(stage):
    if stage=='test':assert (ROOT/'models_frozen.json').exists()
    batches=read(ROOT/('preflight_'+stage)/'batches.json')
    for b in batches:
        name=b['batch'];dest=ROOT/'batches'/name
        if (dest/'postflight_audit.json').exists():
            assert read(dest/'postflight_audit.json')['all_attempted_records_in_global_db'];continue
        write(ROOT/'working_state.json',dict(phase='collecting_'+stage,batch=name,new_rollout_upper_bound=30208,
            protocol_sha256=sha(ROOT/'protocol.json'),wait_mechanism='sbatch --wait, no polling',test_opened=stage=='test'))
        print({'event':'submit','batch':name,'preflight':b},flush=True)
        with (dest/'slurm_submit.txt').open('w') as out:
            r=subprocess.run(['sbatch','--wait',str(ROOT/'rollout.sbatch'),name],cwd=MAIN,stdout=out,stderr=subprocess.STDOUT)
        cache.merge()
        if r.returncode:
            write(dest/'failed.json',dict(exit_code=r.returncode,time=time.time()));raise RuntimeError(f'Batch {name} failed; results journaled, inspect only completed failure')
        for w in range(5):assert read(dest/f'worker{w}.json')['completed']
        cache.postflight(name)
        print({'event':'completed','batch':name},flush=True)
    write(ROOT/'working_state.json',dict(phase=stage+'_collection_complete',protocol_sha256=sha(ROOT/'protocol.json'),test_opened=stage=='test'))
    print({'event':stage+'_collection_complete'},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=('source','test'));collect(a.parse_args().stage)

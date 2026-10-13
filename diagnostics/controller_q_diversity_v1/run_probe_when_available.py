"""One-CPU source-only probe; wait without a reservation and respect cap12."""
import subprocess
import time
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def call(args):
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    assert r.returncode==0 or (args[0]=='scontrol' and 'Job has already finished' in r.stderr),r.stderr
    return r.stdout


def main():
    array='11021'
    call(['scontrol','update',f'JobId={array}','ArrayTaskThrottle=11'])
    try:
        while True:
            running=call(['squeue','-h','-t','R','-o','%j|%C'])
            used=sum(int(v.split('|')[1]) for v in running.splitlines() if v.startswith('controller_q_'))
            if used<=11:break
            time.sleep(5)
        module=sys.argv[1] if len(sys.argv)>1 else 'source_probe'
        assert module in ('source_probe','check_evaluation','source_review','contact_probe','contact_full')
        cmd='cd '+str(ROOT)+' && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -u -m diagnostics.controller_q_diversity_v1.'+module
        job=call(['sbatch','--parsable','--job-name=controller_q_probe','--partition=cpu','--cpus-per-task=1','--mem=4G',
            '--time=00:25:00',f'--output={HERE}/logs/probe_%j.log','--wrap',cmd]).strip().split(';')[0]
        print('Source-only probe submitted',job,flush=True)
        while call(['squeue','-h','-j',job]).strip():time.sleep(10)
        dest={'source_probe':'development_probe64_results.json','check_evaluation':'evaluation_integration_check.json','source_review':'source_review.json','contact_probe':'contact_probe_results.json','contact_full':'contact_features_complete.json'}[module]
        assert (HERE/dest).exists(),job
        print('Source-only probe finished',flush=True)
    finally:
        if call(['squeue','-h','-j',array]).strip():call(['scontrol','update',f'JobId={array}','ArrayTaskThrottle=12'])


if __name__=='__main__':main()

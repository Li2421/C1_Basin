from core import *
import time,subprocess

def main():
    while not all((HERE/'pilot'/f'worker_{i}.json').exists() for i in range(18)):
        live=subprocess.check_output(['squeue','-h','-j','21368','-o','%T'],text=True).strip()
        if not live:raise RuntimeError('Pilot job ended before all worker receipts; inspect logs; do not rerun unknown')
        time.sleep(20)
    from postflight import merge
    merge('pilot');gate=read(HERE/'CAPABILITY_GATE.json');print(json.dumps(gate),flush=True)
    if not gate['eligible']:
        write(HERE/'NO_ELIGIBLE_CONTROLLERS.json',dict(utc=utc(),status='INCONCLUSIVE',reason='all students excluded by preregistered capability gate'));return
    from probes import design
    design();a=subprocess.check_output(['sbatch','--array=0-16%17',str(HERE/'probe.sbatch'),'17'],text=True);b=subprocess.check_output(['sbatch',str(HERE/'benchmark.sbatch')],text=True);write(HERE/'PROBE_SUBMISSION.json',dict(utc=utc(),probes=a,benchmark=b,CPU_shards=18,GPU_shards=0,preflight_sha256=sha(HERE/'probe_PREFLIGHT.receipt.json')));print(a+b,flush=True)
if __name__=='__main__':main()

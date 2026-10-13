import os,subprocess,sys,json
from pathlib import Path
here=Path(__file__).resolve().parent
if len(sys.argv)==1:
    cpus=[x for x in sorted(os.sched_getaffinity(0)) if x>=8];assert len(cpus)>=2,cpus
    jobs=[subprocess.Popen([sys.executable,'-m','diagnostics.controller_native_exposure_v1.replay_ecores',str(index),str(cpu)]) for index,cpu in zip([0,1],cpus)]
    assert all(p.wait()==0 for p in jobs)
else:
    index,cpu=map(int,sys.argv[1:]);os.sched_setaffinity(0,{cpu})
    from .study import fit,JOBS,write
    write(here/f'replay_cpu_seed{JOBS[index]["seed"]}.json',dict(affinity=list(os.sched_getaffinity(0)),cpu_type='E',reason='Reconstruct original Stage17 checkpoints with same CPU arithmetic; no new scientific intervention'))
    fit(index)

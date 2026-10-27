"""Runtime-only forecast; evaluation outcomes never enter action selection."""
from pathlib import Path
import sys,statistics,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from experiment import *
from audit_supplement import evidence
from collections import defaultdict
import numpy as np

def main():
    ix=read(HERE/'HISTORY_EXECUTION_INDEX.json');selection=read(HERE/'HISTORY_SELECTION.json');eta=read(HERE/'FIXED_ETA.json')['13']['values'];hx=np.asarray(eta,np.float64).tobytes().hex();timings=defaultdict(list)
    for key,entry in ix.items():
        scene,ctl,n,index,seed,schedule=json.loads(key)
        if seed!=0 or schedule!=[hx,hx,None]:continue
        allowed={(s['N'],s['index']) for s in selection[scene]['evaluation_states']}
        if (n,index) not in allowed:continue
        p=Path(entry['path']);evidence(p);r=read(p);timings[scene,ctl].append(r['elapsed_seconds'])
    out={scene:dict(existing_eta13_eval_cpu_hours=sum(sum(v) for (s,c),v in timings.items() if s==scene)/3600,controllers=sum(s==scene for s,c in timings),episodes=sum(len(v) for (s,c),v in timings.items() if s==scene)) for scene in selection}
    write(HERE/'RUNTIME_FORECAST_INPUTS_20261023.json',dict(utc=utc(),timings=out,uses_outcomes=False,changes_selection=False,note='Historical timing only. New winning schedules, native query cache, harder controller variants and N8 runtime make total estimate uncertain.'))
    print(out,flush=True)
if __name__=='__main__':main()

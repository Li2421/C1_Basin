import bootstrap
from experiment import *
from evidence_store import load
from swap_diagnostics import summary
import numpy as np
if __name__=='__main__':
    done=0;missing=0
    for c in read(HERE/'swap_main_CASES.json'):
        p=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json"
        if p.exists():done+=1;continue
        r=load('swap_main',c);assert sha(r['trace_path'])==r['trace_sha256']
        with np.load(r['trace_path']) as z:write(p,summary(r,z))
        missing+=1
    write(HERE/'DIAGNOSTIC_COMPLETENESS.json',dict(passed=True,already_present=done,backfilled_from_saved_arrays=missing,new_rollouts=0))

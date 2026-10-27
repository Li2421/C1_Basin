from pathlib import Path
import resume_bootstrap
from experiment import *
from stage2_diagnostics_resume import summary
import numpy as np

def main():
    count=0
    for selection in read(HERE/'PART1_STAGE_SELECTIONS.json').values():
        stage=selection['stage']
        if not stage:continue
        for c in read(HERE/f'{stage}_CASES.json'):
            p=HERE/stage/'raw'/f"{c['rollout_uid']}.json";assert p.exists(),p
            target=HERE/'DIAGNOSTICS'/p.name
            if target.exists():continue
            raw=read(p);assert sha(raw['trace_path'])==raw['trace_sha256']
            with np.load(raw['trace_path']) as z:diag=summary(raw,z)
            write(target,diag);count+=1
    write(HERE/'DIAGNOSIS_COMPLETENESS_RESUME.json',dict(utc=utc(),passed=True,backfilled_from_saved_arrays=count,new_rollouts=0))
    print('Diagnosis completeness passed;',count,'saved-trace backfills',flush=True)
if __name__=='__main__':main()

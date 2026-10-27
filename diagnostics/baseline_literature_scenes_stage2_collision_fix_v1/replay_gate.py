"""Sixteen historical executions, engineering replay only, no new labels."""
from experiment import *
from dynamics_spatial_mask_20261022 import run
from audit import V2
import numpy as np

def main():
    chosen=[]
    for scene in NAMES:
        folder=V2/f'optional_dynamics_{scene}'/'raw'
        # First sorted seed0 safety record for four distinct actor types.
        types=set()
        for p in sorted(folder.glob('*.json')):
            r=read(p);c=r['case'];kind=c['controller_payload']['nominal']['type_code']
            if c['seed']!=0 or c['method']!='safety' or kind in types:continue
            types.add(kind);chosen.append((p,r))
            if len(types)==4:break
    result=[]
    for p,raw in chosen:
        trace=Path(raw['trace_path']);assert sha(trace)==raw['trace_sha256']
        with (HERE/'EVIDENCE_INPUTS.jsonl').open('a') as f:f.write(canonical(dict(path=str(trace),sha256=sha(trace)))+'\n')
        r,z=run(raw['case'])
        with np.load(trace) as old:
            keys=['positions','velocities','actions','nominal','safe','projection_rewrite','swept_clearance']
            matches={k:bool(np.array_equal(old[k],z[k])) for k in keys}
        row=dict(source=str(p),source_sha256=sha(p),trace_sha256=sha(trace),matches=matches,
                 terminal_matches=all(r[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length')),
                 old_outcome=raw['outcome'],new_outcome=r['outcome'],new_error=r.get('error'))
        row['passed']=all(matches.values()) and row['terminal_matches'];result.append(row)
        write(HERE/'REPLAY_GATE.json',dict(results=result,complete=len(result)==16,passed=len(result)==16 and all(x['passed'] for x in result)))
        print(len(result),raw['case']['state']['scene'],raw['case']['controller_payload']['nominal']['type_code'],row['passed'],flush=True)
if __name__=='__main__':main()

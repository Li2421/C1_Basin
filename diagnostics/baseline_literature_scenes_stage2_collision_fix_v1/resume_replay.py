"""Thirty-two saved trajectories, private rebuilt backend, no new research labels."""
import resume_bootstrap
from experiment import *
from audit import V2
import numpy as np

def prepare():
    chosen=[]
    for scene in NAMES:
        seen=set()
        for path in sorted((V2/f'optional_dynamics_{scene}'/'raw').glob('*.json')):
            raw=read(path);c=raw['case'];kind=c['controller_payload']['nominal']['type_code']
            if kind in seen or c['seed']!=0 or c['method']!='safety' or raw['numerical_failure']:continue
            seen.add(kind);chosen.append(dict(path=str(path),sha256=sha(path),scene=scene,type=kind))
            if len(seen)==8:break
        assert len(seen)==8,(scene,len(seen))
    write(HERE/'RESUME_REPLAY_CASES.json',chosen)

def run(index):
    from dynamics_spatial_mask_20261022 import run as engine
    source=read(HERE/'RESUME_REPLAY_CASES.json')[index];p=Path(source['path']);assert sha(p)==source['sha256']
    raw=read(p);trace=Path(raw['trace_path']);assert sha(trace)==raw['trace_sha256']
    result,z=engine(raw['case'])
    with np.load(trace) as old:
        matches={k:bool(np.array_equal(old[k],z[k])) for k in ('positions','velocities','actions','nominal','safe','projection_rewrite','swept_clearance')}
    terminal=all(result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length'))
    write(HERE/'RESUME_REPLAYS'/f'{index:02d}.json',dict(source=source,trace_sha256=sha(trace),matches=matches,terminal_matches=terminal,passed=all(matches.values()) and terminal))
    print(index,source['scene'],source['type'],all(matches.values()) and terminal,flush=True)
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run(int(sys.argv[1]))

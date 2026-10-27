import bootstrap
from experiment import *
from audit_supplement import V2,evidence,check
import numpy as np

def prepare():
    chosen=[]
    for scene in NAMES:
        seen=set()
        for path in sorted((V2/f'distributed_baseline_{scene}'/'raw').glob('*.json')):
            raw=read(path);c=raw['case'];kind=c['controller_payload']['nominal']['type_code']
            if kind in seen or c['seed'] or raw['numerical_failure']:continue
            seen.add(kind);chosen.append(dict(path=str(path),sha256=evidence(path),scene=scene,type=kind))
            if len(seen)==8:break
        assert len(seen)==8,(scene,len(seen))
    write(HERE/'REPLAY_CASES.json',chosen)

def run(index):
    from runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022 import run as engine
    source=read(HERE/'REPLAY_CASES.json')[index];p=Path(source['path']);assert sha(p)==source['sha256']
    raw=read(p);trace=Path(raw['trace_path']);assert evidence(trace)==raw['trace_sha256']
    result,z=engine(raw['case'])
    with np.load(trace) as old:
        matches={k:bool(np.array_equal(old[k],z[k])) for k in ('positions','velocities','actions','nominal','safe','projection_rewrite','swept_clearance')}
    terminal=all(result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length'))
    proof=dict(source=source,matches=matches,terminal_matches=terminal,passed=all(matches.values()) and terminal)
    write(HERE/'REPLAYS'/f'{index:02d}.json',proof);assert proof['passed'],proof
    print(index,source['scene'],source['type'],'PASSED',flush=True)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        shard=int(os.environ.get('SLURM_ARRAY_TASK_ID',0));size=int(sys.argv[1])
        for i in range(shard,32,size):
            proof=HERE/'REPLAYS'/f'{i:02d}.json'
            if proof.exists():assert read(proof)['passed'];continue
            run(i)

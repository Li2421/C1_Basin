"""Saved-free full replays and static nominal geometry checks, no new labels."""
import bootstrap
from experiment import *
from swap_geometry import SwapEnv as OriginalEnv,moving_path as original_path
from swap_geometry_user_fix_20261023 import SwapEnvRevision,moving_path,static_walls,config
from swap_runtime import run
from controllers_layout_user_fix_20261023 import wall_penalties
from types import SimpleNamespace
import numpy as np

def main():
    assert SwapEnvRevision.step is OriginalEnv.step and moving_path is original_path
    selected={}
    for p in (HERE/'capability/raw').glob('*.json'):
        r=read(p);c=r['case'];kind=c['controller_payload']['nominal']['type_code']
        if c['state']['scene']!='lit_swap8_free' or c['seed'] or r['numerical_failure'] or kind in selected:continue
        h=sha(p);assert sha(r['trace_path'])==r['trace_sha256'];result,z=run(c)
        with np.load(r['trace_path']) as old:matches={k:bool(np.array_equal(z[k],old[k])) for k in ('positions','velocities','nominal','safe','actions','projection_rewrite','swept_clearance')}
        assert all(matches.values()) and all(result[k]==r[k] for k in ('success','collision','timeout','outcome','episode_length')),kind
        selected[kind]=dict(source=str(p),source_sha256=h,trace_sha256=r['trace_sha256'],matches=matches);print('Unchanged free replay',kind,'PASSED',flush=True)
        if len(selected)==8:break
    assert len(selected)==8
    walls=static_walls('lit_swap8_obstacles');env=SimpleNamespace(config=config('lit_swap8_obstacles'));mid=walls.mean(1)
    inside,out=wall_penalties(mid,env);assert np.all(inside>0) and np.all(out==0)
    write(HERE/'USER_FIX_UNCHANGED_FREE_AND_GEOMETRY.receipt.json',dict(utc=utc(),passed=True,free_full_bitwise_replays=selected,original_step_inherited=True,original_mover_path_inherited=True,new_walls_visible_in_nominal_cost=True,new_rollouts=0))
if __name__=='__main__':main()

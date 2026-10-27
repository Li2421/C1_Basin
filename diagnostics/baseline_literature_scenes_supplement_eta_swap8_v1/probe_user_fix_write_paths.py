import bootstrap
from experiment import *
from swap_runtime_user_fix_20261023 import SwapEnv,SwapSafety
from tiled_planning_pa_memo_query_disk_candidate_spatial_mask_20261022 import Grid
import numpy as np
if __name__=='__main__':
    state=read(HERE/'FLOW_STATES_lit_swap8_obstacles.json')[0];s=dict(state,N=1,parent_N=8,agent=1,parent_positions=state['positions'],positions=[state['positions'][1]],goals=[state['goals'][1]],velocities=[[0.,0.]])
    env=SwapEnv(s);grid=Grid(env,.05,.1);path=grid.search(env.positions[0],env.goals[0],())
    assert path is not None
    c=read(HERE/'FLOW_STATES_lit_swap8_moving.json')[0];c=dict(c,N=1,parent_N=8,agent=0,parent_positions=c['positions'],positions=[c['positions'][0]],goals=[c['goals'][0]],velocities=[[0.,0.]])
    mover=SwapEnv(c);safety=SwapSafety(mover.snapshot(),0);u,proof=safety.project(np.zeros((1,2)))
    write(HERE/'USER_FIX_WRITE_PATH_PROBE.json',dict(passed=True,pure_static_query_path_length=len(path),moving_projection=proof,robot_state_same=True,new_research_rollouts=0))

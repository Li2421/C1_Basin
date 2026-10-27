import bootstrap
from experiment import *
from audit_supplement import H
if __name__=='__main__':
    proof=read(HERE/'IMPLEMENTATION_FROZEN.receipt.json');changes=[p for p,h in proof['files'].items() if sha(p)!=h];assert not changes,changes
    p=HERE/'USER_FIX_START_20261023.json';assert not p.exists()
    write(p,dict(utc=utc(),protected_original_implementation=proof['files'],original_implementation_sha256=sha(HERE/'IMPLEMENTATION_FROZEN.receipt.json'),part1_complete_sha256=sha(H/'COMPLETE.json'),original_qualification_sha256=sha(HERE/'QUALIFICATION.json'),unchanged_free_states_sha256=sha(HERE/'FLOW_STATES_lit_swap8_free.json'),unchanged_static_robot_states_sha256=sha(HERE/'FLOW_STATES_lit_swap8_obstacles.json'),unchanged_moving_robot_states_sha256=sha(HERE/'FLOW_STATES_lit_swap8_moving.json'),active_old_stage='old_eval',authorization='User correction: only moving-obstacle constraint and static obstacle layout; rerun these two versions under new identities; original results invalidated, preserve unchanged old-scene/free execution; success/collision/95% gate unchanged',original_results_must_be_retained=True,reason='用户指出说明设计错误后的修正，不是看结果调参'))
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+' 用户修正：这是用户指出说明设计错误后的修正，不是看结果调参。仅改变moving障碍约束（速度上界最坏情况半空间，去除一步半径膨胀，ε=.01配对d_safe）和static L布局（指定半径范围、最大臂长、名义走廊与既定扰动state直线可达）。旧结果保留标“原约束/原布局（作废）”，不进主分析；旧场景粗筛和free语义/数据不变、运行不中断；碰撞判据、成功标准、95%能力门都不变。\n')
    write(HERE/'ORIGINAL_SWAP8_RESULTS_INVALIDATED_20261023.json',dict(utc=utc(),status='原约束/原布局（作废）',scenes=['lit_swap8_moving','lit_swap8_obstacles'],affected_original_stages=['capability'],original_case_manifest_sha256=sha(HERE/'capability_CASES.json'),original_qualification_sha256=sha(HERE/'QUALIFICATION.json'),excluded_from_primary_analysis=True,free_scene_status='Unchanged and retained',action='New cases/rollout UIDs, preserve original artifacts and DB rows'))
    print('User fix opening and original-result invalidation recorded')

"""Source-defined physical mapping; no outcome-dependent parameter choices."""
import guard
from guard import HERE
import json
import datetime


def apply():
    path = HERE/'paper_methods/mgr_author_unicycle_tracking/adapter.py'
    source = path.read_text()
    hook = '''
# Physical-environment mapping only; all author algorithm buffers remain fixed.
_unaligned_make = make
def make(env,seed,method,options,context,mechanism=True,root_only=False):
    result=_unaligned_make(env,seed,method,options,context,mechanism,root_only)
    from margin_runtime import align_mgr
    align_mgr(env)
    return result
'''
    if hook not in source:
        path.write_text(source+hook)
    release = HERE/'freeze_execution.py'
    source = release.read_text()
    if "'margin_runtime.py'" not in source:
        release.write_text(source.replace("'core.py','physical_geometry.py'", "'core.py','margin_runtime.py','physical_geometry.py'"))
    changes = [dict(method='mgr_author_unicycle_tracking', parameter='SAFE_RATIO', original=1.1,
                    aligned=1., category='physical safety distance',
                    source='vendor/mgr/params.py:4; cbf.py:33-34; paper arXiv2503.05848 p5 V-A: physical radius0.2, safety radius0.22.',
                    meaning='Original robot-to-robot minimum2*r*SAFE_RATIO; obstacle radius already includes common wall radius and margin. Common pair margin0 gives2*r; common obstacle threshold r+wall_radius+wall_collision_margin.',
                    algorithm_constants_unchanged=['C_R','K_P','tau','K','exit-angle rule','placement-map buffers'])]
    value = dict(time=datetime.datetime.now().astimezone().isoformat(), changes=changes,
                 unchanged_dac='Physical robot radius and capsule wall width already map to common config. cbf_boundary_hold is an active-set threshold, not extra geometric radius, and is retained.',
                 no_success_labels_used_to_select_mapping=True)
    (HERE/'MARGIN_ALIGNMENT.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (HERE/'MARGIN_ALIGNMENT.md').write_text('''# 物理余量与任务定义适配

旋转木马的 params.py:4 SAFE_RATIO=1.1 在 cbf.py:33–34 定义机器人间安全距离2r×1.1和障碍安全半径r×1.1；论文 arXiv2503.05848 第5页V-A也明确列出物理半径0.2m和安全半径0.22m。因此把这项归为显式物理安全距离，而非求解容差。共同环境额外机器人间余量为0，障碍物几何已加墙体半径与碰撞余量，故此倍率映射为1。映射在自有适配层中设置运行时参数，不改作者文件。不同方法的碰撞判据始终相同，未放松共同碰撞条件。

临时圆环半径0.3、轨道增益0.05、预测时长0.5、圆环递增常数0.1、离开角规则和栅格膨胀缓冲是算法参数，保持原值。死锁感知控制的 agent_r 和墙体胶囊宽度已取共同配置；cbf_boundary_hold 为活跃约束判定量，保留作者值，不当作物理距离扣除。

旋转木马作者 agent.py:41 的到达阈值为radius/2（原场景0.1m）；既有共同适配层 paper_adapters.py:118 已设为 env.config.goal_tolerance=0.08m。死锁感知控制没有大于0.08m的硬停车死区；名义项中0.5m是方向饱和尺度，不是停车半径，不修改。Chen原环境复现未通过，退出，本轮不为其做后续余量适配。

异质场景原有不一致墙体模型与物理一致版本分别记录，不用负的墙体半径抵消过大的机器人半径。参数菜单在本诊断和版本合法率核验完成后才可冻结。
''')
    return changes

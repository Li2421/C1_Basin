from common import *
def main():
 assert not (HERE/'PREREGISTRATION.receipt.json').exists()
 text=(ROOT/'docs/prompt_round30e_decentralized_blackbox_planners_20261016.md').read_text()
 extra='''

# 执行前封存的操作定义
原任务全文如上；以下仅固定未指定的实现细节，不使用新评估标签。
- 三族各六实例，CONTROLLERS.json 与 RHR_GRID.json 为固定网格。分布式模型预测控制保留冻结实现对其余三机器人位置和速度的读取，不宣称其是 1.5 米局部感知。高斯行为克隆需要四机器人联合观测。
- A* 网格原点固定世界原点，节点坐标整数乘分辨率；中心/外环按 agent_radius+collision_margin 膨胀，邻居按2*agent_radius圆盘。八连通禁止切对角角点，边用解析线段到圆盘距离校验。起终点接到最近可见空闲节点（距离稳定排序）；不可达返回零动作。纯追踪从最近线段投影起沿折线向前0.3米，速度0.52乘min(1,目标距离/0.3)。每K步重规划，单步四次调用缓存保持幂等；每次reset清空状态。路径无碰撞指规划折线在规划瞬间膨胀地图上无碰撞，跟踪轨迹仍按原生仿真判定。
- 右手规则调参按所有未被复现门挡住的六实例、16调参state平均成功数选优（unknown下界0并另报上界）。平局依次速度小、触发更晚；窗口40晚于20晚于0；邻居规则距离0.8晚于1.2；剩余平局按模式顺序progress,slow,neighbor_right,always，再候选索引升序。跨模式时间与空间触发不可直接比较时先窗口降序、邻居距离升序、模式顺序。每族单独选择。
- 全部18个实例先做无终止200步R5探测，再用冻结critic推理，封存选择；只有之后才生成pilot/tuning/main评估标签。复现门轨迹是明确授权的校验重放，不作为新评估标签。R5投影介入指标为任一次投影改变整个动作矩阵范数超过1e-6；与历史学生R5定义相同。
- A3/A0固定模型种子17,23,41并取预测平均，对32评估state再平均选一个控制器级修正臂（四臂固定顺序F10,F13,S300,S500，argmax首项平局）。G_E沿用冻结变换与阈值0.05。另封存永远S500和永远F10。
- 每族进入主评估少于4实例时端点INCONCLUSIVE。能力门严格小于0.15、严格大于0.98。unknown不能被称作失败；能力门只在上下界都能确认排除条件时排除，否则保留并报告不确定性。
- 主端点按controller→state簇配对bootstrap5000次，两个seed在state内不重抽。随机种子2026101707，沿用round30g stats.py。未知成功的下界0，上界1；配对差采用交叉最坏/最好识别界，确认性通过必须最坏差CI下界>0。事后oracle按每controller四臂32state×2seed平均成功下界最大，平局固定臂序。
- 机制定义完整逐字复制round30g metrics.py和stats.py，不复制其事后敏感性分析。严格低速<0.01m/s且未到目标连续>=40步；在动无进展为40步窗口自身目标距离改善<0.05m且终点速度>=0.01，时长为满足端点数乘0.05秒的机器人秒；绕行累计角阈值严格±0.15，四个机器人全同非零才是全队一致；最后进展时刻按累计锚点总目标距离改善>=0.05m；前100步改善使用已知提前结束轨迹的终点吸收并报告提前结束。
- nominal-only若与round33同物理状态、RNG别名、checkpoint/实现、原生事件/终止规则逐字兼容则复用，不因critic编码或外层封装变更重跑。分布式安全层不同于round33联合安全层，不能复用其安全动作轨迹。所有复用保留源文件/trace哈希。
- 无训练，无参数结果后调整，unknown不重跑，不把timeout称为deadlock；合并学生表仅DEVELOPMENT。机制与critic比较描述性。源码/DB守卫失败整体停止，单族接口/复现/能力缺口继续其他族。
'''
 (HERE/'PREREGISTRATION.md').write_text(text+extra)
 paths=[HERE/'PREREGISTRATION.md',HERE/'CONTROLLERS.json',HERE/'RHR_GRID.json']+[HERE/n for n in ['core.py','guard.py','pipeline.py','episode_frozen.py','fair_rhr.py','controllers.py','legacy_controllers.py','gaussian.py','astar.py','runtime_new.py','metrics.py','stats.py']]
 seal('PREREGISTRATION.receipt.json',paths,sha256=sha(HERE/'PREREGISTRATION.md'),before_any_new_rollout=True,evaluation_labels_generated=0,guard=guard())
if __name__=='__main__':main()

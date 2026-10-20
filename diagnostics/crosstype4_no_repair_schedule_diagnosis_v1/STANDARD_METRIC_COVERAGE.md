# 共同协议指标覆盖

本轮具体动作清单优先于共同协议的通用菜单。所有结论为开发证据。

| 共同协议条目 | 本轮证据或缺口 |
|---|---|
| 不加安全投影 | 本轮B新状态清单未要求；N/A，不扩张动作预算 |
| 只加安全投影 | B全部74个评估controller、24状态、两个seed |
| 全程η10 | B新状态未要求；N/A，相对它的灾难数也为N/A |
| 人定双η时间表 | 第500步切换与按进度切换均完整评估 |
| 原冻结选择模型与探测规则 | 不属于本轮B指定候选和模型；N/A，不混入不同状态的旧结果 |
| controller信息 | 同一critic不看controller分支，开局选择、单独评估 |
| 完整菜单oracle及遗憾 | B每状态最多8个方法动作，未执行441动作菜单；完整菜单平均/最大遗憾为N/A，不能把有限已选动作最佳冒充完整菜单oracle |
| 提议oracle | A覆盖子集56个有序非对角对，按controller留一state交叉拟合，同时报告样本内上界；bootstrap重选 |
| 最佳固定切换步 | A按seed交叉拟合，bootstrap重选 |
| 专家 | 没有本轮特权专家标签，N/A |
| 碰撞、超时、unknown、伤害 | B_EVENT_TABLE.csv、B_CONTROLLER_RESULTS.csv及B_RESULTS.json |
| nominal→safety转移表 | 未执行nominal only新状态动作，N/A，不把旧不同state的转移率混入 |

数值失败保留unknown，点估计按下界；另列上界，配对差相同物理UID的不确定性抵消。timeout不直接视为deadlock。

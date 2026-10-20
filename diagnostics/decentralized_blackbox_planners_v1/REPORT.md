完整性审计通过，Basin_C1 共享数据库与默认 journal 路径断言通过。分布式模型预测控制（DM）在分布式安全层下超时率94.53%；集中式确定的前500步第13组、之后第10组修正时间表，成功率从5.47%变为12.24%，终点一通过；公平调参右手规则成功率100.00%，时间表优于它的终点二未通过（配对区间整体为负，所测右手规则更好）；冻结的通用critic相对控制器级四臂事后最优的平均遗憾为5.73个百分点，未达到所测四臂样本内最优。A*规划控制器（AS）在分布式安全层下超时率5.21%；集中式确定的前500步第13组、之后第10组修正时间表，成功率从94.79%变为39.58%，终点一不确定（不足4实例）；公平调参右手规则成功率100.00%，时间表优于它的终点二不确定（不足4实例）；冻结的通用critic相对控制器级四臂事后最优的平均遗憾为12.50个百分点，未达到所测四臂样本内最优。高斯行为克隆策略（G）在分布式安全层下超时率100.00%；集中式确定的前500步第13组、之后第10组修正时间表，成功率从0.00%变为83.59%，终点一通过；公平调参右手规则成功率0.00%，时间表优于它的终点二通过；冻结的通用critic相对控制器级四臂事后最优的平均遗憾为0.00个百分点，达到所测四臂样本内最优。三种冻结选择器对18个新实例均选择同一个前500步时间表，因此没有相对该固定默认方案的实际选择收益；其是否达到四修正臂的样本内最优另按上述遗憾评价，选择菜单不含右手规则。超时和停滞诊断不等于已证明死锁。

本轮按原任务执行，三新族的主终点为CONFIRMATORY；历史学生与合并总表为DEVELOPMENT。能力门选择会改变适用实例集合；每族少于4个实例时，不能据其成功率下结论。

## 冻结、复用与信息条件

原任务全文、固定网格、阈值、平局规则和统计方案见PREREGISTRATION.md。18个实例的576条无终止R5探测完成后才冻结选择，随后才提交能力门、调参和主评估。分布式安全层和模型预测控制各16条历史记录逐位复现；高斯策略checkpoint及归一化hash逐项核验，无训练。机制代码逐字复制round30g，只更改支持模块的导入路径，未沿用其事后敏感性指标。

分布式模型预测控制各机器人自行随机采样优化，但冻结实现使用其余三机器人的位置和速度预测。高斯行为克隆策略输入四机器人的联合观测，需要共享观测。A*只使用自身位置、自身目标与1.5米内邻居位置。三者都使用相同的分布式安全层与固定修正；不能将“自行计算动作”误写成三者都不通信或仅局部感知。

Round33主评估使用另一组物理状态和随机别名，不能复用为本轮Round28评估标签。14个原学生的6272条角色记录与原冻结选择全部复用，没有为它们重新训练或生成轨迹。新三族使用24候选调参；学生按任务要求复用历史20候选调参结果，没有补跑新增4候选。

## 能力门与右手规则调参

| 实例 | 只加安全层成功率未知界 | 前500步修正时间表成功率未知界 | 能力门 |
|---|---:|---:|---|
| DM0 | 43.75%–43.75% | 50.00%–50.00% | 进入主评估 |
| DM1 | 37.50%–37.50% | 50.00%–50.00% | 进入主评估 |
| DM2 | 43.75%–43.75% | 50.00%–50.00% | 进入主评估 |
| DM3 | 37.50%–37.50% | 43.75%–43.75% | 进入主评估 |
| DM4 | 31.25%–31.25% | 31.25%–31.25% | 进入主评估 |
| DM5 | 31.25%–31.25% | 37.50%–37.50% | 进入主评估 |
| AS0 | 100.00%–100.00% | 31.25%–31.25% | 进入主评估 |
| AS1 | 87.50%–87.50% | 100.00%–100.00% | 进入主评估 |
| AS2 | 100.00%–100.00% | 31.25%–31.25% | 进入主评估 |
| AS3 | 100.00%–100.00% | 100.00%–100.00% | 自身已处理交互、无需修正 |
| AS4 | 100.00%–100.00% | 100.00%–100.00% | 自身已处理交互、无需修正 |
| AS5 | 100.00%–100.00% | 100.00%–100.00% | 自身已处理交互、无需修正 |
| G0 | 0.00%–0.00% | 68.75%–68.75% | 进入主评估 |
| G1 | 0.00%–0.00% | 93.75%–93.75% | 进入主评估 |
| G2 | 0.00%–0.00% | 100.00%–100.00% | 进入主评估 |
| G3 | 0.00%–0.00% | 100.00%–100.00% | 进入主评估 |
| G4 | 0.00%–0.00% | 75.00%–75.00% | 进入主评估 |
| G5 | 0.00%–0.00% | 81.25%–81.25% | 进入主评估 |

能力不足排除名单：无。自身已处理交互排除名单：AS3, AS4, AS5。

分布式模型预测控制（DM）冻结右手规则：{"mode": "progress", "steps": 20, "speed": 0.35}；调参成功率100.00%，未知界上限100.00%，96条。完整24候选见RHR_GRID_RESULTS.json。
A*规划控制器（AS）冻结右手规则：{"mode": "neighbor_right", "distance": 0.8, "speed": 0.1, "steps": 0}；调参成功率100.00%，未知界上限100.00%，96条。完整24候选见RHR_GRID_RESULTS.json。
高斯行为克隆策略（G）冻结右手规则：{"mode": "progress", "steps": 40, "speed": 0.1}；调参成功率0.00%，未知界上限0.00%，96条。完整24候选见RHR_GRID_RESULTS.json。

## 逐族终点与失败机制

### 分布式模型预测控制（DM）

终点一（时间表减只加安全层）：PASS，保守95%CI [1.82, 13.02] 个百分点。
终点二（时间表减公平右手规则）：NOT_PASS，保守95%CI [-93.49, -81.25] 个百分点。
冻结通用critic的选择：{'DM0': 'S500', 'DM1': 'S500', 'DM2': 'S500', 'DM3': 'S500', 'DM4': 'S500', 'DM5': 'S500'}。遗憾5.73个百分点，未知包络95%CI [0.26, 11.72] 个百分点；相对一律前500步时间表的差CI [0.00, 0.00] 个百分点；相对一律全程第10组的差CI [-11.72, 0.78] 个百分点。这些比较为描述性。
尚未证实critic相对固定时间表有选择收益，也不把区间跨零解读为两者等效。
只跑控制器的碰撞对象（可重叠）：中心障碍384，外环0，机器人0；几何核验384条。只跑控制器绕行标签计数：{'0': 207, '1': 35, '2': 56, '3': 86}（0静止，1顺时针一致，2逆时针一致，3混合）。
绕行模式1/2表示出现非零绕行且没有反向机器人，可包含零标签机器人；严格四机器人同向非零另由既有four_same标记统计：{"nominal only": 0.010416666666666666, "safety baseline": 0.0546875, "S500": 0.125}。完整置信区间见STRICT_FOUR_DIRECTION.json。
只加安全层的363条超时局：在动但无进展平均11.003机器人秒，严格低速停滞平均33.775机器人秒、4.438段，最后显著进展平均发生在19.330秒，最终未到达2.584个机器人。任何严格低速停滞的比例为0.8292011019283747；混合方向比例为1.0。
时间表残留失败的平均终止时间（秒）：35.0，95%CI [34.99999999999999, 35.00000000000001]；应结合碰撞对象与超时类型阅读，不能把35秒预算耗尽称作死锁事件。
时间表在前100步相对只加安全层多取得总目标距离进展-0.013米，95%CI [-0.03936099717514256, 0.009939238834970983]。已知提前终止按原定义吸收。
自身名义超时且无碰撞0条，其中加安全层后仍超时0条；名义碰撞转为安全层超时363条。时间表救回计数：{'own_nominal_stall': 0, 'own_stall_persisting_under_safety': 0, 'safety_induced_stall': 26, 'baseline_timeouts': 26}。完整逐条转移及各臂救回计数见DM_FAILURE_COMPOSITION.json及FAILURE_RESCUES.csv。

论文表：[PAPER_TABLE_DM.md](PAPER_TABLE_DM.md)。

修正改善成功率，但未确认优于逐族调参右手规则；调参成本差异仍存在，不能宣称性能优势。 时间表减右手规则的配对区间整体为负，不能仅写成未证实优势；本轮所测公平右手规则的成功率更高。

### A*规划控制器（AS）

终点一（时间表减只加安全层）：INCONCLUSIVE，保守95%CI [-92.71, 8.33] 个百分点。
终点二（时间表减公平右手规则）：INCONCLUSIVE，保守95%CI [-94.79, 0.00] 个百分点。
冻结通用critic的选择：{'AS0': 'S500', 'AS1': 'S500', 'AS2': 'S500'}。遗憾12.50个百分点，未知包络95%CI [0.00, 25.00] 个百分点；相对一律前500步时间表的差CI [0.00, 0.00] 个百分点；相对一律全程第10组的差CI [-25.00, 61.46] 个百分点。这些比较为描述性。
尚未证实critic相对固定时间表有选择收益，也不把区间跨零解读为两者等效。
只跑控制器的碰撞对象（可重叠）：中心障碍0，外环0，机器人170；几何核验170条。只跑控制器绕行标签计数：{'0': 0, '1': 2, '2': 20, '3': 170}（0静止，1顺时针一致，2逆时针一致，3混合）。
绕行模式1/2表示出现非零绕行且没有反向机器人，可包含零标签机器人；严格四机器人同向非零另由既有four_same标记统计：{"nominal only": 0.11458333333333333, "safety baseline": 0.11458333333333333, "S500": 0.125}。完整置信区间见STRICT_FOUR_DIRECTION.json。
只加安全层的10条超时局：在动但无进展平均33.810机器人秒，严格低速停滞平均18.820机器人秒、1.000段，最后显著进展平均发生在15.950秒，最终未到达2.000个机器人。任何严格低速停滞的比例为0.4；混合方向比例为1.0。
时间表残留失败的平均终止时间（秒）：35.0，95%CI [34.99999999999999, 35.00000000000001]；应结合碰撞对象与超时类型阅读，不能把35秒预算耗尽称作死锁事件。
时间表在前100步相对只加安全层多取得总目标距离进展0.259米，95%CI [0.1570461049814972, 0.3656636855843819]。已知提前终止按原定义吸收。
自身名义超时且无碰撞0条，其中加安全层后仍超时0条；名义碰撞转为安全层超时10条。时间表救回计数：{'own_nominal_stall': 0, 'own_stall_persisting_under_safety': 0, 'safety_induced_stall': 6, 'baseline_timeouts': 6}。完整逐条转移及各臂救回计数见AS_FAILURE_COMPOSITION.json及FAILURE_RESCUES.csv。

论文表：[PAPER_TABLE_AS.md](PAPER_TABLE_AS.md)。

实例不足，不能确认该族的修正效果。

### 高斯行为克隆策略（G）

终点一（时间表减只加安全层）：PASS，保守95%CI [70.57, 95.31] 个百分点。
终点二（时间表减公平右手规则）：PASS，保守95%CI [70.57, 95.31] 个百分点。
冻结通用critic的选择：{'G0': 'S500', 'G1': 'S500', 'G2': 'S500', 'G3': 'S500', 'G4': 'S500', 'G5': 'S500'}。遗憾0.00个百分点，未知包络95%CI [0.00, 0.00] 个百分点；相对一律前500步时间表的差CI [0.00, 0.00] 个百分点；相对一律全程第10组的差CI [46.88, 65.36] 个百分点。这些比较为描述性。
尚未证实critic相对固定时间表有选择收益，也不把区间跨零解读为两者等效。
只跑控制器的碰撞对象（可重叠）：中心障碍140，外环119，机器人24；几何核验283条。只跑控制器绕行标签计数：{'0': 1, '1': 120, '2': 156, '3': 107}（0静止，1顺时针一致，2逆时针一致，3混合）。
绕行模式1/2表示出现非零绕行且没有反向机器人，可包含零标签机器人；严格四机器人同向非零另由既有four_same标记统计：{"nominal only": 0.4947916666666667, "safety baseline": 0.4140625, "S500": 0.8463541666666666}。完整置信区间见STRICT_FOUR_DIRECTION.json。
只加安全层的384条超时局：在动但无进展平均63.840机器人秒，严格低速停滞平均0.338机器人秒、0.047段，最后显著进展平均发生在26.544秒，最终未到达3.997个机器人。任何严格低速停滞的比例为0.0234375；混合方向比例为0.4921875。
时间表残留失败的平均终止时间（秒）：35.0，95%CI [34.99999999999999, 35.00000000000001]；应结合碰撞对象与超时类型阅读，不能把35秒预算耗尽称作死锁事件。
时间表在前100步相对只加安全层多取得总目标距离进展2.843米，95%CI [2.5137782144007756, 3.269658444931733]。已知提前终止按原定义吸收。
自身名义超时且无碰撞101条，其中加安全层后仍超时101条；名义碰撞转为安全层超时283条。时间表救回计数：{'own_nominal_stall': 101, 'own_stall_persisting_under_safety': 101, 'safety_induced_stall': 220, 'baseline_timeouts': 321}。完整逐条转移及各臂救回计数见G_FAILURE_COMPOSITION.json及FAILURE_RESCUES.csv。

论文表：[PAPER_TABLE_G.md](PAPER_TABLE_G.md)。

修正在该族通过两个确认性终点。

## 合并总表与限制

PAPER_TABLE_ALL.md/csv合并三新族符合能力门的实例与14个原学生，仅DEVELOPMENT；PAPER_TABLE_V0.md保留学生参照。总体均值按控制器等权，不能掩盖逐族结果或将共享观测高斯策略宣称为不通信。unknown按成功下界0与上界1报告，unknown不重跑且不记作已知失败。事后最优是同一评估集上的控制器级四臂选择，属于描述性上界，未做交叉验证。

A*两个seed的动作/位置/速度逐位一致核验见ASTAR_SEED_IDENTITY.json。没有改变物理、事件、成功标准、安全层、修正、冻结权重，没有安装包，没有git写操作，不交付视频。

## 审计、数据库与执行记录

保护审计核查523766个已有文件。结束审计通过，新增外部Python文件['/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/audit_inputs.py', '/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/verify_and_manifest.py', '/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/certify_arithmetic.py', '/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/agent_b/compute.py', '/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/agent_a/certify_geometry.py', '/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/agent_d/compute_coverage.py', '/home/zhihan/research/Basin_C1/diagnostics/toy_K_quantitative_certification_20261017_v1/agent_c/certify_pass.py', '/home/zhihan/research/Basin_C1/diagnostics/tt_ff_correction_realization_20261017_v1/verify_and_manifest.py', '/home/zhihan/research/Basin_C1/diagnostics/tt_ff_correction_realization_20261017_v1/analyze_saved.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/train_policies.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/single_agent.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/runtime.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/reproduction.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/test_geometry.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/geometry.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/observation.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/controllers.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/write_blocked_status.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/core.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/policies.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/submit_reproduction_when_free.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/sequential_expert.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/design_generated.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/common.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/catalog.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/generated_worker.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/registry.py', '/home/zhihan/research/Basin_C1/diagnostics/multiscene_generalization_v1/audit.py']；Git状态前后详见INTEGRITY_COMPARISON.json，轮外新增不归因于本轮。导入源文件逐项hash一致，见IMPORTED_SOURCE_END.json。Git轮外新增条目另列EXTERNAL_GIT_STATUS.json，并明确不由本轮写入。

DB采用只追加新身份、封存preflight、缺失任务、默认journal、单进程官方merger和postflight；registration/每阶段保存既有登记行前后hash及插入行hash。各阶段的journal唯一性已核验；行为探测仅作为SUMMARY_ONLY来源，没有任务rollout标签。

最初复现作业22825因debug分区时限配置而一直PENDING，取消后改用cpu分区22826，未产生轨迹且没有科学重跑；预注册的路径适配补充发生在新轨迹之前并保留初始收据。资源以全部用户RUNNING/PENDING数组元素预留，提交最多18个CPU shard，没有GPU shard。详细提交和队列记录见各阶段SUBMISSION与SUBMISSION_PREFLIGHT收据。

科研图件见figures/family_success.png、figures/family_endpoints.png（均附PDF），仅根据封存仿真证据绘制。
当前缺口：[{"phase": "main", "kind": "PRESERVED_UNKNOWN", "count": 1, "rerun": false}, {"family": "AS", "kind": "FEWER_THAN_FOUR_ELIGIBLE", "count": 3}]。所有原始证据保留。

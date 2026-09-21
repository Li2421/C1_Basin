# C1 实现状态（2026-09-12 审计）

2026-09-13 更新：已完成原数据集初态方案的代码审计与一轮漏洞修复，完整
87 项回归通过。新增 dataset 训练分布、集合成员核对、验证哈希、非有限风险
拒绝、原子检查点/历史恢复，以及与基线一致的六类正式汇总。详细证据和当前
真实回合诊断状态见 `../../results/C1_DATASET_LOGIC_AUDIT.md`。旧宽初态校准/
混合训练实验已停在 30 次更新，未产生改善。下文保留历史实现记录。

原数据集方案的 200 回合校准、25 回合验证与 2 次更新已完成：第二次实际
梯度非零，参数发生更新；验证风险 0.141873→0.148636，仍高于 epsilon
0.112859，无可行候选。验证前后均 25/25 成功、无碰撞，不能声称学习改善。

最新迭代：默认候选已更新为 `R_risk_v2`（`balanced_terminal_v3`）。
它使用原环境首终止事件掩码和完整时域，修复末动作遗漏、warmup 几何遗漏、
末尾失败被前段稀释、OR 饱和抹除梯度，以及近容差超时反而低于成功的问题。
v2 的失败/成功分离是显式代价设计，不是预测准确性的证据；事件跳变处不声称
无偏 BPTT 梯度。下述 v1_1 状态为上一轮记录，完整最新记录见
`../../results/C1_RISK_ITERATIONS.md`。

## 上一轮状态

控制器仍为 C1 V0：冻结 Flow-BC → baseline nominal → Safety 投影 →
控制空间残差 → 最终硬投影。训练保持单策略 primal-dual 目标及状态 BPTT。

当前默认临时风险为 `R_risk_v1_1`。它保留 soft activity、解析锥、联合任务势、
物理时间进度窗口、风险合成和聚合，修正未完成判据为逐智能体容差的平滑 OR，
并修复零风险处 NaN 梯度。旧 `R_risk_v1` 显式保留旧联合距离判据回放。
评估现按检查点风险版本记录风险、活动集、原始/实际修正及最终投影统计。

**尚未完成研究目标。** 当前训练是固定长度、非终止式的可微片段代理；
完整任务评估才使用首个终止事件。短片段约束成立不等于完整任务活性成立。
尚无通过独立验证的 v1_1 长训练结果，也未完成该版本双 seed 的固定 400 回合对照。
新增 smoke 仅验证训练链路，不能宣称解除死锁。

详细核对、修复范围和下一步见 `../../results/C1_LOGIC_AUDIT.md`。

## 历史结果（不代表当前 v1_1）

以下 soft activity 实验使用 `R_risk_v0_1`；硬 `R_risk_v0` 活动集与 TRO
几何为独立诊断。历史入口 `scripts/experiment_c1_soft_activity.py` 与
`scripts/report_c1_soft_activity.py` 仍使用旧软风险，不是当前版本的训练入口。

软风险修订后 55 项测试通过。200 步、8 次更新及完整任务复评已完成：
R_risk 0.883994 → 0.567905，但仍为 safe_deadlock，进度反而变小。
风险下降约 98.66% 来自 CBF slack 增大，不能算死锁解除。
完整报告与轨迹见 `results/c1_soft_activity_edge/REVIEW.md`。

当前活动实现为 C1 V0：冻结 Flow-BC → baseline nominal → Safety 投影 →
控制空间残差 → 最终投影。原 Flow 向量场残差已从活动路径移除，旧 residual
checkpoint 会被 V0 评估入口拒绝。旧实验以下作为历史记录保留，不是 V0 结果。

此前 A–F 修订的风险为 R_risk_v0（现保留为硬诊断）；先前 `c1_seed0_risk_audit.json` 使用有缺陷的旧锥实现，
其数值不能用于验证 V0。新解析与多步梯度测试位于 `tests/test_c1_v0.py`，
短 smoke 入口为 `scripts/smoke_c1_v0.py`。未执行 V0 长训练或 400 回合评估。

唯一对照基准为 ../../baseline_309_314：两个 25k Flow-BC checkpoint，MAC-only 309/400，Safety 314/400。
训练入口为 `single_integrator.c1.train`，两个 shell 入口为
`scripts/train_c1_309_314.sh` 与 `scripts/plan_c1_309_314.sh`；400 条聚合入口为
`scripts/plan_c1_400.py`。它们锁定短场景、两个 checkpoint、200 测试初态、seed=42、
850 步及原 Safety CBF。

已完成一项可复现的 C1 seed-0 试运行：`results/c1_primal_dual_seed0`
以 4 条起点、48 步 horizon 训练 20 个 primal-dual update，并在固定的 200 个宽初态、
Flow seed=42 上完成评估。完整轨迹、配置、汇总和完成标记在
`results/eval_c1_primal_dual_seed0/`；汇总为 success 154/200 (77.0%)、wall 0、agent
0、safe deadlock 7/200 (3.5%)、other timeout 39/200 (19.5%)。

该结果只覆盖 C1 seed=0，且训练仅 20 updates；它不是 README 所列的两个冻结 baseline
seed、400 回合主结果的替代品。可与 baseline seed=0 的 200-start Safety 行作诊断性比较，
但在同时训练、评估 C1 seed=1 并按相同聚合协议汇总前，不应报告为正式 400 回合对照或性能
提升结论。
# 2026-09-13 optimizer safeguard

Completed 8-update seed-0 experiment: 7 accepted parameter updates, 13 uphill
proposals rejected. Validation risk 0.14187327 -> 0.13940685, still above
epsilon 0.11285929; no feasible checkpoint. Paired 25-case/1200-step planning
remains 20/25, versus Safety 22/25 (lost IDs 2 and 10). This fixes a measured
optimizer failure but has not resolved success-rate regression. Evidence:
`../../results/C1_GUARDED_OPTIMIZATION_RESULTS.md`.

The trainer now replays each Adam proposal on the same batch/noise with lambda
held fixed, backtracking by halves on increases of the existing C1 primal
objective. Rejection rolls back both parameters and optimizer moments; the
pre-step dual update is unchanged. History includes pre/post metrics and trial
losses. Protocol metadata prevents silently resuming an old unguarded run.
This does not certify constraint feasibility or improved planning success.
`scripts/plan_c1_309_314.sh` defaults to the current 25-case/1200-step protocol
and forwards explicit CLI overrides instead of ignoring them.

# C1 Part 3 Round 3 — 真实 Flow checkpoint 条件 Q 迁移可学习性（2026-10-12）

## 结论

**`TRANSFER_GATE_FAILED`。** 在已有 14 个真实 Ring FF Flow checkpoint、24 个已打开的开发 state、共同 K16 和 Q4 标签上，严格源域选模的 R5 软匹配把 held-checkpoint Q MSE 从 R3 的 **0.0714** 降到 **0.0675**，但 K16 top-1 经验 Q 从 **0.5945** 降到 **0.5856**；两者都低于源域固定 η₁₀ 的 **0.6019**。R5 的表示信息确实可改善整体 Q 回归；目前没有稳定的 controller-conditioned selection transfer。低维 R3+near 的 Q MSE 为 **0.0682**，选择结果与 R5 几乎相同，也未过门。此处是打开过的 P1 数据上的留一真实 checkpoint 诊断，不是独立新标签确认，更不是可认证迁移保证。

## A. 历史证据与失败位置

执行前复查了 [P1 原始协议、Q4 标签及报告](../diagnostics/controller_q_transfer_round3_v1/REPORT.md)、[Round 2 R3/R4/R5 静态表示和最近源结果](part3_round2_necessity_representation_20261012.md)。相同 execution semantics 的 P1 标签全部沿用；本轮新增 rollout **0**、GPU **0**、静态 probe **0**。不更改 controller、OrthoFlow3、CBF、安全投影、评价器或旧标签。P1 的 24 state 已经打开，且 14 checkpoint 是现有历史库存；因此 outer LOCO 只保证每次预测及选模不读取该折目标标签，不能把这 24 state 当成独立封闭测试集。

旧最近源诊断按每个 state 找单一距离最近 checkpoint，R3/R5 的 Q MSE 分别为 **0.0826/0.0843**，相对源固定 η 的选中 Q 分别为 **−0.1257/−0.1436**。这只检验了最近邻读出和旧坐标缩放，不能判定 R5 不含可用信息。Round 2 已证明 R5 保留指定 pose/time 的 near-goal moving response，旧 R3 的聚合映射存在代数信息丢失；但在真实 P1 checkpoint 对中未观察到精确的 R3 行为别名。

本轮将五种可能因素分开检查。第 3–5 项中的 split-half、误差分项及候选稳定性是在冻结消融结果之后计算的**描述性诊断**，不参与选模或通过门判定：

1. **表示信息丢失**：R5 包含 R3 和额外 indexed 响应，信息保真成立；本批真实 checkpoint 没有精确 R3 别名，因而不能把选择失败单独归因于 R3 丢失信息。
2. **尺度与无关维度**：内层源域验证在 R5、R3+near 的全部 14 折都选了语义块 RMS 缩放；R3 的 coordinate/块缩放分别被选 6/8 折。软匹配后 R5 的 Q MSE 优于 R3，推翻“旧最近源退化证明 R5 无用”的推论。它也表明新增维度需要缩放；不同指标最优并不说明每一维都与成功相关。
3. **controller×state×η 可学习性**：目标 Q2 两半的 state×η 残差交叉积为 **0.02108**，显示有限面板中存在可重复变化；R5 预测残差与两半目标的交叉积只有 **0.00207/0.00224**，R3 为 **0.00194/0.00179**。当前源条件匹配捕捉这部分结构很弱。该交叉积是有限 seed 估计，不能直接当总体信号下界。
4. **critic 近似误差**：R3→R5 的 MSE 改善主要在 controller 水平项 **0.01153→0.00933** 和平均 η 曲线项 **0.02186→0.01879**；state×η 残差项 **0.03215→0.03318** 反而增加。预测的交互变化未对准决定 top-1 的局部排序。当前轻量读出能证明某些 R5 信息可用，不能证明更强模型必然可学或表示已经充分。
5. **候选 headroom 与 Q4 噪声**：源固定 η 在 14 折始终是原 K16 的 **η₁₀**，即使把源 numerical-unknown 全部置为成功或失败也不变。目标 Q4 hindsight oracle 的平均 Q 区间 **[0.7388,0.7679]**，固定 η 的 **[0.6019,0.6116]**；但目标两半 seed 交叉拟合 oracle 相对固定 η 只有 **[+0.0327,+0.0461]**，14 checkpoint 中 8 个下界正、5 个上界负。Q4 top-two 间隔中位数 **0**，**79.2%** 的 checkpoint×state 间隔 ≤0.25；Q2 两半的 argmax 只在 **62.5%** 的 checkpoint×state 相同。完整 cell 的 split-half Q4 噪声 MSE proxy 为 **0.00667**；这远小于当前总 MSE，但不同 cell 的异方差、未知值和 argmax 不稳定仍会影响选择。Q4 hindsight oracle 不可视为真实总体 oracle。

## B. 冻结的源域学习消融

[预先保存的分析方案](../diagnostics/part3_transfer_feasibility_round3_20261012/PREREGISTRATION.md)与[输入/脚本 SHA256 收据](../diagnostics/part3_transfer_feasibility_round3_20261012/frozen_hashes.sha256)在新消融结果产生前固定。每个外层折留出一个真实 Flow checkpoint；其余 13 个源 checkpoint 内层再逐一留出选超参数。目标 Q4 标签只进入最终评分。源 numerical-unknown 以 Q 区间中点供拟合和内层选模；最终 Q MSE、top-1 Q、成对收益和遗憾另给 unknown 安全区间，未把 unknown 当失败或删除 cell。

R0 是同 physical state、同 η 的源 controller 平均。其他臂使用同一个容量受控的源条件 soft matching：根据 held controller 在当前 state 的静态描述与各源 controller 的距离加权源 `Q(source,state,η)`，再按内层选择的权重收缩到 R0。输入明确包含当前 state 的匹配行、controller 静态响应和 η 的 K16 输出位，故能表达 state×controller×η 交互；该任务的 24 个 state 在所有 checkpoint 间共享，**没有评估新 state 泛化**。固定候选包括 R3、完整 R5、R3 加仅近目标 moving 的 indexed 2×2 Jacobian（额外 144 维，R5 额外 696 维）、R3+R4_action、R5+R4_action。两个缩放法、四个匹配温度及四个收缩权重均只按 13 源的内层 LOCO Q MSE 选择。配置、每折选项、所有预测和代码分别见 [results.json](../diagnostics/part3_transfer_feasibility_round3_20261012/results.json)、[outer_predictions.npz](../diagnostics/part3_transfer_feasibility_round3_20261012/outer_predictions.npz)、[ablate.py](../diagnostics/part3_transfer_feasibility_round3_20261012/ablate.py)。

| 源域选模臂 | Held Q MSE（Q4 中点） | Unknown 安全 MSE 区间 | state 内 rank Spearman | K16 top-1 Q 下界 | 对源固定 η 的 Q 增益区间 |
|---|---:|---:|---:|---:|---:|
| R0 | 0.0814 | [0.0801,0.0844] | 0.394 | 0.5677 | [−0.0350,−0.0327] |
| R3 | 0.0714 | [0.0701,0.0744] | 0.421 | **0.5945** | [−0.0074,−0.0052] |
| R5 | **0.0675** | [0.0662,0.0704] | 0.423 | 0.5856 | [−0.0171,−0.0134] |
| R3+near | 0.0682 | [0.0669,0.0712] | **0.425** | 0.5856 | [−0.0171,−0.0134] |
| R3+R4 | 0.0707 | [0.0694,0.0737] | 0.423 | **0.5967** | [−0.0052,−0.0030] |
| R5+R4 | 0.0679 | [0.0666,0.0709] | **0.425** | 0.5885 | [−0.0141,−0.0104] |

完整 top-1 Q 上界、经验 Q4 oracle regret、确定性非并列 η 对准确率和误差分项均在结果 JSON。R5 相对 R3 的逐 checkpoint Q MSE 降低均值 **0.00395**，14 折中 11 折降低；14-checkpoint 重抽样 95% 区间 **[0.00057,0.00780]**。同一比较的选中 Q 下界变化为 **−0.00893**，区间 **[−0.02083,+0.00223]**，只有 3 折改善、6 折退化。R3+near 的 MSE 改善相对 R3 为 **0.00321**，区间跨零，选择变化与 R5 相同。这里的 checkpoint bootstrap 是对选定且已打开库存的描述性不确定性，不能提升为 controller population 的置信结论。

### 逐 checkpoint 选择审计

下表均为 target Q4 **成功下界**，每折对所有 24 state 平均；完整 unknown 上界及 Q MSE 在 [逐 checkpoint JSON](../diagnostics/part3_transfer_feasibility_round3_20261012/results.json)。`wide/targeted/plus-w` 分别沿用 P1 原 checkpoint 名称。

| Held Flow | Unknown seeds | 固定 η₁₀ | R0 | R3 | R5 | R3+R4 |
|---|---:|---:|---:|---:|---:|---:|
| v4 wide plus-w | 16 | 0.646 | 0.625 | 0.646 | 0.594 | 0.646 |
| base u | 24 | 0.240 | 0.271 | 0.344 | 0.333 | 0.344 |
| v4 | 25 | 0.344 | 0.333 | 0.396 | 0.396 | 0.365 |
| v6 plus-w | 24 | 0.312 | 0.208 | 0.271 | 0.250 | 0.271 |
| v3 plus-w | 30 | 0.906 | 0.854 | 0.865 | 0.875 | 0.865 |
| v4 targeted plus-w | 25 | 0.469 | 0.406 | 0.448 | 0.406 | 0.448 |
| v3 | 33 | 0.771 | 0.688 | 0.708 | 0.708 | 0.708 |
| v2 | 23 | 0.396 | 0.396 | 0.396 | 0.427 | 0.396 |
| v10 local | 9 | 0.656 | 0.667 | 0.667 | 0.667 | 0.667 |
| v5 plus-w | 21 | 0.385 | 0.365 | 0.396 | 0.396 | 0.396 |
| v7 local | 2 | 0.990 | 0.917 | 0.906 | 0.906 | 0.906 |
| v4 plus-w | 28 | 0.417 | 0.406 | 0.396 | 0.354 | 0.417 |
| v8 local | 3 | 0.990 | 0.917 | 0.948 | 0.958 | 0.990 |
| v9 local | 9 | 0.906 | 0.896 | 0.938 | 0.927 | 0.938 |

R5 在 v2 有约 **+0.031** 的固定 η 以上选择收益，但在 v7 local **−0.083**、v3 **−0.063**、v4 plus-w **−0.063**。不能以少数正例掩盖这些退化。R5 相对 R0 的选中 Q 增益 unknown 安全区间 **[+0.0164,+0.0208]**；相对固定 η 却为 **[−0.0171,−0.0134]**，只有 **6/14** 折的固定 η 比较下界非负。R3+R4 虽有最高的 top-1 Q，固定 η 比较仍为 **[−0.0052,−0.0030]**，非负下界 **9/14** 折，未满足预注册的至少 10 折要求。

## C. 选择评价与适用边界

共同 K16 和同一套真实 P1 Q4 标签用于所有臂。目标经验 oracle Q 区间 **[0.7388,0.7679]**；R3/R5 的 unknown 安全 oracle regret 分别 **[0.1391,0.1682] / [0.1481,0.1771]**。这是相对目标 Q4 hindsight max 的有限样本遗憾，不能推导为真实总体后悔界。R3、R5、R3+near、R3+R4、R5+R4 分别有约 **265、260、261、267、259 /336** 个 state 选择 η₁₀；这些读出主要沿用源固定候选，有限的 state-specific 改选尚未可靠提高 Q。

源域盲预测 R0 的固定 η 比较为负 **[−0.0350,−0.0327]**，因此超过 R0 的收益只是较弱的门；本轮明确要求同时超过 R0、R3 与 source-fixed。R5 的平均 Q MSE 胜 R0/R3，R5、R5+R4 的 top-1 Q 胜 R0，R3/R3+R4 的选择又胜 R5，但**没有任何 controller-informed 臂同时达到 Q MSE、固定 η 选择收益、10/14 折一致性和 checkpoint bootstrap 正下界**。R5 的 unknown 安全固定 η 比较本身上界仍为负，因此不能由处理 numerical-unknown 的方式翻转总体结论。

## D. Decision gate 与最小下一步

| 问题 | Verdict | 依据 |
|---|---|---|
| R5 是否改善条件 Q 可学习性？ | **`PARTIAL`** | 同一源域选模读出下，Q MSE 优于 R3/R0；改善主要在水平和平均 η 曲线，未改善 top-1。 |
| 低维近目标增强是否更有效？ | **`REJECTED`** | 与 R5 选择几乎相同，Q MSE 略高；未改善固定 η 收益。 |
| 失败主要在表示还是学习？ | **`PARTIAL`：主要表现为交互读出/选择缺口** | R5 含旧遗漏信息且改善 Q MSE，但可重复 state×η 残差预测弱；Q4 argmax 噪声与 checkpoint headroom 异质性也限制结论。尚不能唯一归因于某一机制。 |
| 稳定超过 controller-blind 与 source-fixed 的 held-checkpoint selection？ | **`REJECTED`** | 无臂同时通过两种基线、逐 checkpoint 一致性及预定不确定性门。 |
| 是否进入新 held Flow Q16？ | **`TRANSFER_GATE_FAILED`** | 当前没有源域冻结模型可作可靠确认对象；不启动新 rollout 或大规模训练。 |

下一步应先在**已有**标签上做一项窄机制检验：预先固定对 state×controller×η 残差的分层、以独立 seed 半估计可重复目标，并只用源 checkpoint 判断该残差是否可由静态 controller response 与物理 state 预测。需报告每个 checkpoint 的方向与误差，并与 η₁₀ 的决策边际相联。若该信号在源域内都无法复现或内层选模不能稳定选出增益，便没有启动 held Flow Q16 的理由；若以后出现可靠源域门，再另行冻结新 checkpoint、state、候选、模型及预算。现阶段不得通过继续堆 fingerprint、放大网络或事后挑 checkpoint 追求正结论。

理论边界与 [Part 3 主报告](part3_theory_validation_20261012.md)一致：行为距离是 sampled probe 量，不是全域 Lipschitz 常数；Q4 迁移误差是所选库存上的经验误差，不能代入非空的可认证理论 bound。Round 2 的固定 pair、双候选 necessity 经验确认仍成立；Round 3 **没有**验证广义 controller-conditioned selector 或跨 controller generalization claim。Part 3 目前不具备这两项 RSS 强 claim 的 paper-ready 证据。

## 复现

[实验目录](../diagnostics/part3_transfer_feasibility_round3_20261012/)含冻结方案、哈希、`ablate.py`、`diagnose.py`、完整逐折结果、预测、运行日志和二级诊断。命令与审计写在 [RUN_LOG.md](../diagnostics/part3_transfer_feasibility_round3_20261012/RUN_LOG.md)。

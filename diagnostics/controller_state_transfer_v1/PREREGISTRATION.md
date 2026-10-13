# 预注册：方法本身是否存在 state 条件化的泛化缺陷（2026-10-11）

本文件在任何新 outcome 产生之前写定，sha256 冻结进 protocol。

## 问题

上一轮实验（`controller_geometry_identifiability_v1`）已排除以下四点：场景数量、环境内数据量、controller 信息、entity 几何表示。

剩下未检验的是：**在“条件允许”的情况下，项目的 critic 能不能学会并跨几何迁移随 state 变化的 η 偏好（state×η）？**

所谓“条件允许”，指的是：
- 只有一个几何因素在变化；
- 有多个几何；
- 每个几何上使用同一批物理 state；
- 并且确实存在强的 state 依赖。

如果在这种条件下仍然失败，而且失败不能用“信息不存在”来解释，就是方法本身的缺陷。

## 设计

**场景与几何变量**
- 场景：Toy Give-Way（单一机制：让行 bay）。
- 唯一的几何变量：`corridor_width`，它同时决定走廊宽度和 bay 开口宽度。合法区间由原 `Config` 校验，为 (0, 0.64)。
- 宽度共 8 个：0.41 / 0.44 / 0.47 / 0.50 / 0.53 / 0.56 / 0.59 / 0.62。
  - 刻意避开 0.40，以免与存档 rollout 重复。
  - 0.40 只用来做逐字节复现检查。

**Controller（2 个，权重冻结，在所有宽度上相同）**
- `flow_isotropic`：冻结的 Toy Flow 端点，加 isotropic FF 算子。
- `flow_anisotropic`：同一个 Flow 端点，加 anisotropic FF 算子。
- 两者都通过 `controller_flow_endpoint_probe_v1.runtime.FlowEndpointMatchedField` 执行，与 factorial 实验的执行路径相同。
- 原生 `flow` 是由外部 runner 执行的，无法逐字节复现，因此不使用；这一修改在产生任何 outcome 之前做出。

**η**：取原 16 候选中的 8 个，索引为 0, 2, 4, 5, 8, 10, 12, 15（包含原 K4）。

**State**
- 使用 `controller_ff_factorial_repair_v1` 的 Toy train 初态。
- study_index 0–23 作为训练 state；24–31 作为全新 state，任何学习器训练时都不使用。

**宽度划分**

| 用途 | 宽度 |
|---|---|
| 训练 | 0.41 / 0.47 / 0.53 / 0.59 |
| 留出（内插） | 0.44 / 0.50 / 0.56 |
| 留出（外推） | 0.62 |

**Seed**
- 训练宽度上的训练 state：seed 0 和 1。
- 训练宽度上的新 state：seed 0–3。
- 留出宽度上的全部 state：seed 0–3。

总共 13,312 次 rollout。

**其余全部固定**：动力学、horizon、安全层、FF、评价器、噪声，以及 `key_for(state, seed, t)`（在不同宽度间共用同一组 common random numbers）。

**流程要求**：先复现门槛，再做 shared_rollout_db 的 preflight，最后 merge。按 AGENTS.md，这是纯 Q 实验，不需要视频。

## 学习器（固定设置，不调参）

两种学习器都只用训练宽度上训练 state 的 seed 0/1 标签来训练。

**ENT（项目 critic）**
- 结构：`Critic('state_field_compact')`。
- 几何：墙体 obstacle entity，由 `GiveWayEnv(config).walls` 生成。
- Controller：每个 agent 一个 2 维 one-hot。
- η：3 维。
- 训练：6000 步，batch 120，AdamW(8e-4, 1e-4)，梯度裁剪 5，逐 rollout BCE，seed 17/23/41 集成，取最终参数。

**MLP（通用参照）**
- 输入：w、位置、目标、速度、C one-hot、η（标准化后）。
- 结构：3×128 SiLU。
- 训练：3000 步，batch 256，AdamW(1e-3, 1e-4)，seed 17/23/41 集成。

**Lookup（非学习参照）**
- 对同一个训练 state，在最近的训练宽度上直接查观测到的 Q（seed 0/1）。
- 用来给出“同一 state、跨几何”这条路径上信息可得性的上界。

## 评估对象

在每个 (w, C) 上考察 state×η 模式：P[s, η] = Q − mean_s − mean_η + grand。真值用评估宽度上 seed 0–3 的标签计算。

| 名称 | 宽度 | state | 含义 |
|---|---|---|---|
| T1 | 训练宽度 | 8 个新 state | 几何内、对新 state 的泛化 |
| T2 | 留出宽度 | 24 个训练 state | 跨几何、同一批 state |
| T3 | 留出宽度 | 8 个新 state | 同时面对新几何和新 state |

## 指标（只在评估集上计算）

**m1：state×η 模式相关**
- 计算预测模式与真值模式的相关，在 (w, C) 上取平均。
- 信度上限 = sqrt(真值 seed 0/1 与 seed 2/3 两半之间的相关)。

**m2：无偏交互 skill**
- 定义：1 − MSE(P_pred, P_truth) / energy。
- energy 用两半 seed 的交叉乘积估计，因此是无偏的。

**m3：state 使用**
- 对每个 (w, C, s)，按 argmax 预测值选 η，计算实际成功率（seed 2/3）。
- 用同一模型的固定 η（取 mean_s 预测值最高的那个 η）作为对照，差值即“state 自适应增益”。
- 再与“错配 state”对照：用同一宽度内其他 state 的预测来选 η，做 200 次随机排列。matched − mismatched 即为 state 使用。
- 以上都用 state bootstrap 给 CI。

## 判定规则

**前提（可评估性）**
- 留出宽度上真值模式的信度 ≥ 0.4。
- T1/T3 的新 state 上，真值 state×η energy 的 CI 下限 > 0。
- 不满足时，判为“该几何族不可评估”，不下结论。

**方法缺陷判定**（只要 ENT 满足任一条，即判为方法缺陷）：
- **D1 跨几何 state 迁移缺陷**：lookup 在 T2 的 m1 ≥ 0.5，但 ENT 在 T2 的 m1 ≤ 0.5 × lookup。
  - 含义：信息存在、并且可以跨几何迁移，但 critic 没有用上。
- **D2 state 泛化缺陷**：真值可评估，但 ENT 在 T1 的 m1 ≤ 0.2，且 matched − mismatched 的 CI 包含 0。
  - 含义：在已知几何内，对新 state 都不能泛化。
- **D3 新 state × 新几何的额外缺陷**：ENT 在 T3 的 m1 < 0.5 × T1，或 T3 的 matched − mismatched CI 包含 0，而 T1 的为正。

**排除方法缺陷**需要同时满足：
- ENT 在 T2 的 m1 ≥ 0.5 × lookup，并且 > 0.3；
- T1 与 T3 的 m1 都 > 0.3；
- T3 的 matched − mismatched CI 下限 > 0。

**表示特异性**
- ENT 失败而 MLP 成功（同一判定）：缺陷在 entity 表示。
- 两者都失败：缺陷在“从结果标签学习 state 条件”这一更一般的层面（样本量、目标或优化）；同时报告 lookup，以区分信息是否存在。

## 范围

这是单一机制、单一几何族、2 个 controller、K8 η 的 DEVELOPMENT 实验，不证明跨机制的迁移能力。它针对的是“方法自身”这一层：如果方法连这里都过不了，那么四场景的失败就不能只归因于条件。

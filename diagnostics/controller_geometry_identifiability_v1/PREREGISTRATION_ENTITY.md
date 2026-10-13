# 预注册补充：项目 entity-encoder critic 在 DB 宽度族上的迁移（2026-10-11）

本文件在任何 entity-critic 拟合之前写定。实验不新增 rollout，复用 main 阶段的 13,440 个标签，以及同样的训练集划分（MANY/FEW/FEW_x2/EXTRAP_A/EXTRAP_B）、test 宽度和指标（`analysis.evaluate`：B1、B2、选择）。

## 学习器

- **网络结构**：`controller_q_diversity_v1.train.Critic('state_field_compact')`，即 `rep.Encoder` 加 η 编码，再接 128/64 的 trunk。

**几何输入：**
- 每个宽度的墙段都来自 `double_bottleneck.scenario.build_layout`，并以 obstacle entity 表示（与 `orthoflow3_unified_representation_v1.parse` 相同）。
- Flow 通道为 0，globals 为 DB 原配置的常量，在各宽度间不变。
- 因此几何只能通过 obstacle entity 进入模型。

**controller 输入**：每个 agent 一个 5 维 one-hot，表示精确的算子身份，拼接到 agent 通道上。

**优化**：采用项目原设置——6000 步，batch 120，AdamW（lr 8e-4，wd 1e-4），梯度裁剪 5，逐 rollout BCE，seed 17/23/41，取最终参数，不做 checkpoint 选择。

## 两个对照臂

| 臂 | 输入 |
|---|---|
| ENT | 只有 entity |
| ENT+w | 与 ENT 相同，另在 globals 末尾追加显式宽度 w；用来排除 encoder 优化本身的问题 |

## 预测与解读

以下用 EXTRAP 指 EXTRAP_A 与 EXTRAP_B 两个外推臂，B2 取两者在各自目标宽度上的平均。

| 编号 | 条件 | 解读 |
|---|---|---|
| R1 | ENT 在 EXTRAP 上的 B2 与 MLP(w) 相差在 0.1 以内（MLP(w) 为 0.63），且 FEW 的 B2 ≥ 0.6 | 在有共享坐标的几何族中，项目表示不是瓶颈；四场景失败更可能源于各场景没有共享的几何坐标 |
| R2 | ENT 在 EXTRAP 上的 B2 ≤ 0.3，而 ENT+w ≥ 0.5 | 项目的几何表示（entity 编码）本身就是瓶颈 |
| R3 | ENT 与 ENT+w 在 EXTRAP 上都 ≤ 0.3 | 该 encoder/训练组合本身难以外推，问题不限于几何表示 |
| 其他结果 | — | 按数值如实报告，不归入以上任何一类 |

## 范围限制

这仍是同一个几乎不依赖 state 的几何族（state 相关能量占 0–6%），所以本实验不检验 state 条件化能力。

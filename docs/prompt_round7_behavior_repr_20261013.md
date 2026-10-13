你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不生成任何新 rollout，只用已有标签、已有的行为探测数据和冻结模型，训练小型 critic 与分类器，资源占用很小。**

# 一、目标与本轮问题

研究总目标是让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在**未见过的 controller** 上选出更好的 η。

到 round6 为止的进展分两部分：

**已定位并修复（DEVELOPMENT 证据）：“给 controller 信息反而有害”的原因。**
- 原因是 critic 的 controller 通路缺乏归纳偏置：输入维度高、没有约束，加上 argmax 会选中被高估的 η。
- 约束后的架构消除了灾难（E3 = 0），并且不劣于 R0：
  - 低维 PCA；
  - 在冻结 R0 上的有界残差；
  - 悲观或保守的选择规则；
  - 两段式 regime 模型。

**尚未解决：在真实 Flow 上仍然没有“既安全又有效”的正迁移。**
- 收益只来自 v10_local；base_u 上为 0；能带来收益的规则会伤害 v9_local。
- 安全的规则在必要 controller 上只是回退到源先验。
- round5 的 Stage 0 显示：R3 的 LOO regime 分类准确率为 64.3%，等于多数类基线。也就是说，静态描述并没有编码 controller 的 η regime。

本轮的问题：**剩下的瓶颈是“表示”还是“覆盖”？**
- 若是表示：静态描述捕捉不到决定 regime 的闭环行为。
- 若是覆盖：每个 regime 的 controller 太少，从数据上无法辨识。

检验方法：把 round5 已采集、但从未用于训练的**无标签闭环行为探测（R5）**，放进 round6 修复后的架构里使用。

# 二、必读材料（先读，并亲自核对关键数字）

1. `diagnostics/controller_pathway_fix_v1/`：HANDOFF.md、REPORT.md、results.json、blend_results.json、real_results.json、F1/F2/F3/F4 的代码与 `fits/`、`shared_sparse_fits/`。
2. `diagnostics/controller_flow_blend_v1/`：HANDOFF.md、REPORT.md，以及 R5 相关文件：`R5_FEATURES.npz`、`probes/POSTFLIGHT.json`、`RAW_TRACE_INDEX.json`。
   - R5 共覆盖 26 个 controller（14 个原始 Flow 和 12 个融合 controller）。
   - 每个 controller：16 个 probe state × {η10, η13} × 200 步，记录无终局标签的统计量，共 160 维。
   - 832 次探测中有 831 次有效，1 次数值失败。
3. `diagnostics/controller_loco_real_flow_v1/`：真实 Flow 的 LOCO 标签与划分（7 折），其中必要折为 base_u 和 v10_local。

# 三、设计（在任何新分析之前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 0：数据与复现

- **R5 质量审计：**
  - 找出那一次失败的探测属于哪个 controller、哪个 state、哪个 η；
  - 检查每个 controller 的有效探测比例；
  - 检查统计量的分布，确认没有以 0 填充失败值。
- **预先写定失败探测的处理方式：**
  - 主方案：每个 controller 的统计量只在它自己的有效探测上计算，并另加一维“有效比例”特征。
  - 敏感性方案：所有 controller 都去掉该 probe state 和该 η，再重算。
- **复现 round6：** 重放 round6 中 R0 和 R3-PCA2（F2 + 悲观/保守规则）在真实 LOCO 与融合 LOPO 上的选择，结果必须与原报告一致。

## 阶段 1：regime 可辨识性审计（不训练 Q）

- 分别用 R3、R3-PCA2、R4_action、R5、R5-PCA2、R3-PCA2+R5-PCA2 作为 controller 表示，预测 controller 的经验 regime（其最佳固定 η 的类别）。
  - PCA 只用源 controller 拟合。
- 两种划分：
  - 真实 Flow：在 14 个真实 Flow 上做留一法（LOO）。
  - 融合 controller：按锚点对分组留出，防止同一对内的泄漏。
- 分类器：1-NN，以及 C = 1.0 的 logistic 回归，均不调参。
- 报告：
  - 准确率、与多数类基线（真实 Flow 上为 64.3%）的差、混淆矩阵；
  - 预注册的重点：base_u 与 v10_local 两个 η13 型 controller 在每种表示下被预测成哪个 regime，以及各自的最近邻是谁。

## 阶段 2：修复架构 + 行为表示的迁移检验（主检验）

**测试床：**
- 主：真实 Flow LOCO 的 7 折。
- 次：融合 LOPO 的 4 折，SPARSE 与 DENSE 两种覆盖都做。

**训练设定：** 全部沿用 round6——数据、划分、源验证集、训练预算、3 个 fit seed、JAX x64 = false。

**表示（都在修复架构内使用）：**
- R3-PCA2：round6 的已有结果，作为参照，可直接复用 fit；
- R5-PCA2（主）；
- R5-PCA3（敏感性分析）；
- R3-PCA2 + R5-PCA2。

**架构与规则（超参数全部沿用 round6 的预注册值，不得重新调整）：**
- F2：在冻结 R0 上学习有界残差（c = 1，weight decay = 1e-2），配合两种选择规则：
  - 悲观选择，κ = 1；
  - 保守偏离，δ = 0.10，且 3 个 fit 一致才偏离。
- F4：两段式 regime logistic，阈值 0.6，低于阈值时回退到源先验 η。
- 基线：B0（R0）。

**终点（只用原终点 seed：真实 LOCO 用 seed 4–7，融合用 seed 2–3）：**

| 编号 | 指标 | 判定方式 |
|---|---|---|
| E1 | 在所有留出 controller 上相对 R0 的成功率 | 非劣界 −2pp，按 state 成组的同时 CI |
| E3 | 灾难率：比源先验低 ≥ 10pp 的 controller 所占比例 | 需报告 possible 与确定两种口径 |
| E2 | 在必要 controller 上相对源先验与 R0 的收益 | 真实 Flow 上逐个报告 base_u 与 v10_local，并给出合并的描述性 CI |

**“安全正迁移”的判定：** 同时满足以下四条：
- E1 过门；
- E3 = 0（possible 口径）；
- base_u 与 v10_local 的收益点估计都 ≥ 0；
- 二者合并 E2 的 CI 下界 > 0。

**成本对照（描述性）：**
- R5 每个 controller 需要 32 次 200 步的短探测；
- k = 1 少样本校准需要 32 次完整 H700 episode，且要用成功标签。

报告两者的模拟步数，以及它们在必要 controller 上的效果。少样本校准的数字直接引用已有结果。

# 四、结论映射（预注册，必须按此下结论）

| 观察结果 | 结论 |
|---|---|
| 阶段 1 中 R5 的 LOO regime 准确率明显高于多数类（且正确识别 base_u 与 v10_local），同时阶段 2 中 R5 达到“安全正迁移” | **剩余根因是表示**：决定 regime 的是闭环行为，静态描述捕捉不到。解决路径是“行为探测表示 + 约束架构”。同时写出前瞻确认所需的最小设计（新 controller 与新 state） |
| R5 能识别 regime，但迁移不安全或没有收益 | 瓶颈在“从 regime 到选择”的环节，例如 Q 的校准、选择规则，或 regime 内部的差异 |
| R5 也不能识别 regime，准确率约等于多数类 | 在现有探测预算和 14 个真实 controller 的条件下，regime 无法辨识，**覆盖（controller 太少）是主要瓶颈**。下一步需要构造更多真实 controller（例如用不同 seed 或数据训练新的 Ring Flow），并写出最小规模与成本估计 |
| 融合族与真实 Flow 结论不一致 | 分别报告，并解释原因（例如融合族的 regime 由 λ 连续决定，较容易辨识） |

# 五、硬性规则

- **不得新增 rollout 或探测，不得写入共享 DB。** 只能读取 DB 和已有文件，读取时使用只读或 immutable 模式。
- **文件写入范围：** 只在新目录 `diagnostics/controller_behavior_repr_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **不得用终点 seed 做选择或调参。** 所有超参数和规则都沿用 round6 的预注册值；新增的只有表示本身，以及失败探测的处理方式。
- 所有数据早已被打开过，结论一律标为 DEVELOPMENT。
- unknown 的处理沿用 round6 的配对保守规则。
- **资源：** GPU 通过 Slurm，最多 4 个 shard；CPU 最多 4 核。若预计墙钟时间超过 6 小时，先报告再执行。
- 本任务是纯 Q 实验，按 AGENTS.md 不需要视频。

# 六、交付

在 `diagnostics/controller_behavior_repr_v1/` 中交付：
- PREREGISTRATION.md 及收据；
- 代码；
- R5 质量审计；
- regime 可辨识性结果，含混淆矩阵与 base_u、v10_local 两例的诊断；
- 每个 arm × 折 × seed 的 checkpoint 与预测；
- results.json；
- 图表；
- REPORT.md；
- HANDOFF.md（不依赖本对话即可读懂）。

REPORT 必须做到以下几点：
1. 明确回答：剩余瓶颈是“表示”还是“覆盖”，并按第四节的结论映射给出结论。
2. 把 Q 预测、给定 controller 时的 η 选择、controller 级 regime 识别这三件事分开报告。
3. 写出下一步的最小设计：
   - 若结论是表示：写前瞻确认，需要新 controller 与新 state，并列出 rollout 数；
   - 若结论是覆盖：写构造新真实 controller 的方案与成本。
4. 每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。事实、解释和猜想分开写，不夸大。

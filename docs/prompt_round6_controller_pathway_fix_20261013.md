你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不生成任何新 rollout，只用已有标签和冻结模型重新训练小型 critic，占用资源很少。**

# 一、研究目标与本轮问题

研究总目标：让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在**未见过的 controller** 上选出更好的 η。

本轮要验证的根因假设是 H_method：

> critic 的 controller 通路过参数化。controller 输入有 256–513 维，而不同的训练 controller 只有 14–23 个，且模型没有任何约束：
> - 不向不使用 controller 信息的先验收缩；
> - 选择时不考虑不确定性。
>
> 因此在新 controller 上，模型会在 controller 特征空间里任意外推，用 argmax 选出被高估的 η。结果是“提供更多、更好的 controller 信息反而更差”。

本轮用低成本的方法修复来证实或否定 H_method。如果证实，它同时就给出修复方案。

# 二、事实依据（先读，并亲自核对数字）

## 1. `diagnostics/controller_flow_blend_v1/`（HANDOFF.md、REPORT.md、results.json）

在 η13 型与 η10 型真实 Flow 之间做了速度场融合，共 12 个融合 controller，按 LOPO 划分为 4 折，并有 DENSE / SPARSE 两种源覆盖设置。

- 正式主门：必要 controller 只有 3 个，少于 4 个，判为 INCONCLUSIVE。
- **但逐折结果显示：**
  - R3 和 R_oracle（提供真实坐标 [λ, R3_A, R3_B]）在多个融合 controller 上严重失误，相对源先验低 30–58pp，Q2 MSE 达 0.15–0.24（R0 约为 0.06）。
  - 这种失误也出现在“不需要 controller 信息”的 controller 上。
  - DENSE − SPARSE：R_oracle 为 −26.9pp，CI [−34.2, −19.6]；R3 为 −9.9pp。
  - 学到的 controller 级先验常常选 η1 / η8 / η4，而 η1 和 η8 在所有源 controller 中都很少是好的 η。
  - R0 与 R4_action 基本安全：R4_action 的结果接近 R0。

## 2. `diagnostics/controller_loco_real_flow_v1/`（HANDOFF.md、REPORT.md、results.json）

在 7 个真实 Flow 上做 LOCO：
- 必要折只有 2/7。
- R3 在 fold4 上成功、在 fold1 上失败。
- Stage C 中 R3 相对源先验为 +5.2pp，CI [−0.7, 11.1]，未得到确认。

14 个 Flow 的 regime 分布：η10 有 9 个，η13 有 2 个，η2 有 2 个，η4 有 1 个。

## 3. 可复用的代码与资产

上述两个目录中的：标签 npz、静态特征（R3、R4_action、R_oracle）、`fits/` 下的冻结 checkpoint 与预测、训练代码，以及 P1 的 24 个验证 state 标签。

**训练设置一律沿用 round5：** physical23 + 表示，填充到 513 维，接 64/64/16-head MLP；AdamW；12,000 步；用源验证集 NLL 选择 checkpoint；JAX x64 = false。

# 三、设计（在任何新分析之前，写好 PREREGISTRATION.md 与 UTC/sha256 收据）

## 阶段 0：复现

用冻结 checkpoint 重放 round5 的 R0、R3、R_oracle（DENSE 与 SPARSE 都要），以及 round4 LOCO 的 R0 与 R3。核对 argmax 选择与成功率和原报告一致。如有不一致，先定位原因，再继续。

## 阶段 1：融合族上的方法修复（主测试床：round5 的 4 个 LOPO 折 × DENSE/SPARSE）

所有 arm 保持相同的数据、划分、源验证集和训练预算，只改变下列因素。

**F1 低维 controller 表示（重新训练）：**
- F1-PCA-k：只用源 controller 拟合 controller 级 R3 均值的 PCA，取前 k 维作为 controller 输入。k = 1、2、3 各训练一个 arm。PCA 不得使用目标 controller。
- F1-oracle-low：特权低维坐标，输入为 [λ, A 的 regime one-hot, B 的 regime one-hot]。原始 Flow 使用 λ = 0 和自身 regime。regime 由源 S_train 标签确定。

**F2 向先验收缩（重新训练）：**
- 冻结 R0 的 logits，只训练残差 Δ(s, C, η)：一个小型 MLP，输出为 c·tanh(·)，取 c = 1.0；残差网络使用强 weight decay 1e-2。
- 输入使用 F1-PCA-2；另做一个 arm，输入使用 F1-oracle-low。

**F3 不确定性感知的选择（不重训，作用于 B1、B2、F1、F2 的预测）：**
- F3a 悲观 argmax：选择使 mean_fit(Q) − κ·std_fit(Q) 最大的 η，κ = 1。另报告 κ = 0.5 的结果作为敏感性分析。
- F3b 保守偏离：只有当“预测成功率 − 源先验 η 的预测成功率 ≥ 0.10”，并且 3 个 fit 都同意时，才偏离源先验 η；否则就选源先验 η。

**F4 两段式 regime 模型（重新训练，很小）：**
- 用源 controller 训练 controller 级 logistic 分类器：输入 F1-PCA-2，输出经验 regime（η10 / η13 / η2 / η4）。
- 部署规则：最大预测概率 ≥ 0.6 时，选该 regime 的 η；否则回退到源先验 η。
- 不使用逐 state 残差。前几轮已证明逐 state 选择没有可迁移的增益。

**基线（用冻结 checkpoint 复现）：** B0 = R0，B1 = R3，B2 = R_oracle。

**以上所有超参数（k、c、weight decay、κ、δ = 0.10、0.6 阈值）在预注册中写定，不得根据终点结果调整。**

## 阶段 2：真实 Flow LOCO（第二测试床）

- 在 round4 的 7 个 LOCO 折上，直接应用 F3（无需重训）。
- 只有阶段 1 中满足下面 E1 且 E3 = 0 的 arm，才在这 7 折上重新训练并评价。

## 终点（只用原终点 seed：融合族用 seed 2–3，LOCO 用 seed 4–7）

| 终点 | 定义 | 判定标准 |
|---|---|---|
| E1 不伤害 | 在所有留出 controller 上（融合 12 个；LOCO 7 个）的平均选择成功率，与 R0 比较 | 非劣界 −2pp；用按 state 成组的同时 CI，要求下界 > −2pp |
| E2 收益 | 在必要 controller 上（融合 3 个、LOCO 2 个）相对源先验与 R0 的提升，以及“缩小到 target-oracle 固定 η 的差距比例” | 合并的 CI 只作描述 |
| E3 灾难率 | 选择成功率比源先验低 ≥ 10pp 的留出 controller 所占比例 | — |

## 机制诊断

- **D1 外推距离：** 在每种表示空间中，计算目标 controller 到源 controller 的距离（最近邻距离，以及到凸包的距离），以及它与“相对 R0 的损失”之间的关系。
- **D2 winner's curse：** 比较被选中 η 的预测 Q 与实际成功率（终点 seed），即高估量。
- **D3 头稀疏度：** 统计被选中的 η 头在源标签中的覆盖次数与经验成功率，例如 η1 和 η8 被选中时，它们在源中的情况。

# 四、结论映射（预注册，必须按此给出结论）

| 结果 | 结论 |
|---|---|
| 存在修复 arm 满足 E1 且 E3 = 0，而 B1/B2 有灾难 | **H_method 成立**：方法的归纳偏置是“给 controller 信息反而有害”的根因。同时指出起作用的是哪一部分：降维（F1）、收缩（F2）、悲观选择（F3）还是两段式（F4） |
| 在上一行基础上，必要 controller 上 E2 的合并 CI 下界 > 0 | 该修复同时带来正迁移。列为候选方案，并写出下一步前瞻确认（新 state、新 controller）的最小设计 |
| 所有修复都无法同时满足 E1 和 E3 = 0 | H_method 作为充分解释被否定；根因回到 controller 覆盖不足或表示不足 |
| 修复消除了伤害，但 E2 无收益 | 两部分根因：方法缺陷解释了“伤害”；正迁移仍受 controller 覆盖或表示限制 |

# 五、硬性规则

- **不得新增 rollout，不得写入共享 DB。** 只读 DB 和已有文件。
- **文件写入范围：** 只在新目录 `diagnostics/controller_pathway_fix_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **seed 隔离：** 不得在终点 seed 上做任何选择、调参或校准。
- **证据等级：** 所有数据都已打开过，本轮结论一律为 DEVELOPMENT，不得称为确认。
- **unknown 处理：** 保留 unknown，按原报告的配对保守规则处理。
- **资源：** GPU 通过 Slurm 使用，最多 4 个 shard；CPU 最多 4 核。新拟合约 144 个（阶段 1：6 个重训 arm × 8 个任务 × 3 个 seed），另加阶段 2 中按条件执行的部分。如果实测耗时显示总量会超过 8 小时墙钟时间，先报告再决定，不得自行删减 arm。
- **纯 Q 实验**，按 AGENTS.md 不需要视频。

# 六、交付

在 `diagnostics/controller_pathway_fix_v1/` 中提供：
- PREREGISTRATION.md 及收据
- 代码
- 每个 arm × 折 × seed 的 checkpoint 与预测
- results.json
- 图表：E1 / E2 / E3 对比图、D1–D3 诊断图
- REPORT.md
- HANDOFF.md（不依赖本对话也能读懂）

REPORT.md 必须做到：
1. 阶段 0 的复现结论。
2. 每个 arm 的 E1、E2、E3，以及 D1–D3 结果。
3. 按第四节的结论映射，明确回答两点：
   - “给 controller 信息反而有害”是否由方法的归纳偏置导致；
   - 哪一种修复既安全又有效。
4. 对“零样本 controller 泛化的根本原因”给出当前最强的结论，并写清边界：Ring、固定几何、K16、DEVELOPMENT 证据。
5. 所有假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。事实、解释、猜想分开写，不夸大。

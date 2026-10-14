你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不新增任何 rollout 或探测**，只用已有标签与冻结特征训练和评估 critic。这是纯 Q 实验，不需要视频。

# 一、要回答的问题

研究目标的原始表述是：**让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在未见过的 controller 上零样本选好动作。**

前几轮已在 Ring 上确立（全部可在报告中核对）：
- **根因 H_compress：** 整局固定 η，会把前期效应 E 与后期效应 L 压缩成一个符号。
- **规则 G_E：** 动作菜单为时间表，用 200 步无标签探测 R5 预测 E；Ê < 5pp 选 F10，否则选 S500。这一规则已两次前瞻确认：
  - round12：8 个同 run 兄弟，平均遗憾 1.42pp；
  - round14A：14 个独立 seed 的 controller，平均遗憾 0.61pp，相对 F10 高 29.6pp。
- **round15**（`diagnostics/ring_controller_population_v1/`）：56 个独立 Flow × 32 个新 state × 8 个动作 × Q4，共 57,344 条标签。
  - E、L 的配方 ICC 约为 0.9。
  - L 的排序可学（Spearman 约 0.73–0.84），但对新配方的幅度还学不准（RMSE 比 0.76）。
  - 学切换时刻的规则 G_EL 没有优于 G_E。

但到目前为止，**真正的 critic Q(s, C, a) 从没有在正确的动作空间和足够的数据上训练过。** 早期失败的 critic 都用的是固定 η 的 K16 动作，训练 controller 也只有 14 个。

**本轮问题：**
1. 在时间表动作空间、由探测得到的 controller 表示、56 个训练 controller 这三个条件下，一个通用的 critic 能否学会 controller–Q 关系，并在**未见过的 controller** 上零样本选好动作？
2. 它是否不劣于手工规则 G_E？
3. 逐 state 的选择能否带来额外价值？

# 二、数据（只读；先审计身份、unknown 与对齐，写入预注册）

**训练集：** round15 的 `dataset.jsonl`，以及 R5_FEATURES.npz、R6_FEATURES.npz，加上 32 个 state 的物理特征。
- 56 个 controller；
- 动作：F10、F13、F2、S200、S300、S400、S500、S600；
- seed 0–3。
- 每个 controller 的 R5 / R6 特征按 round15 冻结的预处理计算。

**外部测试集（只用于最终评估，不得用于训练、调参或选 checkpoint）：**

| 测试集 | 目录 | controller | 已标注的动作 |
|---|---|---|---|
| T1 | `diagnostics/ring_independent_flow_confirmation_v2/`（round14A） | 14 个独立 seed controller × 64 个 state | F10、F13、S300、S500、F2、F4 |
| T2 | `diagnostics/controller_universal_eta_schedule_v1/`（round10） | 14 个原 best controller × 64 个 state | F10、F13、Freg、S300、S400、S500 |
| T3 | `diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/`（round12） | 8 个同 run 兄弟 controller × 64 个 state | F10、F13、S300、S500、F2、F4 |

- 每个测试 controller 都已有冻结的 R5 特征；R6 特征只有部分测试集有。
- **R6 规则：** 测试 controller 若没有 R6，只能使用不含 R6 的模型变体，不得为此补跑探测。
- **可选动作：** 在每个测试集中，critic 只能在“它的菜单 ∩ 该测试集已标注的动作”中选择。oracle 取该测试集已标注动作中的最佳者。
- **Freg（T2）：** 只参与 oracle，不作为可选动作。

# 三、模型（全部超参数在预注册中写定，不得根据测试结果调整）

输入为 state 的物理特征（与 round6 相同的 physical23）、controller 表示 z_C，以及动作 one-hot。输出为成功概率。损失为按有效标签计算的 binary NLL，unknown 不参与训练。

| 代号 | 模型 |
|---|---|
| A0 | 不使用 controller 的 critic Q(s, a)，用作基线 |
| A1 | 在冻结的 A0 上加有界残差 Δ(s, z_C, a)：16/16/16 SiLU，输出 1.0·tanh，weight decay 0.01（与 round6 的 F2 相同）。z_C = R5-PCA2 |
| A2 | 同 A1，但 z_C = R5-PCA2 ⊕ R6-Δ。只能在有 R6 的 controller 上训练和评估 |
| A3 | 无约束的对照：64/64/16-head MLP，直接以 [s, z_C, a] 为输入。用来复现 round5 / 6 的“无约束通路有害”是否在新动作空间中仍然存在 |

**训练设置：**
- AdamW，lr 0.0008，12,000 步；
- fit seed 17 / 23 / 41；
- 用训练集内按配方划分的内部验证集，以平均 NLL 选 step；
- JAX x64 = false。

**选择规则：**
- 主规则：逐 state 取 argmax。
- 悲观选择：κ = 1，使用 3 个 fit 的均值 − 标准差。
- controller 级选择：先在评估 state 上平均预测值，再给每个 controller 选一个固定动作。

**参照：**
- G_E 规则（冻结的 Ê 模型）；
- 源平均动作；
- 全程 F10；
- oracle。

# 四、评估与预注册判定

**评估一：round15 内按配方分组的交叉验证（DEVELOPMENT）。** 7 折，每折留出 2 个配方。报告：
- 每个模型 × 规则的平均遗憾与最大遗憾；
- Q 的 NLL 与 Brier 分数。

**评估二：外部零样本测试（主判定）。** 用全部 56 个 controller 训练，模型冻结并留收据之后，才读取 T1–T3 的标签。

判定以 T1（独立训练、未参与任何建模）为主，T2、T3 作为补充报告：

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| Q1 | critic 是否学到了 controller–Q 关系 | A1（或 A2）的平均遗憾比 A0 低，按 state 成簇 bootstrap 的配对 95% CI 上界 < 0 |
| Q2 | 是否不劣于手工规则 | A1 的平均遗憾 ≤ G_E 的平均遗憾 + 0.5pp（非劣），且最大遗憾 ≤ 5pp |
| Q3 | 逐 state 选择的价值 | A1 逐 state argmax 的遗憾 < A1 controller 级选择的遗憾，配对 CI 上界 < 0。否则判定逐 state 选择没有额外价值 |
| Q4 | 无约束通路是否仍然有害（描述性） | 比较 A3 与 A1 的遗憾和灾难数。灾难定义为：比 F10 低 ≥ 10pp 的测试 controller |

另外报告：
- **Q 预测质量：** NLL、Brier、校准。按要求，只作为与选择分开的描述，不能当作选择收益的证据。
- **E、L 的推出值：** 由 critic 推出的 E 与 L（用 critic 预测的 S500 − F10 等）与真实值的相关。

# 五、结论映射

| 观察结果 | 结论 |
|---|---|
| Q1 与 Q2 都通过 | 研究目标的原始表述在 Ring 上实现：一个通用 critic 学会了 controller–Q 关系，能在独立训练的新 controller 上零样本选好动作，而且不劣于手工规则。若 Q3 也通过，则逐 state 选择有额外价值 |
| Q1 不通过，Q2 通过 | critic 能选好动作，但 controller 输入没有带来超出“只看 state”的价值。在时间表动作空间里，controller 依赖已经小到可以忽略 |
| Q2 不通过 | 通用 critic 不如结构化规则 G_E。controller–Q 关系目前更适合用“探测 → E → 规则”的分解形式来表达，报告差距来自哪里 |

# 六、硬性规则

- 不新增 rollout 或探测，不写共享 DB；只读已有文件，读取 DB 时使用只读模式。
- 只在新目录 `diagnostics/ring_controller_critic_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **外部测试集的隔离：**
  - 在模型、规则与全部超参数冻结并留收据之前，不得读取 T1–T3 的成功标签；
  - 先完成审计，审计只核对身份与数量，不计算任何成功率。
- 所有结论标为 DEVELOPMENT。T1 上的结果可以称为“对冻结模型的外部零样本测试”，但 T1 的标签此前已被 round14A 打开过，不得称为全新的前瞻确认。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 七、交付

全部交付物放在 `diagnostics/ring_controller_critic_v1/`：
- PREREGISTRATION.md 及收据；
- 数据审计；
- 代码；
- 每个模型 × fit 的 checkpoint 与预测；
- 冻结收据；
- results.json；
- 图表：各测试集上每个模型 × 规则的遗憾，以及逐 controller 的选择；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：一个通用 critic 能否在独立训练的新 controller 上学会并使用 controller–Q 关系（Q1、Q2），以及逐 state 选择是否有价值（Q3）。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE，事实与解释分开写。

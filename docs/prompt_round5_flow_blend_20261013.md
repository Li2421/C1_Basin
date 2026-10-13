你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。

- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

# 一、研究目标

本轮要回答的问题是：controller 泛化失败，根因到底是“controller 覆盖不足”（可辨识性），还是“controller 表示不足”，还是“方法本身学不会”。研究的总目标是让 critic 学会 Q(s, C, η) 中 controller 的作用，并能在未见过的 controller 上选出更好的 η。

# 二、上一轮事实（`diagnostics/controller_loco_real_flow_v1/`，先读其 HANDOFF.md、REPORT.md、results.json、SOURCE_REGIME_COVERAGE.json）

**1. controller 对 η 决策的作用主要是“regime”。**
14 个真实 Ring Flow 的经验最佳 η 只落在少数几类上：

| regime | Flow 数 |
|---|---|
| η10 | 9 |
| η13 | 2（`base_u_macflow`、`base_u_v10_local_macflow`） |
| η2 | 2 |
| η4 | 1 |

LOCO 中只有 2/7 折需要 controller 信息，这两折的留出 controller 恰好都是 η13 型，而源合并先验是 η10。

**2. 可迁移的部分是 controller 级别的 regime 识别，不是逐 state 选择。** 在各阶段中，“逐 state 选择”的成功率都不高于“模型自身学到的 controller 固定 η”。

**3. 两个必要折里，源中各只有 1 个同 regime 样例，结果正好一成一败：**
- fold 4（留出 v10_local，源中样例为 base_u）：R3 选中 η13，比源先验高 +18.6pp。
- fold 1（留出 base_u，源中样例为 v10_local）：R3 失败。

**4. 几种表示的表现：**
- R4_action 和 R4_openloop（t0 时刻的局部投影响应）几乎不携带 regime 信息，错配对照的差约为 0。
- R3（compact64）携带部分信息：Stage C 中错配对照差为 +7.5pp，CI > 0；但相对源先验的提升为 +5.2pp，CI [−0.7, 11.1]，没有通过主门。

**5. Stage B 没有真正检验“覆盖”。** 新增的 7 个 Flow 全是 η10/η2/η4 型，η13 型源仍然只有 1 个。现有 checkpoint 库中每个训练 run 只有 best.pkl 和 final.pkl，不存在更多 η13 型的真实 controller。

**因此，本轮需要人为构造一个连续的 controller 族，让它跨越 regime 边界，并对“覆盖”和“表示”做析因检验。**
这在 controller 空间里对应之前的 DB 宽度族：在单一共享坐标下，泛化是可以成立的。

# 三、实验设计

## Stage 0：regime 审计（不新增 rollout，必须先做）

只使用已有标签、静态特征和冻结模型，完成以下四项：

1. 给出 14 个 Flow 的 regime 表：最佳 η、它与 η10 和 η13 的成功率差，以及 CI。
2. 在 R3、R4_action、R3+R4 各自的空间中，计算 Flow 之间的距离，并用最近邻和 logistic 做留一法（LOO）预测 regime。与“多数类”基线对比，报告准确率和混淆矩阵。
3. 解释 fold 1 与 fold 4 的不对称：在每个表示空间中，base_u 与 v10_local 各自的最近邻是谁，以及冻结模型学到的 controller 级先验 η 分别是什么。
4. 把必要折的 regret 分解为“controller 级 regime”部分和“逐 state”部分。

Stage 0 只描述现象，不做门判定。

## Stage 1：Flow 融合族 pilot（约 4,600 次新 rollout）

**controller 定义（实现前先写入预注册）。** 对 regime 不同的锚点 A（η13 型）和 B（η10 型），定义融合 controller C(A, B, λ)：
- 在 FF 的 pseudo-time 积分中，基础速度场取凸组合 v_λ(h, s) = (1−λ)·u_A(h, s) + λ·u_B(h, s)。
  - u_X 是 Flow X 在世界动作坐标下的速度场，使用其各自的 observation chart 和归一化（复用 round3/round4 的 `flow_adapter.py`）。
- 初值 h0 = (1−λ)·D_A(ξ) + λ·D_B(ξ)，两者使用同一个 ξ。
- 之后照常接 OrthoFlow3 修正、admissible-field 投影、硬安全层、H700，评价器不变。
- **静态审计（不调用 env.step）：** λ = 0 和 λ = 1 时，执行动作必须与锚点 A、B 逐值一致（最大误差 ≤ 1e-10）。

**锚点对。** A ∈ {`base_u_macflow`, `base_u_v10_local_macflow`}，B ∈ {`base_u_plus_w_v4_wide_macflow`, `base_u_v9_local_macflow`}，共 4 对。内部 λ ∈ {0.25, 0.5, 0.75}，共 12 个融合 controller。

**样本。**
- 复用 round4 的 S_train state，取其前 24 个；锚点在这些 state 上已有标签。
- η 只用 pilot 子集 {η2, η4, η10, η13}，即已知的 4 个 regime。Q4。
- 合计 12 × 24 × 4 × 4 = 4,608 次。
- pilot 只用于定位 regime 边界，不做选择收益的检验。因此删减 η 在这里是允许的，但必须在预注册中写明。

**门（预注册）。** 以下两条同时满足才算通过：
- 至少 3/4 的锚点对，沿 λ 方向“η13 与 η10 中较优者”发生翻转；
- 在翻转两侧，较优 η 的成功率都 ≥ 40%（双方都可胜任，不是全部失败）。

报告每个融合 controller 的 K4 成功率曲线和 regime 判定。

- 门失败：不运行 Stage 2，报告 regime 地图并结束（结论：该融合族不能产生可用的 regime 覆盖）。
- 门通过：进入 Stage 2。

## Stage 2：覆盖 × 表示 析因主实验（约 6 万次新 rollout，上限 65,000）

**标签。**
- S_train：12 个融合 controller × 96 个 S_train state（复用 round4 的 S_train，以便与锚点标签配对）× 每个 state 8 个 η（在 K16 中分层随机分配）× Q4。
- S_test：用 `state_generator.py` 新生成 32 个自然 state，并通过历史哈希去重。12 个融合 controller × 32 × K16 × Q4：
  - seed 0–1 用于固定 η 参照和必要性门；
  - seed 2–3 只用于终点评价。
- 14 个原始 Flow 的已有标签全部复用。
- 如果实测单次 rollout 耗时推算超出上限：按预注册先把 S_train 降到 64 个 state；不得用其他方式缩减。

**折（leave-one-pair-out，LOPO）。** 每次留出 1 个锚点对的 3 个融合 controller 作为测试 controller，共 4 折。

**覆盖因子（主要的析因因子）：**

| 臂 | 源 controller |
|---|---|
| DENSE | 14 个原始 Flow + 其余 3 对的 9 个融合 controller |
| SPARSE | 只用 14 个原始 Flow（锚点都在其中） |

另外报告一个更难的次要划分：留出包含某一个 η13 锚点的全部 2 对（leave-one-anchor-out）。

**表示因子**（网络、数据、预算完全相同，只换 controller 输入）：

| 表示 | 内容 |
|---|---|
| R0 | 不给 controller 信息 |
| R3 | 与 round4 相同 |
| R4_action | 与 round4 相同 |
| R_oracle | [λ, A 的 R3, B 的 R3]：特权信息，即融合成分已知；用来检验在知道真坐标时能否学会 |
| R5_behavior | 无标签的闭环行为探测（定义见下） |

**R5_behavior 的定义：**
- 用 16 个独立的 probe state。这些 state 新生成，并且与 S_train、S_test 都不重叠。
- 对每个 controller，分别在 η10 和 η13 下各跑 200 步（10 秒），截断，不运行到 H700。
- 只记录以下无标签统计量：t = 200 时向目标推进的比例、安全投影处于激活状态的步数比例、最小 agent 间距、平均速度、停滞步数。
- 不使用成功、超时或碰撞等终局标签。
- 每个 controller 共 32 次短 rollout。这需要在模拟器里运行新 controller，部署假设与静态表示不同，必须单独标注；并与“用 k = 1 个目标 state 做少样本校准”（相同的 rollout 量级）对比。

**critic 与训练。** physical23 + 表示 → 16-head MLP。只对有标签的 head 计算 loss。用源验证集的 NLL 选 checkpoint，验证集复用 P1 的 24 个开发 state 和锚点标签。3 个 fit seed。归一化只用源数据。超参数沿用 round4。

**必要性门（每个测试 controller 单独计算）。** 与 round4 相同：
- L = （目标 controller 自身固定 η 的交叉拟合成功率）−（源先验 η 的成功率），只用 seed 0–1。
- 要求 L ≥ 5pp，且 CI 下界 > 0。
- 源先验在每个“覆盖臂 × 折”内分别计算。

**主终点。** 只在必要 controller 上，只用 seed 2–3 评价。比较选择的闭环成功率，与以下三项相比：
- (a) 源先验 η；
- (b) 在 R3 空间中最近的源 controller 的最佳 η；
- (c) R0。

另外报告 matched − mismatched，即把目标 controller 的表示换成其他 controller 的表示后的差。

- 在每个表示内部，按 state 配对，用 max-T 同时置信区间。
- 再单独报告覆盖效应：DENSE − SPARSE 的配对差。

**结论映射（预注册，必须按此给出结论）：**

| 观察结果 | 结论 |
|---|---|
| DENSE 下 R3/R4/R5 有可靠收益，SPARSE 下没有 | 根因是 controller 覆盖不足（可辨识性） |
| R_oracle 在 DENSE 下成功，但 R3/R4 失败 | 根因是表示不足；若此时 R5 成功，则 regime 是闭环行为性质，静态局部描述捕捉不到 |
| DENSE 下 R_oracle 也失败 | 方法本身无法学到 controller→regime 映射，是方法层面的缺陷 |
| 必要 controller 太少（少于 4 个） | INCONCLUSIVE |

# 四、硬性规则

**数据复用与预注册**
- 实验前先检索历史证据，复用兼容的标签；不同执行语义的标签不得混用。融合 controller 必须使用新的 controller UID 和精确的 execution identity。
- 每个阶段在读取新 outcome 之前写预注册（包括门、样本、划分、指标、结论映射），并保存 UTC 时间和 sha256 收据。之后的修改另行记录，不得覆盖原文件。

**rollout 流程**
- 顺序：study design → `python -m shared_rollout_db.plan` preflight → 只提交 genuinely missing 的请求 → worker 写 raw 和 journal → 在 merge 之前登记 DB experiment 行 → 单进程 merge → postflight。
- 数值失败保留为 unknown：不重跑，也不当作失败；做配对比较时要正确处理两侧的 unknown。

**禁止事项**
- 在 Stage 2 的选择检验中不得删减 η，必须使用 K16 并包含 η13。
- 不得在 seed 2–3 上做任何选择、调参或校准。
- 不得把 Q MSE 当作选择收益。
- round4 中已打开的 state 只能当作 DEVELOPMENT 使用。

**资源与环境**
- CPU rollout 最多 24 路并行；GPU 训练通过 Slurm shard 运行。
- 融合 controller 每一步要计算两个 Flow，正式提交前先用约 100 次 rollout 实测耗时，并写入成本门。
- 上限：Stage 1 为 5,000 次，Stage 2 为 65,000 次。超过上限先停下来报告，不得自动扩张。
- 记录并匹配 JAX x64 设置。

**文件与提交**
- 只在新目录 `diagnostics/controller_flow_blend_v1/` 中写文件；不修改、不删除历史资产；不做 git commit。
- 这是纯 Q 实验，按 AGENTS.md 不需要视频。

# 五、交付

**文件（放在 `diagnostics/controller_flow_blend_v1/`）：**
- PREREGISTRATION*.md 及收据；
- 代码、raw、labels、静态特征和 R5 探测数据；
- 每折、每表示、每个 seed 的 checkpoint 和预测；
- results.json、REPORT.md；
- HANDOFF.md，要求不依赖本对话也能读懂。

**REPORT 必须包含：**
1. 每个阶段的状态：完成 / 门未通过 / 未执行及原因。
2. Stage 0 的 regime 审计，以及 fold 1/fold 4 不对称的解释。
3. 融合族的 regime 地图。
4. 覆盖 × 表示的结果表，以及 DENSE − SPARSE 效应。
5. 按上面的结论映射，明确给出 controller 维度上的根因结论，并说明它在多大程度上能解释 round4 的失败。
6. 用 SUPPORTED / REJECTED / INCONCLUSIVE 标注每个假设；事实、解释、猜想分开写；Q 预测与 η 选择分开报告；不夸大结论。

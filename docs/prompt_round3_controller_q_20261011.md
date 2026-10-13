你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。工作根目录：`/home/zhihan/research/Basin_C1`。Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。开始前先读 `AGENTS.md`，并严格遵守。

# 研究目标（所有设计以此为准）

解决 controller 泛化：学会 Q(s, C, η) 中 controller 的作用，使 critic 面对训练中没见过的 controller 时，也能预测 Q，并在该 controller 下选出比固定 η 更好的 η。

- **η**：OrthoFlow3 的三维、episode 级修正参数。
- **controller**：Flow checkpoint、FF/OrthoFlow3、硬安全投影三者组成的完整执行程序。
- **主要终点**：未见 controller 上 η 选择的闭环成功率。Q 误差只作次要指标，不能当作成功依据。

# 必读背景（先读完，再设计）

1. `docs/next_experiments_controller_q_round3_20261011.md`
   本任务的实验选择，顺序为 P0 → P1 → P2 或 P2′ → P3。
2. `diagnostics/ring_state_selection_transfer_v1/` 下的 HANDOFF.md、REPORT.md、results.json。上一轮的结论：
   - Ring 原生 FF 上 Q512 的选择收益已被精确复现；
   - 16-head MLP 约 128 个 state 就能可靠超过固定 η，原标量 MLP 需要 512 个；
   - FF/TT 会改变 state 层的 η 排序；
   - κ 算子族（C1）因为只用了 32 个“容易”的 state，结果饱和；
   - 投影后的 Jacobian 与未投影基底差异很大，而且 η = 0 时各 controller 的执行动作已经不同。

   可复用资产：`c1/runtime.py`、`FLOW_CHECKPOINT_INDEX.json`（列出 14 个 Ring Flow checkpoint）、`EFFECTIVE_BASIS_*`、`C2_STATIC_FEATURES*`、N1 的模型与预测。
3. `diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/` 与 `real_state_expansion/`
   Q512 的协议、512 个训练 state、256 个开发 state。注意：这 256 个 state 已被打开过，在它们上面的结果只能算 DEVELOPMENT。
4. `diagnostics/controller_eta_selection_db_audit_v1/`
   全库 headroom 普查，以及 seed 对齐的方法。
5. `shared_rollout_db/README.md`
   preflight、journal、单进程 merger 的规则。
6. `docs/controller_generalization_audit_20261011.md` §10–§11
   更早的根因审计及其适用范围。

# 任务顺序与决策门

## P0 设计校准（不新增 rollout）

只用原生 Ring v10 已有的数据，评估协议沿用 N1：
- 训练集：512 个 state，K16，Q4；验证集：32 个 state。
- 在 256 个开发 state 上评估：seed 0–7 用来选固定 η 参照，seed 8–15 用来评估。

**P0a 等标签预算下的 η 分配。** 比较以下三种分配：
- 128 个 state × 16 个 η；
- 512 个 state × 每个 state 随机 4 个 η；
- 256 个 state × 每个 state 8 个 η。

用 16-head 网络，只对有标签的 head 计 loss。每种分配取 3 个随机子集，每个子集训练 3 次。报告 top-1 成功率相对“16 个固定 η 中最强者”的差，以及同时置信区间（simultaneous CI）。

**P0b checkpoint 规则。** 对比“验证集 NLL”与“验证集上的 selected-success”（可选加上 Q-margin）。

在冻结 P2 设计之前，把 P0 的结论写成预注册补充，并保存 sha256 收据。

## P1 真实 Flow controller 普查 pilot（约 2 万次新 rollout）

**兼容性审计。** 对 `FLOW_CHECKPOINT_INDEX.json` 中的 14 个 Ring checkpoint 逐一检查：
- 能否在“自由 t0、FF、K16、H700、相同安全层与评价器”下运行；
- 按各自的 manifest 适配观测格式（local frame 或 global frame）；
- 不修改任何源文件，需要的 adapter 写在新目录里；
- 登记精确的 execution identity 和 JAX x64 设置。

不兼容的 checkpoint 记录原因，不要强行运行。

**静态表示。** 对每个兼容的 controller，用静态查询计算以下表示（禁止调用 env.step）：
- R3：compact64；
- R4_action：η = 0 时的投影执行动作，加上投影后的 Jacobian ∂a_exec/∂η。在 t0 和预先固定的少量 probe state 上计算。

**样本。**
- 从 256 个开发 state 中分层抽 24 个。必须包含 adaptive-demand 组，不能只取容易的 state。抽样规则要在看到任何新 outcome 之前冻结。
- 使用 K16（包含 η13），每格 Q4，按 seed 做 2/2 交叉拟合。

**测量。**
- 每个 controller：固定 η 的成功率、最佳固定 η、按 state 选 η 的 headroom。
- 交叉 controller 遗憾矩阵，分两层：
  - 全局层：用 A 的最佳固定 η 去执行 B；
  - state 层：在每个 state 上用 A 的最佳 η 去执行 B。
- controller×η 与 state×controller×η 的无偏能量。

**预注册的门槛。** 同时满足以下两条才算通过：
- 至少 8 个 controller 的固定 η 成功率落在 [40%, 95%]；
- 交叉 controller 遗憾（全局层或 state 层）的中位数 ≥ 5pp，且同时 CI 下界 > 0。

通过则进入 P2；不通过则进入 P2′。

## P2 留出真实 controller 的主实验（约 5–6 万次新 rollout，只在 P1 通过后进行）

**划分（在看到任何 P2 outcome 之前冻结）。**
- 7–9 个 Flow 用于训练，留出 3 个。
- 留出的 3 个中，至少 2 个位于训练 controller 在 R4 空间中的凸包内，1 个位于凸包外。

**训练数据。**
- 每个训练 controller 用 256 个新的自然 state，由原生开发生成器、新的随机序列生成。
- η 的分配方式按 P0a 的结论，Q4。
- 原生 v10 的 512 个 state 作为补充源。

**测试数据。**
- 每个留出 controller 用 64 个全新 state × K16 × Q8。
- seed 0–3 用来选固定 η 参照，seed 4–7 用来评估。

**Critic。**
- 结构：physical23 + controller 表示 → 16-head MLP。
- checkpoint 规则按 P0b 的结论。
- 每种设置 3 个 fit seed；归一化只用源训练数据。

**表示对照**（网络、数据、预算完全相同，只换 controller 表示）：

| 名称 | 内容 |
|---|---|
| R0 | 不提供 controller 信息 |
| R1 | one-hot；留出的 controller 输入全零 |
| R3 | compact64 |
| R4_action | η = 0 投影动作 + 投影 Jacobian |
| R3+R4_action | 两者拼接 |

另外加两个对照：
- 错配对照：把 controller 描述换成另一个 controller 的描述；
- 训练 controller 数量消融：取 {3, 5, 全部}。

**终点。** 留出 controller 上的 top-1 闭环成功率，与以下参照比较：
- (a) 所有训练 controller 合并后的最佳固定 η；
- (b) 在 R4 空间中最近的训练 controller 的最佳 η；
- (c) R0 critic；
- (d) 留出 controller 自身的 hindsight 最佳固定 η，只作上界参照。

同时报告 matched − mismatched（正确描述减去错配描述）。CI 用按 state 成组的 bootstrap，至少 2000 次。

Q MSE 和 controller×η 模式相关也要报告，但不能作为成功依据。

## P2′（P1 不通过时）

- 用 placement 族（FF/TT/TF/FT）加 κ 族，组成结构不同的 controller 集合。
- 做同样的留出检验，例如：用 FF、TT 和 κ 训练，留出 TF 和 FT。
- state 的选取规则和规模与 P2 相同。不能再用 32 个容易的 state。

## P3（P2 失败时的诊断；不新增或只少量新增 rollout）

- 失败发生在 Q 水平，还是发生在 η 排序上？
- 只用 t0 特征的表示，与加入短时 open-loop 探测 state 的表示，结果有何差别？
- 用“训练 controller 数”画学习曲线，判断问题是否只是 controller 覆盖不够。

# 硬性规则

**历史证据与标签复用**
- 开始任何实验前，先检索历史证据，复用兼容的标签（见 AGENTS.md）。
- execution 语义不同的标签不得混用。

**预注册**
- 每个阶段在读取新 outcome 之前写预注册，内容包括门槛、样本、划分和指标；保存 UTC 时间和 sha256 收据。
- 之后的任何修改单独记录，不得覆盖原文件。

**rollout 流程**
- 新 rollout 按以下顺序：study design → `python -m shared_rollout_db.plan` preflight → 只提交 genuinely missing 的部分 → worker 写 raw 和 journal → 单进程 merge → postflight。
- 数值失败保留为 unknown：不重跑，也不当作失败。
- 不使用处于冲突隔离状态的记录。

**防止人为制造收益或夸大结论**
- 不用 32 个容易的 state 或已饱和的面板做选择检验。
- 不删减 η 来制造收益：K16 必须包含 η13。
- 不把 Q 相关或 Q MSE 当作选择迁移的证据。
- 256 个开发 state 上的结果只算 DEVELOPMENT。

**文件、视频与资源**
- 只在新目录 `diagnostics/controller_q_transfer_round3_v1/` 中写文件；不修改、不删除历史资产；不做 git commit。
- 这是纯 Q 实验，按 AGENTS.md 不需要视频。
- CPU rollout 最多 24 个并行（每个 worker 1 个 CPU）；GPU 训练通过 Slurm shard 运行。上一轮 Ring rollout 平均约 2.1 秒一次。
- P1 新 rollout 上限 25,000，P2 上限 70,000。超出上限先停下来报告，不自动扩张。
- 记录并匹配每个模型和运行时的 JAX x64 设置。上一轮曾因精度设置不一致导致重放结果不同。

# 交付

在 `diagnostics/controller_q_transfer_round3_v1/` 中交付：
- PREREGISTRATION*.md 及其收据；
- 代码、raw、labels；
- 模型 checkpoint 与逐 seed 的预测；
- results.json、REPORT.md；
- HANDOFF.md，要求不依赖本对话也能读懂。

REPORT 必须做到：
1. 逐阶段给出状态：完成、门槛未通过，或未执行及原因。
2. 分开判断三件事：Q 预测、给定 controller 时的 η 选择、场景或 controller 层面的 prior 校准。
3. 明确回答：在未见的真实 controller 上，哪种 controller 表示（如果有）能带来可靠的选择收益；训练 controller 的数量如何影响结果。
4. 每个假设标注 SUPPORTED、REJECTED 或 INCONCLUSIVE。事实、机制解释和猜想分开写，不夸大。

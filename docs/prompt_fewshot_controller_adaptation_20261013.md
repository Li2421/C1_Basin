你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。工作根目录：`/home/zhihan/research/Basin_C1`；Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。开始前先读 `AGENTS.md`，并严格遵守。

**本任务不生成任何新 rollout。** 另一个 agent 正在用 CPU 跑 Flow 融合实验（`diagnostics/controller_flow_blend_v1/`），请勿与之争抢资源：CPU 最多同时用 4 核；GPU 只能通过 Slurm 申请，最多 2 个 shard。

# 目标

研究总目标是让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在**新 controller** 上选好 η。零样本迁移目前还没有得到确认。本任务回答它的实用互补问题：

**面对一个新的真实 Flow controller，最少需要多少条有标签的目标 rollout，才能把 η 选到接近该 controller 自身最优固定 η 的水平？利用源 controller 的知识，能把这个数量降低多少？**

这决定了 controller 泛化的一条可部署路线是否成立：零样本做不到时，用少量目标标签做快速适应。

# 背景（先读）

**`diagnostics/controller_loco_real_flow_v1/`：** HANDOFF.md、REPORT.md、results.json、LABELS.npz、protocol.json、SOURCE_REGIME_COVERAGE.json、`fits/`（冻结的 checkpoint 与预测）、`controller_prior/`。

关键事实：
- 14 个真实 Ring Flow 的最佳 η 只有少数几种（“regime”）：η10 占 9 个，η13 占 2 个（base_u、v10_local），η2 占 2 个，η4 占 1 个。
- 只有 2/7 个 LOCO 折需要 controller 信息。
- 能迁移的主要是 controller 级的 regime，而不是逐 state 的选择。
- 已有 1/2/4 个目标 state 的粗略校准结果：fold4 只用 1 个 state 就恢复到约 90%；fold1 噪声很大。但这些校准只用了预先固定的前 k 个 state，没有重复抽样。

**`diagnostics/controller_q_transfer_round3_v1/`：** P1 的 14 个 Flow × 24 个开发 state × K16 × Q4 标签。

# 可用数据（全部复用，必须先审计其身份与 unknown）

- **14 个 Flow 的 S_train：** 128 个 state，每个 state 8 个 η，Q4。其中 7 个来自 Stage A，另 7 个来自 Stage B。
- **7 个原始 Flow 的 S_test：** 64 个 state，K16，Q8。其中 seed 4–7 是评估用 seed。
- **Stage C：** 2 个 Flow × 32 个 state × K16 × Q8。
- **P1：** 14 个 Flow × 24 个 state × K16 × Q4。

# 设计（在任何新分析之前写 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 1. 模拟采集

“目标标签”只能从**已有标签**中抽取，用来模拟预算为 B 时的采集过程。

- 预算 B（目标 rollout 条数）∈ {0, 8, 16, 32, 64, 128, 256}。
- 每个预算重复 200 次随机抽样，随机种子固定。
- 目标标签只能从目标 controller 的 S_train（以及 P1 的 24 个 state）中抽取。
- 评估只使用目标 S_test 的 seed 4–7，两者必须严格分离。

## 2. 采集策略（每种都是一种预先写定的规则）

| 代号 | 策略 |
|---|---|
| U | 在 state × η 上均匀随机抽取 |
| S | 先抽 state，再对该 state 用完整 K16（不足的 η 不能伪造；若 S_train 中该 state 只有 8 个 η，则只在可用 η 上进行，并在报告中注明） |
| A | 自适应：逐步淘汰候选 η。从源先验最高的若干 η 开始，按已采标签的 Beta 后验分配下一条标签。只能用已存在的标签进行模拟；某格标签用尽时记为“无法采集” |

## 3. 适应方法

| 代号 | 方法 | 说明 |
|---|---|---|
| P0 | 源先验 | 不用目标标签，即 B = 0 时的参照 |
| E | 只用目标标签 | 选出经验最优的固定 η，并统一规定平局处理规则 |
| H | regime 混合先验 | 用 13 个源 controller 在 K16 上的成功率曲线构成先验（经验 Bayes 或混合 Beta），再用目标标签做后验更新，选出后验期望最大的 η。这是“利用源知识”的主方法 |
| C | 冻结 critic + 目标 offset | 用 LOCO 已冻结的 R0、R3 模型，加上目标标签校准的 offset（按 state 预测 η） |
| C-FT | 冻结 critic 的少量微调 | 只微调输出层；步数固定，不在测试集上调参。只在 GPU 上运行，规模很小 |

## 4. 终点与参照

- **主终点：** 在目标 S_test 的 seed 4–7 上，计算每种“方法 × 策略 × 预算 B”的闭环成功率，以 unknown 下界计。
- **参照：** 目标 controller 自身最优的固定 η（hindsight，用 seed 0–3 选出），以及源先验。
- **样本效率指标：** B* = 达到“距目标自身最优固定 η 不超过 2pp”所需的最小预算，并给出 bootstrap CI。分别对必要折（fold1、fold4）和非必要折报告。
- **源知识的价值：** H 相对 E 节省的预算（B* 之差），以及每个 B 下的成功率配对差。
- **逐 state 价值：** C 与 C-FT 相对 E/H 是否额外带来收益，即逐 state 选择在少样本下是否有用。
- **CI：** 按 state 做成组 bootstrap，同时纳入 200 次抽样的变异；多重比较用同时 CI 控制。

## 5. 额外分析（描述性）

- 对 14 个 Flow 做留一法：只用 P1 的 24 个 state 作为评估集，估计 H 在全部 14 个 controller 上的 B* 分布（包括另外 7 个不在成功率窗口内的 Flow）。在报告中注明这些 state 属于 DEVELOPMENT。
- 平局与 unknown 的敏感性分析。

# 硬性规则

- 不生成 rollout，不写入共享 DB；只读 DB 和已有文件。
- 不修改、不删除任何历史文件；只在新目录 `diagnostics/controller_fewshot_adaptation_v1/` 中写入；不做 git commit。
- 任何选择与调参都不得使用 S_test 的 seed 4–7。
- 已打开过的 state 一律标为 DEVELOPMENT，不得把本结果称为新的确认。
- 记录并匹配 JAX x64 设置（LOCO 中为 false）。
- 资源上限：CPU 最多 4 核；GPU 通过 Slurm 申请，最多 2 个 shard；总墙钟时间建议不超过 6 小时。

# 交付

在 `diagnostics/controller_fewshot_adaptation_v1/` 中提供：

- PREREGISTRATION.md 及收据
- 代码
- results.json
- 学习曲线图（成功率 vs B；每种方法 × 策略；必要折与非必要折分开）
- REPORT.md
- HANDOFF.md（无需本对话即可读懂）

REPORT 必须明确回答：

1. 新 controller 达到“接近其自身最优固定 η”需要多少条目标 rollout；在必要折上与非必要折上分别是多少。
2. 源 controller 知识（H）能节省多少目标标签。
3. 少样本条件下，逐 state 选择（C、C-FT）是否带来超出 controller 级固定 η 的收益。
4. 对研究的含义：controller 泛化是否可以用“少样本快速适应”作为可部署路线，以及零样本研究还值得解决什么。

假设一律标注 SUPPORTED / REJECTED / INCONCLUSIVE；事实与解释分开写，不夸大。

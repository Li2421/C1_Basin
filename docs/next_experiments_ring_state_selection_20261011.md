# 下一轮实验选择：Ring 上已确认的 state 选择收益能否迁移、需要多少数据（2026-10-11）

> **目标对齐修订（同日，优先阅读 §3）。** 本对话的目标是解决 controller 泛化：让 controller 与 Q(s, C, η) 之间的关系可以被学会，使模型对未见过的 controller 也能预测 Q 并选出 η。
> - 原 N1–N5 主要研究 state 维度和几何维度，只能作为辅助。
> - §3 给出以 controller 为中心的 C0–C3，执行顺序以 §3 为准。

本文件只做实验选择，供实现 agent 执行；选择者不运行实验。

上游证据：
- `diagnostics/controller_eta_selection_db_audit_v1/`（HANDOFF、REPORT、results.json）
- `diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/`（verdict.json、protocol.json、frozen_predictions.npz）

## 0. 已确立的事实（选择依据）

**F1 收益真实存在。** Ring FF（固定 Flow、FF、H700、包含 η13 的 K16）上，按 state 选 η 有真实的部署收益。
- 256 个新 state 上：冻结的 Q512 为 93.54%，所有 16 个固定 η 中最好的 η13 为 88.33%，同时 CI 为 +5.21pp [2.64, 7.94]。
- 数据库 E0 面板（312×13×Q16）：+5.79pp [3.35, 8.81]。

**F2 学习它很吃数据。**
- 在同一个几何上，只用 32 个 state 训练的 Q32 只有 85.07%，比固定 η 还差；比 Q512 低 8.46pp [6.14, 11.08]。
- 旧 Ring 32-state 面板出现训练记忆：TRAIN 相关 0.955，VAL 相关 −0.06。
- 旧 Ring FF 的 nuisance 诊断：state 相关的 context 会过拟合，并削弱 η 先验。

**F3 饱和只是局部现象。** Toy FF、DB、Gap1 的若干面板被固定 η 饱和；Toy TT 有收益（+14pp，但 N 小）。

**F4 跨场景迁移还没有干净的检验。**
- 四场景 LOSO 失败，但场景、几何与 controller 混杂在一起。
- 两个单因素宽度族都没有选择 headroom。
- **Ring 的 state×η 规律能否跨几何或跨 controller 迁移，目前完全没有检验。**

**F5 场景级 prior 可以靠少样本校准。** 1–4 个目标 state 就能修复 η regime；机制代理在零标签下的优势不稳定。

由此，核心未知量排在第一位的是：在单个几何内，学会这种 state 选择需要多少数据，以及这种需求是方法本身低效造成的，还是问题本身就难。答案决定任何迁移实验的规模与设计。

## 1. 选定的实验（按顺序）

### N1 Ring 学习曲线与学习器效率（零 rollout，第一优先）

**问题。** 要比固定 η 有收益，需要多少训练 state（n*）？项目 critic 是否比简单学习器更吃数据？

**做法。**
- 先复核 Q512 的训练数据和完整冻结流程，包括 state 来源、K、seed 数、输入与 checkpoint 规则（见 `real_state_expansion/`、`heldout_adaptive_selection_v1/code_snapshots`）。复现之后再改变 n。
- n 取 {32, 64, 128, 256, 512}，用嵌套子集；每个 n 取 3 个随机子集，每个子集训练 3 个 seed。
- 比较的学习器（输入都是同样的 state 和 η 信息）：
  - (a) 原冻结 critic 流程；
  - (b) 只用 η 的 prior；
  - (c) 物理 state 上的 kNN 或核回归，加 η one-hot；
  - (d) 原始 state 上的 MLP。

**评估。**
- 在 256 个新 state 上进行，使用原 discovery 与 evaluation 的 8/8 seed 划分。
- 指标：top-1 闭环成功率，减去“16 个固定 η 中最好者”的同时 CI；再报告 matched − mismatched，以及与 Q512 的差距。
- 注意：这 256 个 state 已经被打开过一次，所以 N1 只能算开发级证据；任何新的正面结论都需要新的确认集。

**判读。**
- 给出 n*，即收益的 CI 下界首次 > 0 时的 n。
- 若 (c)/(d) 在明显更少的 state 下就达到收益，说明项目 critic 存在方法层面的样本效率缺陷。这是一个明确的“方法缺陷”候选。
- 若所有学习器都需要约 256 个以上的 state，则说明 Ring 的 state×η 映射本身复杂。后续迁移实验必须按此估算数据量，或改用迁移与微调的设计（见 N5）。

### N2 Ring 偏好的物理机制（零 rollout）

**问题。** 哪些可计算的初始物理量决定了一个 state 偏好 η1/η4/η10 还是 η13？

**做法。**
- 数据：608 个开发 state 加 256 个新 state（Q16），以及数据库中的 312×13 面板。
- 标签：用 discovery seed 定义每个 state 的偏好组。
- 特征：选少量可解释的几何特征，在拟合之前固定，例如：
  - 各 agent 相对中心障碍的角度间隔与顺序；
  - 到障碍的径向距离；
  - 相向配对的相位；
  - 初始 clearance 与通道（环带）宽度之比。
- 用交叉验证评估稀疏 logistic 或决策树在 evaluation seed 上的预测力，并与随机特征和置换基线比较。

**价值。**
- 把“controller 机制”和“泛化”连接起来，得到一个可以跨几何定义的候选机制坐标。
- 为 N4 提供可证伪的预测：改变几何之后，偏好组的边界应当如何移动。

### N3 同一几何内的 state 空间外推（零 rollout）

**问题。** 在 state 空间分布偏移下，学到的 state 选择能否外推？

**做法。**
- 按 N2 的特征（若 N2 失败，则用预先固定的径向/角度特征）把 state 切成两个区域。
- 在一个区域上训练，在另一个区域上测试，并与同样 n 的随机划分对比。

**判读。**
- 如果区域外推本身就失败，说明跨几何迁移的障碍首先是方法在 state 分布外选择失效，而不是“几何”。

### N4 Ring 几何 headroom pilot（约 1.2 万次新 rollout；在 N1–N3 之后执行）

**问题。** 换成别的 Ring 几何之后，state 选择收益是否仍然存在？迁移检验只有在 headroom 存在时才有意义。

**做法。**
- 几何族：只改变 `obstacle_radius`（例如 0.75 和 1.05，原生值为 0.9），`outer_radius` 等其他参数不变。
- 先做几何有效性与 Flow 语义检查：初态合法；固定 η13 下基本能完成任务；没有通道封死。
- 每个几何使用 24 个新 state，由原 native generator 生成，并按几何做有效性过滤；过滤规则要预先写定，不能看 outcome。
- 使用原 K16（包含 η13），Q16，8/8 交叉拟合。
- 所有 rollout 遵守 shared_rollout_db 的 preflight/merge 流程。

**门槛（每个几何）。** 沿用 E0 的规则：
- H_xfit ≥ 5pp 且 CI 下界 > 0；
- U − I ≥ 5pp；
- 覆盖数 ≥ 2；
- 候选中包含 η13。

同时检验 N2 预测的偏好边界移动方向（预注册）。

**判读。**
- 某几何没有 headroom：说明 state 选择的价值本身依赖几何。这也是一个重要结论，可以与 F5 的场景 prior 线合并。
- 有 headroom：进入 N5。

### N5 迁移效率实验（重；仅在 N1 给出 n*、N4 至少有一个几何合格后执行）

**设计原则。** F2 表明从零学习需要几百个 state，所以不要再沿用“每个几何 16 个 state 的 MANY/FEW”设计，那样注定样本不足。改问实际更有用的问题：**源几何的训练能否减少目标几何所需的标签？**

**做法。**
- 源：原生 r = 0.9，已有 512 个 state。
- 目标：N4 中合格的几何。
- 目标训练 state 数取 {0, 16, 32, 64, 128}（Q4 标签）。
- 每个目标 n 下比较三种方式：从源开始微调、只在目标上从零训练、只用场景 prior 校准。
- 目标测试集为 64 个新 state × K16 × Q16。
- 估算成本：每个目标几何约 2.5 万次 rollout。

**主要终点。**
- top-1 成功率相对固定 η 的收益（同时 CI）；
- matched − mismatched；
- 以及“达到同样收益所需的目标 state 数”这一比值。

### N6 可选：跨 controller（零 rollout，先盘点）

- 先查数据库中，是否有同一批 Ring state 在其他 controller 下也有 K16、Q16 的面板。
- 如果有，测试 state×η 偏好在不同 controller 之间是否一致，以及一个 controller 上学到的选择器迁移到另一个 controller 后的收益。
- 如果没有，只记录这个缺口，暂不补数据。

## 2. 明确不做

- 不再运行“每个几何 16 个 state”的从零 MANY/FEW 迁移实验（被 F2 否定）。
- 不在饱和面板（Toy FF、DB、Gap1）上测试选择迁移。
- 不扩大或删减 η 面板来制造 headroom。
- 不把 Q 相关当作选择证据。
- 所有新 rollout 必须先 preflight，再 merge；这些都是纯 Q 实验，不需要视频。

## 3. 目标对齐修订：以 controller 为中心的实验（执行顺序以本节为准）

**目标。** 学到 Q(s, C, η) 中 controller 的作用：给定一个训练中没见过的 controller C*，critic 要能
- 预测 Q；
- 在 C* 下选出 η，并且优于“C* 的固定 η”和“沿用别的 controller 的 η”。

已有机制线索：endpoint-matched FF 算子只通过“修正怎样被执行”起作用。执行后的修正是 R(M)·δ(s, η)，等价于把 η 的三个基底换成 R·B_k(s)。所以 controller 与 η 是乘法式耦合的。这给出一个可检验的 controller 表示：**有效修正基底**，即 controller 在当前 state 下对 η 各方向的执行响应。

| 原编号 | 在新顺序中的位置 | 理由 |
|---|---|---|
| N1 学习曲线 | 保留为第一步 | 任何 Q(s, C, η) 学习都需要先知道数据规模 |
| N6 跨 controller | 提升为 C0 | 直接对应目标 |
| N2、N3 | 辅助 | 研究 state 机制与外推 |
| N4、N5 几何 | 后置 | 留给“跨场景的 controller 迁移”阶段 |

### C0 Ring 上 controller×η 的决策结构盘点（零 rollout；与 N1 并行）

**问题。** 在 Ring 这个有 state 选择收益的场景里，换 controller 会不会改变最优 η？如果不会，对 η 选择而言 controller 泛化只剩 Q 水平的校准问题；如果会，controller 信息就是选择所必需的。

**数据。** 从共享数据库中找“同一批 Ring state、同一组 η”下有 ≥ 2 个 controller 的面板，候选包括：原生 FF、TT、其他 Flow checkpoint（v7、v8、v11）、endpoint-matched 算子族。

**指标。**
- 每个 controller 的最佳固定 η。
- 交叉 controller 遗憾：B 在“自己的最佳固定 η”下的成功率，减去 B 在“A 的最佳固定 η”下的成功率。
- controller×η 与 state×controller×η 的无偏能量。
- 按 controller 分别计算的 state 选择收益。
- 用 seed 交叉拟合，并对 state 做 bootstrap。

**产出。** 判断“controller 改变最优 η”在 Ring 上是否成立；如果数据缺失，列出缺口。

### C1 Ring 连续 controller 族 pilot（约 1–2 万次 rollout；C0 之后）

**设计。** 在 Ring 原生 Flow 端点上构造一个单参数的 FF 算子族，相当于 controller 空间里的“宽度族”。例如修正映射的旋转量：
- M(κ) = 2.5·(I + κJ)，κ ∈ {−1, −0.5, 0, 0.5, 1}，对应执行修正的旋转角约 −45° 到 +45°；
- 同时以原生 FF controller 作为锚点。
其余全部固定：几何、horizon、安全层、评价器、噪声。

**样本。**
- 从已有的 512/256 Ring state 中取 32 个，便于和 Q512 的已知结论对照。
- η 取 K16 或决策相关的 K8，必须包含 η1、η4、η10、η13。
- 每格 Q8（4/4 划分）。

**门槛（预注册）。**
- (a) 最优固定 η 随 κ 改变：交叉 controller 遗憾 ≥ 5pp，且 CI 下界 > 0；
- (b) 至少一半 κ 值上仍有 state 选择收益，标准同 E0。

**判读。**
- (a) 不成立：在这个算子族里 controller 只影响 Q 水平。C2 只评估 Q 预测与校准，不评估选择。
- (a) 成立：进入 C2。

### C2 留出 controller 的泛化与 controller 表示（主实验；仅在 C1 通过后执行）

**划分。**
- 在部分 κ 上训练，在留出的 κ 上测试，内插与外推分别报告。
- 源数据：原生 controller 的 512 个 state（已有）。
- 每个训练 κ 的 state 数由 N1 的 n* 决定，建议 ≥ 128。
- 测试：每个留出 κ 用 64 个新 state，Q16。

**critic 中 controller 输入的对照（网络、数据、预算都相同，只改输入）：**

| 编号 | controller 输入 | 作用 |
|---|---|---|
| R0 | 不给 controller 信息 | 下界 |
| R1 | one-hot 身份 | 对照：无法迁移到新 controller |
| R2 | 真参数 κ | 参数族内的上界参照 |
| R3 | 项目现有静态描述子（compact64 或 H20） | 现有做法 |
| R4 | 有效修正基底：t0 及少量探测 state 上，对 η 三个方向做静态有限差分，得到执行动作响应 ∂a_exec/∂η_k | 机制级表示 |
| — | 把 R3 或 R4 换成错配 controller 的描述 | 检验模型是否真的在用 controller 信息 |

R4 不需要 rollout，可以用到任何黑盒 controller 上。

**主要终点（在留出 controller 上计算）：**
- 给定 C* 选 η 的闭环成功率，相对于：
  - C* 自己的固定 η（hindsight 参照）；
  - 最近训练 controller 的最佳 η；
  - 所有训练 controller 合并后的最佳 η。
- controller 描述的 matched − mismatched。

**次要终点：** 在留出 controller 上的 Q MSE、controller×η 模式相关。

**判读。**
- R4 > R3 且优于各固定 η 参照：机制级 controller 表示使 controller–Q 关系可以迁移。这是本目标的正面突破。
- R2 成功但 R3、R4 都失败：问题在 controller 表示本身。
- R2 也失败：在这个规模下，controller×state×η 关系不可学。再结合 N1 判断，是样本效率问题还是方法问题。

**估算成本。** 训练 4 个 κ × 128 个 state × 16 η × 4 seed ≈ 3.3 万次；测试 3 个 κ × 64 个 state × 16 η × 8–16 seed ≈ 2.5–4.9 万次。分批执行，并在 C1 之后重新评估。

### C3 真实的未见 controller：留出 Flow checkpoint（C2 成功后执行）

- 用 Ring 的多个 Flow checkpoint（v7、v8、v10、v11 等）作为 controller。
- 在其中几个上训练，留出一个测试；使用 C2 中胜出的表示，同样的终点与对照。
- 这是从“参数化算子族”走向“任意黑盒 controller”的必要一步。
- 之后，“跨场景的 controller 迁移”再与 N4、N5（几何）结合。

### 执行顺序

1. N1 与 C0（都是零 rollout）。
2. C1 pilot。
3. C2。
4. C3。

N2、N3 作为解释性辅助，按需执行；N4、N5 后置。

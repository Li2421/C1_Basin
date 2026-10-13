# 第三轮实验选择：在“条件允许”的面板上检验 controller–Q 关系能否迁移到未见 controller（2026-10-11）

本文件只做实验选择，供实现 agent 执行。

**目标（不变）：** 学会 Q(s, C, η) 中 controller 的作用，使 critic 对未见过的 controller 也能预测 Q 并选出 η。

**上游结果：** `diagnostics/ring_state_selection_transfer_v1/`（REPORT、HANDOFF）。

## 0. 上一轮对目标的推进

**controller 与 Q 的关系确实存在，而且不只是 Q 的整体水平：**
- **FF 与 TT 对比：** 物理 state、几何、η、seed 都相同，只换 placement。
  - Q 平均差约 14pp；
  - 最佳固定 η 都是 η13，固定 η 的交叉遗憾为 0；
  - 但有 15/18 个 state 在独立 seed 半分中出现严格的 η 排序反转，state×controller×η 交互能量的 CI 为正。
  - 结论：即使全局最佳 η 不变，controller 也会改变“哪个 state 该用哪个 η”。
- **真实 Flow（v9 与 v7，旧协议）：** 最佳固定 η 不同，交叉损失分别为 22.9pp 和 9.2pp。

**单个 controller 内的学习尺度（N1）：**

| 学习器 | 最少需要的训练 state | 备注 |
|---|---|---|
| 原标量 MLP | 512 | — |
| 16-head MLP | 128 | 训练 512 个 state 时成功率 94.2% |
| kNN | 64 | 收益只有约 0.85pp |

此外，checkpoint 的选择规则会明显改变选择结果。

**有效修正基底：**
- 未投影的解析基底与真实投影后的 Jacobian 相差 23%–94%。
- η = 0 时，各算子的执行动作已经相差 0.33–0.36。
- 因此 controller 表示必须包含“η0 下的投影动作”和“投影后的 Jacobian”。R4_action 降低了 Q MSE，但错配对照的差异 CI 跨 0。

## 1. 上一轮 C1/C2 没能回答目标的原因（设计缺陷，必须修正）

| 缺陷 | 具体情况 | 后果 |
|---|---|---|
| 测试面板太容易 | C1 用的 32 个历史 state，在原生 FF 下固定 η 成功率 95.3%（总体为 88.3%）；fallback 的测试 state 上固定 η 达到 100% | 选择 headroom 被人为压没 |
| 每个 controller 的 state 太少 | fallback 每个 controller 只有 16 个训练 state，而 N1 表明单个 controller 至少需要 128 个 | 学不到 state×η 结构 |
| controller 数太少 | 只有 3 个训练 κ | 与跨场景的教训相同：可辨识性取决于“环境数”与“同时变化的因素数”之比，3 个 controller 无法确定 controller 空间里的规律 |
| 合成 κ 族的决策效应弱 | 最优 η 基本不随 κ 改变 | 不适合作为检验 controller 泛化的主面板 |

## 2. 选定的实验

### P0 设计校准（零 rollout，先做）

只用已有的原生 Ring 数据：512 个训练 state，K16，Q4；在 256 个开发 state 上评估。

**P0a 稀疏 η 标注是否有效。** 标签总预算相同时，比较三种标注方式：
- 128 个 state × 16 个 η；
- 512 个 state × 每个 state 随机 4 个 η；
- 256 个 state × 8 个 η。

16-head 网络只对有标签的输出头计算 loss。若稀疏标注不比满 η 标注差，正式实验就按“更多 state、每个 state 少量 η”分配标签，成本约降到 1/4。

**P0b checkpoint 规则。** 对比 VAL NLL 与“VAL 选择成功率”两种 checkpoint 规则（可以叠加 Q-margin 正则），选定后冻结，供 P2 使用。

两个结果都在 P2 之前冻结。

### P1 真实 Flow controller 普查 pilot（约 2 万次 rollout）

**为什么用真实 Flow。** 本目标真正面对的 controller 就是 Flow checkpoint。旧证据显示，真实 Flow 会大幅改变最优 η，而合成 κ 族几乎不会。`FLOW_CHECKPOINT_INDEX.json` 列出了 14 个 Ring checkpoint（v2–v6、v10_local 及 wide/targeted 等变体）。

**兼容性检查（先做，零或极少 rollout）：**
- 每个 checkpoint 能在“自由 t0 + FF + K16 + H700 + 同一安全层与评价器”下运行；
- 观测格式正确（local frame 还是 global frame，按各自 manifest 适配）；
- 写入精确的 execution identity；
- 静态计算 R3（compact64）与 R4_action（η0 投影动作，以及在 t0 和少量探测 state 上的投影 Jacobian）。

**设计。**
- 所有兼容的 checkpoint，各用 24 个自然开发 state。
- 这些 state 从 256 个开发 state 中分层抽取：必须包含 adaptive-demand 组，不能只取“容易”的 state。
- K16（包含 η13），每格 Q4，2/2 交叉拟合。

**测量。**
- 每个 controller 的固定 η 成功率、最佳固定 η、state headroom。
- **交叉 controller 遗憾矩阵：**
  - 全局层：用 A 的最佳固定 η 去执行 B；
  - state 层：用 A 每个 state 上的最佳 η 去执行 B。

**门槛（预注册）：**
- 至少 8 个“可胜任且不饱和”的 controller：固定 η 成功率在 [40%, 95%] 之间；
- 交叉 controller 遗憾（全局或 state 层）的中位数 ≥ 5pp，且同时 CI 下界 > 0。

**判读。**
- 通过：controller 改变 η 决策，而且有足够多的真实 controller 可以构成训练集与留出集，进入 P2。
- 不通过：把 P1 的数据作为 controller 层面的描述性证据保留下来，转向 P2′。

### P2 留出真实 controller 的泛化主实验（约 5–6 万次 rollout；P1 通过后执行）

**划分（在看任何结果之前，按 checkpoint 年代或表示空间位置固定）：**
- 用约 7–9 个 Flow 训练，留出 3 个。
- 留出的 3 个中，至少 2 个位于训练 controller 在 R4 表示空间的凸包内（内插），1 个位于凸包外（外推）。

**训练数据：**
- 每个训练 controller 使用 256 个新的自然 state，η 分配按 P0a 的结论；
- 原生 v10 已有的 512 个 state 作为补充源；
- 标签 Q4。

**测试数据：**
- 每个留出 controller 使用 64 个全新 state，K16，Q8；
- 用前 4 个 seed 选出固定 η 参照，后 4 个 seed 评估。

**critic：**
- 16-head 网络，物理 state + controller 表示，checkpoint 规则按 P0b。
- 只改变 controller 输入：

| 编号 | controller 输入 |
|---|---|
| R0 | 不给 controller 信息 |
| R1 | 身份 one-hot（留出 controller 给零向量） |
| R3 | compact64 |
| R4_action | η0 投影动作 + 投影 Jacobian（含探测 state） |
| R3 + R4_action | 两者合并 |

另做两个对照：
- 错配：把 R3 或 R4 换成另一个 controller 的描述；
- 控制规模的消融：训练 controller 数取 {3, 5, 全部}，直接测量“controller 数量”对可辨识性的影响。

**主要终点：留出 controller 上的 η 选择闭环成功率。** 与以下参照比较：
- (a) 训练 controller 合并后的最佳固定 η（零样本可用）；
- (b) 在表示空间里最近的训练 controller 的最佳 η；
- (c) R0 critic；
- (d) 留出 controller 自己的 hindsight 最佳固定 η，只作上界参照。

同时报告 matched − mismatched。所有 CI 以 state 为单位做 bootstrap。

**判读。**

| 结果 | 含义 |
|---|---|
| R3 或 R4 优于 (a)、(b)、(c)，且 matched − mismatched > 0 | controller–Q 关系可以迁移到未见的真实 controller，目标的核心突破 |
| 只在训练 controller 数多时成功 | controller 泛化的瓶颈是 controller 覆盖，可辨识性问题落在 controller 维度 |
| R1 = R0 ≈ R3 ≈ R4，都不如 (a) | 这些表示没有承载决策相关的信息，进入 P3 诊断 |
| Q MSE 改善但选择没有改善 | 分开报告，不当作成功 |

**估算成本。**
- 训练：8 个 controller × 256 个 state × 4 个 η × Q4 ≈ 3.3 万次；
- 测试：3 个 controller × 64 个 state × 16 个 η × Q8 ≈ 2.5 万次；
- 合计约 34 CPU 小时，在 22 CPU 上约 1.5 小时。

### P2′ 备选：P1 门失败时

- 用 placement 族（FF、TT、TF、FT）加上 κ 族，组成一个“结构不同”的 controller 集合。C0 已证明 FF/TT 之间存在 state 层排序反转。
- 以同样方式做留出 placement 的检验（例如在 FF、TT、κ 上训练，留出 TF/FT）。
- state 选取规则和样本量与 P2 相同；不再使用 32 个容易的 state。

### P3 失败时的诊断（零或少量 rollout，按需执行）

- 失败发生在 Q 水平，还是在 η 排序上（对 controller×η 的交互能量做分解）？
- 表示中缺的是“未来会遇到的约束”吗？对比只用 t0 的特征与加入短时 open-loop 探测 state 的特征。
- 用 controller 数的学习曲线判断，是否只是 controller 覆盖不够。

## 3. 明确不做

- 不再用 32 个容易 state 或已饱和的测试 state 来检验选择。
- 不再用每个 controller 只有 16 个 state 的训练集去否定可学性。
- 不靠删减 η 来制造收益：K16 保留 η13。
- 不把 Q MSE 的改善当作选择迁移。
- 几何阶段（N4/N5）继续后置，等 controller 维度有结论后，再做“跨场景的 controller 迁移”。
- 新 rollout 一律先 preflight 再 merge；这是纯 Q 实验，不需要视频。

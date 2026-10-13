# 预注册：controller × 几何 可辨识性的最小验证实验（2026-10-11）

本文件在任何新 outcome 产生之前写定，sha256 冻结进 `protocol_pilot.json`。pilot 只允许按 §3 的信号规则决定宽度范围，不得修改本文件中的任何假设、指标、学习器或判定标准。

## 1. 被检验的说法

来源：`docs/controller_generalization_audit_20261011.md` §10。

- **说法 A（几何依赖）**：controller 对 Q 的作用不是 controller 自身的属性，而是 controller 与几何在闭环中耦合的结果；它随几何改变形式，甚至改变符号。
- **说法 B（可辨识性）**：跨几何迁移的失败来自“几何层面的样本不足 / 不可辨识”，而不是 controller 信息缺失、单个几何内的数据不足或学习器能力不足。具体是两点：
  - 只要训练几何覆盖了几何变化，同一类学习器就能迁移到留出几何；
  - 把同样的标签预算集中在少数几何上（更多 state、更多 seed）不能替代几何多样性。

## 2. 设计（只改变一个几何参数）

- **场景与几何参数**：Double-Bottleneck，变化 `bottleneck_width`。合法范围 (0.3433, 1.64) 由环境自己的 `Config.__post_init__` 校验；不修改任何源文件。
- **固定不变的部分**：4 个 agent、动力学、dt、horizon 850、max_speed、半径、墙厚、两个 chamber 与 staging 区、目标点、deadlock 监测、硬安全投影、FF 积分、OrthoFlow3 修正基底、评价器、噪声律（DB Flow checkpoint 的 act_scale/act_mean）、未来噪声 key（`key_for(state, seed, t)`，在各宽度间为同一组 common random numbers）。
- **Controller**：5 个 endpoint-matched goal-reference 算子（isotropic、anisotropic、rotate±、coupled）。它们与四场景研究中使用的解析代码完全相同（`controller_response_families_v1/runtime.py`）；controller 身份对学习器**精确已知**。
- **η**：原 K4 面板，索引 0/5/10/15。
- **State**：存档的 DB 自然初态（`controller_four_scene_mechanism_v1/protocol.json`），在所有宽度上使用**同一组物理初态**。pilot 用 train study_index 0–7；main 用 train study_index 16–31（共 16 个，outcome-blind，与 pilot 不重叠）。
- **复现门槛（已通过）**：在原宽度 0.50，本 runner 与存档 raw 记录逐字节一致（`reproduction.json`）。

## 3. Pilot 与宽度网格规则（预先写定）

- **候选宽度**：0.36, 0.42, …, 1.62（步长 0.06，共 22 个）。
- **Pilot 设计**：宽度取 0.36, 0.54, 0.72, 0.90, 1.08, 1.26, 1.44, 1.62；8 个 state × 5 C × 4 η × seed0，共 1280 次。
- **有信息宽度**：在该 pilot 宽度上，20 个 (C, η) 单元格（每格是 8 个 state 的均值）中，最大值与最小值之差 ≥ 0.25。
- **Main 区间**：[a, b] = [最小有信息宽度 − 0.18, 最大有信息宽度 + 0.18]，再与 [0.36, 1.62] 取交集。
- **Main 网格**：在 [a, b] 上取 `np.linspace(a, b, 14)`，吸附到最近的候选宽度，去重后排序为 g_0..g_{n-1}。
- **宽度分工**：
  - test 宽度 = { g_round(k(n−1)/5) : k = 1..4 }，共 4 个内部点；
  - pool = 其余宽度；
  - FEW = { pool[0], pool[len//2], pool[−1] }；
  - MANY = 全部 pool。
- **无信号时**：若没有任何有信息宽度，则终止并报告“该几何族无信号”，不换族重试，以免挑选结果。
- **Seed 与预算**：pool 宽度用 seed 0, 1；FEW 宽度另加 seed 2, 3；test 宽度用 seed 0, 1, 2, 3。

## 4. 检验 A：几何依赖（全部 main 宽度，seed 0/1）

**统计量：**
- **算子效应模式**：E_w[C, η] = mean_s Q(w, s, C, η) − mean_C(·)，共 20 维。
- **宽度内信度**：r_w = corr(E_w(seed0), E_w(seed1))。r_w ≥ 0.5 的宽度称为“有信息宽度”。
- **跨宽度相关**：R[w, w′] = ½ [corr(E_w(s0), E_w′(s1)) + corr(E_w(s1), E_w′(s0))]。
- **rotate 效应**：Δ_rot(w) = mean_{s,η}[ ½(Q_rot+ + Q_rot−) − Q_iso ]，CI 由 state bootstrap 得到。

**预测：**
- **A1**：存在一对有信息宽度，其 R ≤ 0.3；同时存在相邻的有信息宽度，其 R ≥ 0.7（结构性变化，而非噪声）。
- **A2**：Δ_rot 在某些宽度显著为负、在另一些宽度显著为正，且两侧 95% CI 都不含 0（符号反转）。如果只有一侧显著，记为“部分支持（出现或消失）”。

**证伪 F1**：所有有信息宽度对的 R ≥ 0.7，且 Δ_rot 符号恒定。这表示在该几何族中 controller 效应不依赖几何，说法 A 不成立。

## 5. 检验 B：可辨识性（不新增 rollout，只用 main 标签）

**学习器（固定，不调参）：**
- 输入 = [w, 初始位置(8), 目标(8), 初速度(8), C one-hot(5), η(3)]；连续输入用训练集的均值和标准差标准化。
- MLP 3×128 SiLU，sigmoid 输出；逐 rollout 的 BCE；AdamW(lr 1e-3, wd 1e-4)，batch 256，训练 3000 步。
- 初始化 seed 取 17/23/41/53/71，对 5 个模型的概率取平均作为 ensemble。
- numerical_failure 的 rollout 不进入训练和评估，但会报告数量。
- 不设 early stopping；任何选择都不使用 test 宽度。

**训练集（同一个学习器）：**
- **MANY**：全部 pool 宽度。第 i 个 pool 宽度使用 state {(5i + k) mod 16 : k = 0..4}，seed 0/1。
- **FEW**：3 个 FEW 宽度 × 16 个 state，seed 0/1。预算与 MANY 大致相同。
- **FEW×2**：同样 3 个宽度 × 16 个 state，seed 0–3。预算是 FEW 的 2 倍，检验“环境内更多数据”能否替代几何多样性。
- **ALL-POOL**（上界参照）：全部 pool 宽度 × 16 个 state，seed 0/1。
- **EXTRAP-A**：只用 < t_1 的 pool 宽度（16 个 state，seed 0/1），在 {t_2, t_3} 上评估。
- **EXTRAP-B**：只用 > t_2 的 pool 宽度（16 个 state，seed 0/1），在 {t_0, t_1} 上评估。

两者都是“LOSO 的缩影”：controller 身份精确、state 精确、几何参数精确，只是几何被外推。

**指标（只在 test 宽度上计算）：**
- **B1 Brier skill**：1 − Brier(model) / Brier(prior)。prior 为训练集中各 (C, η) 的均值，对应四场景研究里的 η/controller prior。另报告最近宽度查表基线。
- **B2 算子效应模式相关**：在每个 test 宽度上计算 corr(Ê_w, E_w)，再在 test 宽度上取平均。同时报告去掉 C 主效应后的纯 C×η 版本，以及由 seed 半分信度给出的上限。
- **B3 选择**：对每个 (w, s)，选出 argmax 预测值对应的 (C, η)，用 seed 2/3 的标签计算实际成功率。对照项：
  - 交叉拟合 oracle（用 seed 0/1 选，用 seed 2/3 评）；
  - prior 固定（取训练集中均值最高的 (C, η)）；
  - 随机。
  另报告 B3′：给定 C 时只选 η。
- **CI**：对 16 个物理 state 做配对 bootstrap，2000 次。

**预测：**
- **B-P1**：MANY 的 B2 ≥ 0.6，且 B1 > 0。
- **B-P2**：MANY − FEW 在 B2 上 ≥ 0.15，且 CI 下限 > 0。
- **B-P3**：(FEW×2 − FEW) ≤ ½ (MANY − FEW)（按 B2 计）。
- **B-P4**：EXTRAP-A 与 EXTRAP-B 在各自的外推 test 宽度上 B2 ≤ 0.3，且 B1 ≤ 0。

**证伪：**
- **F2**：FEW×2 ≥ MANY（B2 与 B3 的配对 CI 都包含或高于 0），表示环境内数据才是瓶颈。
- **F3**：MANY 的 B2 ≤ 0.3 且 B1 ≤ 0，表示即使有几何多样性也学不出关系，根因判断不完整。
- **F4**：EXTRAP 的 B2 ≥ 0.7 且 B1 > 0.3，表示关系跨 regime 不变，说法 A 不成立。

**预先写定的解读：**
- FEW 与 MANY 都好：在平滑的几何族中，3 个张成范围的几何已经足够，需要把说法修正为“四场景缺少共享的几何连续轴”，而不是“数量少”本身。
- 只有 A 成立而 B 不成立（或反之）：分别报告，不合并成一个结论。

## 6. 范围

- 这是 DEVELOPMENT 级别的机制验证：单一几何族、一个 controller 家族、K4 面板。
- 不声称四场景 LOSO 已修复，也不声称 Flow controller 已被测试。
- 这是纯 Q 数据/评估实验，按 AGENTS.md 不要求视频。
- rollout 遵守 `shared_rollout_db` 的 preflight 与 journal 规则。

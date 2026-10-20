你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，以及总结报告 `docs/controller_generalization_ring_summary_20261014.md`（重点是第 2、13、14 节）。严格遵守 AGENTS.md。

**本任务不新增任何 rollout、探测或模型训练，只对已有的前瞻标签与冻结预测做统一的预注册统计分析。** 这是纯 Q 数据分析，不需要视频。

# 一、目的

研究的核心现象是：把整局固定的 η，换成“前期一个 η、第 t 步换成 η10”的时间表，在同一场景内能大幅提高成功率，并消除或大大削弱对 controller 的依赖；但它不能消除场景之间的冲突。

**现状：** 总结报告第 14 节中的多数对比，是设计者用临时脚本算出的描述性数字，没有 CI，也没有预注册。

**本轮任务：** 用一套统一的、预先写定的代码与统计方法，对全部已有数据重新计算这些对比，加上安全性检查、理论假设检查和切换时刻敏感性分析。产出作为论文的主表与主图。

# 二、数据（只读；逐项审计身份、标签数与 unknown）

| 代号 | 目录（均在 `diagnostics/` 下） | 内容 |
|---|---|---|
| R9 | `controller_eta_switch_time_v1` | Ring，4 个 controller，64 个 state，双向切换 t = 100–500，Q4；全程基线来自 round4 的标签 |
| R10 | `controller_universal_eta_schedule_v1` | Ring，14 个原 Flow，64 个新 state；臂为 F10、F13、Freg、S300、S400、S500；Q4 |
| R12 | `controller_new_checkpoint_confirmation_v1/fp32_n8` | Ring，8 个同 run 兄弟，64 个新 state；臂为 F10、F13、S300、S500、F2、F4；Q4；含冻结的 G_E 选择 |
| R14A | `ring_independent_flow_confirmation_v2` | Ring，14 个独立 seed，64 个新 state；臂同 R12；Q4；含冻结的 G_E 选择 |
| R15 | `ring_controller_population_v1` | Ring，56 个新 seed，32 个 state；臂为 F10、F13、F2、S200、S300、S400、S500、S600；Q4 |
| R16 | `ring_controller_critic_v1` | 在 R15 上训练的 critic，对 T1 / T2 / T3（即 R14A / R10 / R12）的冻结预测 |
| R17 | `ring_critic_new_recipe_confirmation_v1` | Ring，17 个新配方（N2 13 个、N1 4 个），48 个 state；8 个臂；Q4；A0、A3、A3_all 的冻结预测 |
| R18 | `ring_eta2_gap_v1` | Ring，TRAIN-new 18 个（32 个 state）与 TEST 18 个（48 个 state）；8 个臂；Q4；C0–C3 与 A0 的冻结预测 |
| R13 | `cross_scene_eta_schedule_v1` | 四场景 × 3 个 Flow 算子 × 40 个 state × 7 个臂 × Q4 |
| R19b | `four_controller_axis_v2` | Four，25 个 controller，32 个 state；固定 3 个臂与 5 个时间表臂；Q2 |
| R20 | `cross_scene_probe_selection_v1` | Toy、DB 共 6 个有效变体 × 3 个算子 × 40 个 state × 7 个臂 × Q2 |

**数据重建要求：**
- 将每一份数据统一重建为 controller × state × arm × seed 的标签数组：成功记 1，失败记 0，unknown 单独标记。来源为各轮的 LABELS 文件、数据集文件，或共享 DB / raw 记录（只读）。
- **重建门（必须先通过）：** 用重建出的标签，重新计算各轮报告中的主要数字，必须完全一致。例如：
  - R10：S500 − oracle = +26.09pp；
  - R12：G_E 平均遗憾 1.416pp；
  - R14A：0.614pp；
  - R17：A3 平均遗憾 1.70pp；
  - R19b：S15→10@786 为 1597/1597；
  - R13：X2 的平均遗憾 68.13 / 44.27pp。

  任何一项不一致，先定位原因并报告，不得继续。

# 三、预注册的分析（先写 PREREGISTRATION.md，附 UTC 时间与 sha256 收据，再计算任何新数字）

**统一的统计约定：**
- 成功率默认把 unknown 当失败（下界），同时报告把 unknown 当成功的上界。
- CI 采用分层 bootstrap，5000 次：先在 controller（或场景、变体）层重抽，再在 state 层成簇重抽。种子写入预注册。
- 比较一律配对（同一 controller、同一 state、同一 seed）。

**场景默认时间表（预先写定）：**

| 场景 | 默认时间表 | 依据 |
|---|---|---|
| Ring | S500 | round10 的预注册 |
| Four | S15→10@786 | round19b |
| DB | S15→10@5H/7 | round20 的源臂 |
| Toy | S13→10@3H/7 | round13 |

## A. 时间表与固定 η（oracle 层面）

对每个数据集、每个 controller，计算：
- 最佳固定臂（在该数据集可用的固定臂中取最好）；
- 最佳时间表臂；
- 两者之差。

报告：
- 均值、中位数与 CI；
- 时间表更好（≥ 1pp）、固定更好（≥ 1pp）、持平的 controller 数量。

**另外比较三种单一策略：**
- 全选 F10（旧做法）；
- 每个 controller 用各自的最佳固定臂（这需要知道 controller）；
- 全选该场景的默认时间表（不需要 controller 信息）。

报告三者的成功率，以及两两之间的配对差与 CI。核心检验对象是：“不看 controller 的默认时间表”减去“知道 controller 的最佳固定臂”。

## B. 零样本选择：用与不用时间表

遗憾一律以完整菜单（含时间表）的 oracle 为参照，同时报告相对“固定菜单 oracle”的遗憾。

- **源平均选择：** 在数据集内按 controller 留一（同一 run 的 best 与 final 一起留出），用源 controller 的平均成功率选臂。比较只用固定菜单与用完整菜单两种情形。
- **冻结的 critic：** 分别把选择限制在固定臂，以及允许全部臂：
  - R16：T1–T3 上的 A0、A3；
  - R17：A0、A3、A3_all；
  - R18：C0–C3、A0。
- **冻结的 G_E 规则：** R12、R14A，相对 F10 的收益。
- **跨场景：** 用 R13 的 LOSO 与 R20 的数据，比较固定菜单与完整菜单。

## C. 安全性与数值稳定性（逐臂，比较固定臂与时间表臂）

- 碰撞率（预期为 0，必须核实）；
- 最小两两距离与最小离墙距离的分布：中位数、5% 分位、最小值；
- 数值失败率（numerical_failure）；
- **切换前后的局部检查：** 凡有轨迹记录的数据（例如 R9、R10 的 traces），比较两个窗口的最小两两距离：时间表臂在 [t − 20, t + 60] 步内的值，以及同一 state、同一 seed 的固定臂在同一窗口内的值。如果轨迹的记录间隔太粗而无法检验，就如实说明。

## D. 理论假设检查（阶段排序不变性；报告第 14.4 节）

- **支配率：** 对每个时间表对 (η_a → η10)，统计满足 S ≥ max(F_a, F10) − 1pp 的 controller 比例。
- **排序一致性：** 统计 E = S − F10 ≥ 0 且 L = S − F_a ≥ 0 的 controller 比例，即前期 η_a 与后期 η10 的排序是否一致。同时对比 N = F_a − F10 的符号分布：预期 N 的符号在 controller 之间不一致，而 E、L 的符号一致。
- **跨场景：** 用 R13 与 R20，按场景和变体报告同一时刻的“前期 η13 对 η15”的比较（S13→10@t 对 S15→10@t），检验 Toy 族与 DB 族的排序是否相反。
- **切换曲线：** 用 R9 的双向切换曲线，描述成功率随切换时刻变化的形状（单调、平台或非单调）。

## E. 切换时刻的敏感性

用 R15、R17、R18 的 S200–S600 网格，计算：
- 每个 controller 的最佳切换步；
- 每个固定切换步相对该 controller 最佳时间表的遗憾分布；
- 平台宽度：遗憾 ≤ 2pp 的切换步范围，覆盖了多少比例的 controller。

## F. 论文主表

对每个 Ring 数据集，按顺序列出以下各级策略的成功率，以及每一级相对上一级的增量（附 CI）：
1. 全选 F10；
2. 每个 controller 的最佳固定臂；
3. 全选 S500；
4. 冻结的 critic 或 G_E（加时间表菜单）；
5. 完整 oracle。

对 Four 与跨场景，给出相应的简表。

# 四、硬性规则

- **不新增 rollout、探测或训练，不写共享 DB。** 读取 DB 一律使用只读或 immutable 模式。
- **文件写入范围：** 只在新目录 `diagnostics/schedule_phenomenon_consolidated_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **预注册：** 所有分析定义、默认时间表、阈值（1pp、2pp、5000 次 bootstrap）都在 PREREGISTRATION.md 中写定，不得根据结果调整。
- **证据等级：** 这些数据大多已被打开过，本轮属于对前瞻采集数据的统一事后分析。报告中不得称之为新的前瞻确认，但要逐项注明每个数据集的标签是否为前瞻采集。
- **unknown：** 保留并报告上下界，不重跑。timeout 不能直接称为 deadlock。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 五、交付

全部交付物放在 `diagnostics/schedule_phenomenon_consolidated_v1/`：
- PREREGISTRATION.md 及收据；
- 数据重建代码与重建门报告；
- 每个分析（A–F）的代码、CSV 表与 results.json；
- 图表（建议 PNG + PDF，版式适合论文）：
  - 各数据集的阶梯图（F10 → 最佳固定 → S500 → critic → oracle）；
  - “时间表 − 固定”的分布图；
  - 切换时刻敏感性曲线；
  - 各臂的安全指标；
  - E、L、N 的符号一致性图；
- REPORT.md：第一段用三句话概括，即时间表的总体收益（附 CI）、对 controller 依赖的影响，以及安全性是否受影响；
- HANDOFF.md（不依赖本对话也能读懂）。

每个结论都要标明数据集、样本量、证据等级与 CI。事实与解释分开写。

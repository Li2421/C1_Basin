你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

# 零、源码保护（最高优先级，先于本文件其他所有要求）

用户明确要求：**不得改动现有源码。** 本轮的全部新代码与输出，只能写在新目录 `diagnostics/gain_attribution_v1/` 中。

**1. 只读，不得修改、删除、移动、重命名或重新格式化：**
- Basin_C1 中所有已有文件，包括 `new_benchmark_common/`、`ring_exchange/`、`shared_control/`、`shared_rollout_db/`（代码只读；DB 的写入方式见第六节）、`bottleneck_family/`、`toy_giveway/`、`double_bottleneck/`、`four_way_intersection/`、`docs/`、`AGENTS.md`，以及 `diagnostics/` 下其他所有子目录；
- PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`（不在 git 中）；
- MAC-Flow 官方仓库 `/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/`，它带有用户尚未提交的改动；
- Python 环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/`：不安装、不升级任何包。缺少依赖时，在新目录内另建环境，或停下报告。

**2. 复用已有代码的方式：**
- 只读 import；
- 或复制到新目录后再修改，并记录源文件路径与 sha256；
- 需要不同行为时，在新目录中写子类或新函数。不得在运行时替换已有模块的函数或全局状态来改变它的行为；唯一的例外是按协议显式恢复 JAX 精度。
- 有些历史模块在 import 时会改环境变量或往自己的目录写文件，例如 round28 的 `core.py` 会把 TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME 指向自己的目录。这类模块不要直接 import，按第三节的规则复制。

**3. git：** 两个仓库都只允许只读命令（status、diff、log、rev-parse、ls-files）。严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。

**4. 完整性审计（交付必需）：** 开始与结束时各记录一次：
- Basin_C1 的 `git status --porcelain`，以及 `git diff` 输出的 sha256；
- MACFlow_Official 的 HEAD、`git status --porcelain`，以及 `git diff` 输出的 sha256；
- PoC 目录 `poc/` 下全部 .py 文件的 sha256 清单；
- 本轮 import 或复制的每个已有源文件的 sha256。

除新目录外，结束时的记录必须与开始时完全一致。不一致就在 REPORT.md 第一段写明，并列出改动。

**5. 被挡住时停下报告。** 如果已有代码缺少需要的接口，或者只有改源码才能继续，就停下报告，不要绕过。

# 一、要回答的问题：泛化的收益从哪里来？

开始前先读：
- `AGENTS.md`、`shared_rollout_db/README.md`；
- 总结报告 `docs/controller_generalization_ring_summary_20261014.md`（全文，重点是第 0、2、14–17 节）；
- round26 的报告与代码：`diagnostics/method_benchmark_v1/`。

**背景：** 我们的方法由三部分组成：
- **修正方向 B：** δ = η_goal·B_goal + η_⊥·B_⊥ + η_rel·B_rel。修正加在经过安全投影的 controller 动作上（Flow 为 FF 位置，其他类 controller 为 TT 位置）。
- **时间表 T：** 前期用 η13，第 t 步换成 η10，例如 S500。
- **条件化学习 L：** critic 以 state 物理特征和 controller 的无标签探测特征（R5-PCA2）为输入，零样本选动作（A3）。

各泛化轴上已观察到的收益：

| 轴 | 已观察到的收益 |
|---|---|
| Ring Flow（FF） | S500 比每个 controller 的最佳固定 η 高约 27pp；A3 平均遗憾 1.8pp，一律 S500 为 3.4pp（round26） |
| 其他类 controller（TT） | A3 比只加安全层高 23pp（MPC）、27pp（势场）；时间表约为 0；照搬 η10 比不加修正还低 41–65pp（round27、round29） |
| 分散式安全层 | 时间表 +44pp；一律 S500 的遗憾 0.11pp（round28） |
| 跨场景 | 时间表无法统一；探测规则失败（round13、round20、round23） |

这些收益混合了三个来源：B 的三个方向、时间表、条件化学习。条件化学习又分两种：按 controller 条件化（C）和按 state 条件化（S）。

**本轮问题：**
1. **B：** 修正的收益由 B_goal、B_⊥、B_rel 中的哪一个承担？H_compress 中的前期收益 E（= S500 − F10）与后期伤害 L（= S500 − F13）分别由哪个分量承担？在不同 controller、不同类别之间，是否是同一个分量？
2. **T 与 L：** 各泛化轴上的收益，有多少来自时间表、多少来自按 controller 条件化、多少来自按 state 条件化？分别给出潜在收益（oracle）与已实现收益（冻结的 critic）。
3. **交互：** 时间表是否替代了 controller 条件化？也就是说，有了时间表之后，controller 信息是否变得不那么重要？

本轮结果用于反推论文的叙事与理论。报告只给证据，不写论文。

# 二、第一部分：离线归因（不新增 rollout，不训练新模型）

**数据（只读；DB 一律用只读或 immutable 模式）：** 按用户的泛化框架组织，全部结果都按下表的轴与子项分别报告。源数据为 round15（`ring_controller_population_v1`，56 个 controller）；Ring Flow 的测试集直接复用 round26 的标签重建与菜单定义。

| 泛化轴 | 子项 | 数据 | 可算的格子 |
|---|---|---|---|
| 1. State | 已见场景与 controller、新 state | T2：round10 的 14 个原 controller 在 round15 的训练集中，而 round10 的 64 个 state 不在其中。另附 round2 的单 controller 逐 state 选择结果作为背景 | 全部 8 格 |
| 2. Controller | 新 checkpoint | T3（round12，同 run 兄弟） | 全部 8 格 |
| | 新 seed | T1（round14A） | 全部 8 格 |
| | 新架构 | T4（round17 N2）、T5（round18 TEST） | 全部 8 格 |
| | 新类型 | round27 与 round29（MPC、势场）。round29 的记录在它目录的 raw 文件中，可能尚未并入全局 DB | 全部 8 格；另含 η = 0 与 nominal only |
| | 附加：分散式执行 | round28；round30（teacher–student）若在本轮分析时已完成就加入，否则不等待 | 只有 oracle 格与不看 controller 的格；缺少冻结 critic 的输入时，学习格记为 N/A |
| 3. Cross-Scene | LOSO：完全排除目标场景 | round13、round20、round23 | 只有 oracle 格，描述性 |
| 4. Cross-Factor | 几何 | round20 的宽度族（Toy、DB）与 Ring 障碍半径数据 | 只有 oracle 格，描述性；controller 本身失败的几何如实标注 |
| | 机器人数量 | 无可用的泛化数据（`datasets/gap1_eta_scaling_*` 未完成） | 记为缺口，不计算 |
| | 时间预算（horizon） | 无 | 记为缺口，不计算 |

**三个因子与 8 个格子：**
- **T：** 菜单是否包含时间表臂。否：固定菜单（各数据集按 round26 的定义，如 {F10, F13, F2}）；是：完整菜单。
- **C：** 是否按 controller 选择。
- **S：** 是否按 state 选择。

| 格子 | 含义 | 学习版（冻结模型，不重新训练） |
|---|---|---|
| v(∅) | 所有 controller、所有 state 用同一个固定臂 | 在源 controller 上平均成功率最高的固定臂（round26 的 M1） |
| v(T) | 同上，但从完整菜单中选 | round26 的 M2 |
| v(C) | 每个 controller 一个固定臂 | A3，固定菜单，在 state 上取平均后选 |
| v(T,C) | 每个 controller 一个动作，完整菜单 | A3，完整菜单，在 state 上取平均后选 |
| v(S) | 每个 state 一个固定臂，不看 controller | A0（不看 controller 的 critic），固定菜单，逐 state 取 argmax |
| v(T,S) | 同上，完整菜单 | A0，完整菜单，逐 state 取 argmax |
| v(C,S) | 每个 (controller, state) 一个固定臂 | A3，固定菜单，逐 state 取 argmax |
| v(T,C,S) | 同上，完整菜单 | A3，完整菜单，逐 state 取 argmax |

A0、A3 都使用 round16 冻结、round26 已用过的版本。另外报告两种次要的 controller 条件化学习版，用来区分"按 controller 条件化"与"用行为探测条件化"：round26 的 M3（静态描述）与 G_E（规则）。

**oracle 版的估计（必须交叉拟合）：** 用同一批标签既选择又评估会高估。
- **交叉拟合值（下界）：** 用一半数据选、另一半评估，再交换，取平均。
  - 按 state 选择时，按 seed 分半；
  - 按 controller 选择时，按 state 分半；
  - Q4 的数据集按 seed 两两分组。
- **样本内最大值（上界）：** 同时报告，标明有偏。
- **平局规则：** 平局时选上一级格子的选择，例如 v(C,S) 平局取 v(C) 的臂。这样，只有当证据支持时，更细的条件化才会偏离上一级。
- **不看 controller 的 state 级 oracle（v(S)、v(T,S)）：** 对每个 state，用全部 controller 的平均标签来选，同一个 state 上所有 controller 用同一个臂。

**输出（每个轴，学习版与 oracle 版分别给出）：**
- **收益阶梯：** η = 0（有数据时）→ 一律 F10（旧做法）→ v(∅) → v(T) → v(T,C) → v(T,C,S)。每一级给出平均成功率的增量与 CI，并给出最大遗憾、灾难数（比 F10 低 ≥ 10pp 的 controller 数）。
- **Shapley 分解：** 三个因子的 Shapley 值，以 v(∅) 为起点：
  - φ_T = ⅓[v(T) − v(∅)] + ⅙[v(T,C) − v(C)] + ⅙[v(T,S) − v(S)] + ⅓[v(T,C,S) − v(C,S)]；
  - φ_C、φ_S 同理；
  - 三者之和等于 v(T,C,S) − v(∅)。
- **交互项：** I_TC = v(T,C) − v(T) − v(C) + v(∅)。I_TC < 0 表示时间表替代了 controller 信息。同样报告 I_TS 与 I_CS。
- **尾部：** 每个格子的最大遗憾与灾难数。

**CI：** 先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次；交叉拟合的 oracle 在每次重抽内重新计算。unknown 按下界计，另报上界。

**证据等级：** 这一部分的标签此前都已打开过，结论标为 DEVELOPMENT。分析定义必须在计算任何结果之前写入预注册。

# 三、第二部分：三个 B 的分解（新 rollout）

η10 与 η13 的三个分量都不同：goal 分别为 1.246 与 1.075，⊥ 分别为 −0.289 与 0.058，rel 分别为 0.186 与 0.266。所以已有的臂分不出哪个分量在起作用。本部分对分量做析因与敲除。
- η10、η13 的精确 float64 值，从共享 DB 的 eta 表中，按 round10 这两个臂的 eta_uid 读取。不得使用文档中的三位小数。
- 每个新臂都登记为新的 eta 或时间表身份。

**3.1 Flow（FF，主管线）**
- controller：round10 的 14 个原 Ring Flow（best.pkl）。
- state：round10 的 64 个新 state 中，按 state_uid 排序取前 32 个，写入预注册。
- seed：0–1，seed key 与 round10 相同，可与已有臂逐条配对。
- 已有、直接复用的臂：F10、F13、S300、S400、S500。
- 新臂（时间表都在第 500 步切换）：

| 代号 | 前期（第 0–499 步） | 后期（第 500 步起） | 数量 | 用途 |
|---|---|---|---|---|
| Z | η = 0 | η = 0 | 1（DB 中已有就复用） | 不加修正的 FF 基线 |
| K-goal、K-perp、K-rel | S500 中把该分量置 0 | 同左 | 3 | 每个分量是否必要 |
| E[g,p,r] | 每个分量取 η13（记 1）或 η10（记 0） | η10 | 6（E[0,0,0] = F10、E[1,1,1] = S500 已有） | 前期收益 E 由哪个分量承担 |
| L[g,p,r] | η13 | 每个分量取 η13（记 1）或 η10（记 0） | 6（L[0,0,0] = S500、L[1,1,1] = F13 已有） | 后期伤害 L 由哪个分量承担 |

规模：最多 14 × 32 × 2 × 16 = 14,336 条。

**3.2 MPC 与势场（TT）**
- controller：round27 的 12 个（M0–M5、P0–P5）。
- state：round27 的 32 个评估 state；seed 0–1，与 round27 配对。
- 已有、直接复用：F10、F13、S500，以及 round29 的 η = 0 与 nominal only。
- 新臂：

| 代号 | 内容 | 数量 | 用途 |
|---|---|---|---|
| F[g,p,r] | 全程固定，每个分量取 η13（1）或 η10（0） | 6（F[0,0,0] = F10、F[1,1,1] = F13 已有） | η10 为什么对这两类有害 |
| K13-goal、K13-perp、K13-rel | F13 中把该分量置 0 | 3 | 每个分量是否必要 |

规模：12 × 32 × 2 × 9 = 6,912 条。

**3.3 执行代码与复现门**
- **Flow：** 复用 round10 的 FF 执行代码（`diagnostics/controller_universal_eta_schedule_v1/`）。
- **MPC 与势场：** 复用 round27 的 TT 执行代码与 controller 实现（`diagnostics/other_controller_classes_v1/`）。round27 的部分脚本含全局变量替换和向其他目录的写入，不能直接运行。
- **复制规则：** 若这些代码在 import 时有副作用（改环境变量、往自己的目录写文件、替换全局变量），就复制到新目录。只允许修改路径常量与环境变量，其余逻辑逐字保持不变；提交 diff 与原文件 sha256。新臂只改 η 的取值与切换规则，不改代码逻辑。
- **复现门（在任何新 rollout 之前必须通过）：**
  - Flow：复现 round10 的 16 条记录：4 个 controller × {F10, S500} × 2 个 state，seed 0；
  - MPC 与势场：复现 round27 的 16 条记录：MPC 与势场各 2 个 × {F10, F13} × 2 个 state，seed 0。

  首动作字节、终止类型、步数与完整位置轨迹必须逐位一致，不一致就停下报告。

**3.4 流程：** design（登记新身份）→ preflight（在提交作业之前封存）→ 只跑缺失 → journal → 单进程官方 merger → postflight。

**3.5 分析：**
- **必要性：** 每个分量 c 的敲除损失 = S500 − K-c（MPC 与势场为 F13 − K13-c）。报告平均值与 CI，以及损失 ≥ 10pp 的 controller 数。
- **析因 Shapley：** 在三个 2³ 析因中，分别求三个分量的 Shapley 值：
  - E：对 S500 − F10 的分解，v(集合) = 前期取 η13 的分量集合对应的成功率；
  - L：对 S500 − F13 的分解，v(集合) = 后期换回 η10 的分量集合对应的成功率；
  - MPC 与势场：对 F13 − F10 的分解。

  同时报告各分量的份额（Shapley 值 ÷ 总差）及 CI，以及二阶、三阶交互。
- **逐 controller：** 每个 controller 的分量 Shapley 值，画热图。

# 四、预注册的终点与判定

第二部分的标签全新，下列终点为 CONFIRMATORY。第一部分为 DEVELOPMENT，只做描述，不判定通过与否。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| HB1 | 每个分量对 S500 是否必要（Flow FF） | 敲除损失的 CI 下界 > 5pp 记为必要。三个分量分别报告 |
| HB2 | 前期收益 E 的承担者 | 份额最大的分量，若份额的 CI 下界 > 0.5，记为单一承担者；否则记为分散 |
| HB3 | 后期伤害 L 的承担者 | 同 HB2 |
| HB4 | 承担者是否在 controller 之间共享 | 对 HB2、HB3 的承担者：≥ 80% 的 controller 上，它的 Shapley 值同号且绝对值 ≥ 5pp |
| HB5（描述性） | 跨类 | MPC、势场各自 F13 − F10 的分量份额；与 Flow 的 L 承担者是否为同一分量 |

# 五、结论映射（用于反推叙事与理论）

| 观察结果 | 对叙事与理论的含义 |
|---|---|
| 第一部分中 φ_T 远大于 φ_C、φ_S；C 主要降低尾部风险；学习版 φ_S ≈ 0 | 论文的核心是修正的阶段结构：时间表是不依赖 controller 的稳健动作；学习的作用是防止个别 controller 上的灾难 |
| oracle 的 φ_C 大，学习版的 φ_C 小 | controller 与 Q 的关系存在，但当前的探测与 critic 只学到一部分，是待解决的问题 |
| oracle 的 φ_S 大，学习版小 | 按 state 选择有空间但没有迁移；下一步方法应按进度或状态触发切换，而不是按固定时间 |
| I_TC < 0 且 CI 不含 0 | 时间表替代了 controller 信息：有了时间表，controller 差异大部分被吸收 |
| E 与 L 由不同的单一分量承担，且在 controller 间共享（HB2–HB4 通过） | 可以写出机制性理论：前期需要分量 X，后期需要分量 Y；固定 η 必须在两者间折中，这正是 H_compress |
| MPC、势场中使 η10 有害的分量，与 Flow 的 L 承担者相同 | 跨类机制相同：该分量的取值在后期或对非 Flow 类有害 |
| 承担者在 controller 间不共享（HB4 不通过） | 需要在分量层面按 controller 条件化，叙事应强调学习 |
| 某个分量敲除后损失 ≈ 0 | 该方向对这一轴不必要，基可以精简 |

# 六、其他硬性规则

- **源码保护：** 见第零节，优先于一切。
- **DB 写入：** 新目录之外唯一允许的写入，是共享 DB 的官方流程，共三处：
  - **design 阶段登记新身份：** 按 round28 `design.py` 的做法，用官方的 `shared_rollout_db.src.rollout_db.connect()` 与 `transaction()`，对 experiment、state、controller_config、eta 表（需要时还有 scenario 表）执行 `INSERT OR IGNORE`，只追加本轮的新身份。要求：
    - 不得 UPDATE 或 DELETE 任何已有行，也不得直接插入 rollout 行；
    - 同一 uid 已存在且内容不同时，停下报告；
    - 记录插入的全部行及其 sha256；
    - 插入前后，各登记表中已有行的内容哈希必须相同。
  - **journal：** worker 用 `shared_rollout_db.src.cache_writer.append_journal` 写到默认位置 `shared_rollout_db/journals/<experiment_uid>/`。
  - **合并：** 用 `python -m shared_rollout_db.src.merger --experiment <experiment_uid>` 单进程合并。

  这些位置都已被 git 忽略。不要通过修改 `cache_writer.ROOT` 等模块全局变量来重定向。
- **不得改动核心组件：** 不修改物理、安全投影、事件判定、成功标准与 controller。新臂只改变 η 的取值与切换规则。
- **参数写死：** 分析定义、state 子集、新臂的 η 值、阈值、平局规则与 bootstrap 种子，全部写入预注册，不得根据结果调整。
- **冻结时序：** 预注册（附 UTC 与 sha256 收据）必须早于第一部分的任何结果计算，也早于第二部分的任何新 rollout。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **精度：** 全部导入完成之后，显式恢复协议规定的精度（round25 的教训），并记录 x64 状态。
- **视频：** 本轮不要求视频。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 七、交付

全部交付物放在 `diagnostics/gain_attribution_v1/`：
- 源码完整性审计：开始与结束两次记录，以及二者的比对结果；
- PREREGISTRATION.md 及收据；
- 历史审计：已有标签的来源、可复用性、缺口；
- 复现门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- **收益账本：** 每个轴一张表，列出收益阶梯、8 个格子（学习版、oracle 交叉拟合值、oracle 样本内值）、Shapley 值与交互项、尾部指标；
- **B 分解表：** 敲除损失、三个析因的 Shapley 份额，以及逐 controller 热图；
- 图表：
  - 各轴收益来源的堆叠条形图（B、T、C、S）；
  - 收益阶梯的瀑布图；
  - 分量 Shapley 热图；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：
- 源码完整性审计是否通过；
- 按四个泛化轴（State、Controller、Cross-Scene、Cross-Factor）及其子项，收益有多少来自 B（以及是哪个分量）、多少来自时间表、多少来自 controller 条件化、多少来自 state 条件化。用 Shapley 值与 CI 表示，区分潜在收益与已实现收益，并列出没有数据的子项；
- E 与 L 各由哪个分量承担，是否在 controller 之间共享。

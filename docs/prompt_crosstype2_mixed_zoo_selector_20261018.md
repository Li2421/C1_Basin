你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。本轮是"跨 type"系列的第 2 轮（跨type2）。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`（已有 jax、flax、optax、distrax；不安装任何新包）
- 本轮目录：`diagnostics/crosstype2_mixed_zoo_selector_v1/`

**写作约定：** 本文件中的 controller、做法和终点都先写全称，括号中给出文件里的代号，代号只用来定位文件与身份。你的报告也照此书写：代号第一次出现时给出全称，之后正文用全称。

# 零、本轮要做什么

"跨 type"系列的目标（用户确定）是**做出**一条能跨 controller 类型零样本修好死锁的 pipeline：训练时见过很多种 controller，再零样本泛化到没见过的类型。方法就是我们论文的 pipeline：
- **生成器（generator）** q(η | h) 对每个 state 给出候选修正；
- **critic** Q(h, C, a) 预测"在 state h、controller C 下执行候选 a 的成功概率"，对每个 state 在候选中取最大者执行。

跨type1b 已经在学习型动物园上把这套 pipeline 跑通了。本轮把**最佳修法不同的非学习型 controller** 加进动物园，并把持续右手规则加进候选，让 critic 第一次面对"有的 controller 需要修正参数、有的需要持续右手规则、有的不需要修正"的情况。

本轮是**开发轮**：
- 在开发类型上允许迭代 critic 的结构、输入与候选组成，每次迭代都写进 `ITERATION_LOG.md`，不得只报告最好的一次；
- **锁定类型**（第三节）在冻结之前完全不能碰；冻结后在全新 state 上只评估一次，作为本轮唯一的确认性结果。

# 一、先读、边界与源码保护

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）** 第一至六节全部适用，不得放宽。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. **跨type1b（`diagnostics/crosstype1b_generator_critic_pipeline_v1/`，只读）：** `REPORT.md`、`HANDOFF.md`、`GAPS.md`、`ITERATION_LOG.md`、`AXIS_TABLE.md`、`AXIS_METRICS.json`、`CANDIDATES.json` 与 `condition_candidates_v2.py`（冻结的候选规则）、`critic_model.py`、`action_only_model.py`、`train_critic.py`、`train_action_only.py`、`select_models.py`、`select_models_v2.py`、`select_matched_v2b.py`、`new_evaluation_choices.py`（冻结推理与归一化 dtype 的恢复方式）、`statistics_pipeline.py`、`frozen_networks.py`、`resource_guard.py`、`FROZEN_PIPELINE.json`。HANDOFF 写明：不得直接运行对旧目录有写副作用的构造器，需要的实现复制到本轮目录并记录来源。
4. **pipeline 的冻结部件：** `docs/controller_generalization_audit_20261011.md` 第 1 节；`diagnostics/orthoflow3_generator_critic_v1/`（`REPORT.md`、`ARCHITECTURE_AUDIT.md`、`train_evaluate.py`、`normalization.json`、冻结的生成器 `generator/seed41/` 与原 critic `critic/seed23/`）；`diagnostics/orthoflow3_generator_critic_frozen_test_v1/run_frozen_test.py`（`FrozenNewRuntime.conditioning`）。
5. **跨type1（`diagnostics/crosstype1_learned_zoo_selector_v1/`，只读）：** `REPORT.md`、`HANDOFF.md`、`ZOO_MANIFEST.csv`、`features.py`、`assemble.py`、`initial_actions.py`、`DEVELOPMENT_FEATURES.npz`、`LOCKED_FEATURES.npz`、隐式能量策略的合成门记录、强化学习训练环境与 `ippo` 实现。
6. round38（`diagnostics/blackbox_families_v1/`）：`REPORT.md`、`GAPS.md`、`DESIGN_CONFIG.json`、`controllers.py`（交叉熵方法模型预测控制 `cross_entropy`）、`episode.py`、`EVALUATION_LABELS.npz`、`PILOT_STATES.json`、`TUNING_STATES.json`、`EVAL_STATES.json`、`pilot_*_FAIR_RULES.json`、`FIXED_RIGHT_HAND_RECOMPUTATION.json`、`corrected_astar_reference.py`、`CORRECTED_ASTAR_ENGINEERING_TESTS.json`、`ASTAR_OBLIQUE_UNIT_TEST.json`、`IMPLEMENTATION_GAP_ANALYSIS_AMENDMENT.json`、`fast_astar.py`、`astar_kernel.cpp`。
7. round37（`diagnostics/blackbox_addendum_v1/`）：`REPORT.md`、`continuing_rhr.py`（持续右手规则 `RightHand`）、`RIGHT_HAND_GRID_RESULTS.json`、`RIGHT_HAND_SELECTED.json`、`controllers.py`。round33（`diagnostics/axis2_controller_v1/`）：`controllers.py`（分布式模型预测控制 `distributed_shooting`）。round27（`diagnostics/other_controller_classes_v1/`）：`controllers.py`（集中式随机采样模型预测控制 `shooting` 与代价 `shooting_cost`）。
8. round35 的按进度切换实现 `diagnostics/axis4a_geometry_horizon_v1/progress.py`；round41 提示词 `docs/prompt_round41_multiscene_decentralized_reinforcement_learning_20261017.md` 第二至四节（只读提示词，不读多场景线目录）。

**源码保护：** 新代码与输出只写在本轮目录。已有源码、`docs/`、`AGENTS.md`、`datasets/`、其他全部 diagnostics 目录（包括跨type1、跨type1b 与 `orthoflow3_generator_critic_v1`）、PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004`（不在 git 中）、MAC-Flow 官方仓库与 venv 一律只读；需要改动的实现按共同协议复制到本轮目录并记录 sha256。冻结的生成器与 critic 权重只读加载。只做只读 git 操作（两个仓库都有用户未提交的工作）。开始与结束各做一次完整性审计。共享 DB 只通过官方设计阶段的 INSERT OR IGNORE 登记、`append_journal` 默认路径、单进程 merger 写入；导入后断言 `shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/` 之下（PoC/source 中有旧的同名副本）。

**与并行轮次的边界：**
- 另一个对话的分布式线目录：`diagnostics/decentralized_*`（含以后新建的同前缀目录）与 `diagnostics/teacher_student_*`，不得读写。唯一例外是共同协议第二节引用的 `diagnostics/decentralized_pipeline_v1/`，允许按协议只读。
- 多场景线目录：`diagnostics/multiscene_*`（含以后新建的同前缀目录），不得读写。
- 以上禁止目录一律按目录名前缀剪枝，不按轮次代号匹配。对禁止目录和并行任务目录，审计只允许记录路径、大小与修改时间（`os.stat`），不得打开文件、不得计算内容哈希。开工审计时，把最近 24 小时内被其他运行中任务写入的目录列为"并行任务目录"，其变化单独列出，不作为本轮审计失败的理由。
- 若因剪枝错误误读了禁止目录的文件（只读、内容未被使用、未写入），记录为边界偏差：写明涉及的文件、修正剪枝，然后继续。写入禁止目录，或在实验中使用了其中的内容，仍然整体停下。

**场景与安全层：** 只用 Ring 场景（round10 生成器）与集中式安全投影，H700，x64 = False。执行链与跨type1b 相同：环形交换场景流匹配策略用 FF 位置；其余全部用 TT 位置（PoC `Runtime.action` 的 `chain = "old"`），只替换 nominal 动作的来源。OrthoFlow3 修正每步重算，与历史实现相同。

# 二、论文定位与已有结果

**论文定位（用户确定）：** 修正黑盒 controller 在安全层下出现的死锁，不论死锁成因。黑盒指方法不依赖 controller 的控制律式子，只探查它的行为；学习得到的模型、模型预测控制、A* 一类规划器都算黑盒，手写闭式规则不纳入。死锁是否被解决，用成功率相对 safety baseline（只加安全投影、不加修正）的提升来衡量。timeout 不得直接称为 deadlock。

**跨type1（按 controller 选臂，学习型）：** 全部选择器对每个 controller 都选了按进度切换时间表。8 个学习型类型的臂排序完全相同，按 controller 选的空间只有 0.39pp。

**跨type1b（真正的 pipeline，学习型 9 类 62 个 controller）：** 新评估 state 上，每个 controller 都用没见过它所属类型的模型：
- pipeline 97.48%，safety baseline 15.83%，提升 81.65pp；比原 pipeline 原样（只选整局固定 η）高 22.13pp；
- 但相对一律按进度切换时间表只高 0.72pp [−1.14, 2.87]；按 state 交叉拟合事后最佳也只比它高 1.11pp；
- 不看 controller 的 critic（98.26%）不比完整 critic 差：在学习型类型之间，controller 信息没有用处，因为它们的最佳修法都一样；
- 留出类型的负对数似然在训练后期上升（例如流匹配策略从第 300 步的 0.589 升到第 2,000 步的 0.913）。

结论：pipeline 本身是成立的，但在只有学习型 controller 的 Ring 中，按进度切换时间表已经接近上限，pipeline 没有可赚的空间，controller 信息也派不上用场。

**已有标签显示，非学习型类型的最佳修法不同（Ring，集中式安全投影）：**
- 分布式模型预测控制：持续右手规则 100%，时间表约 80%（round37）；
- 集中式随机采样模型预测控制（round38，逐实例）：实例 0、2 用固定参数的持续右手规则 96.9% 和 93.8%，按比例切换时间表只有 75.0% 和 71.9%；实例 1、3、5 只加安全投影就已 100%；
- 交叉熵方法模型预测控制：6 个实例中 5 个只加安全投影就已 100%（另一个 98.4%）；全程 η10 对所有模型预测控制实例都有害（31–41%）；
- round38 的公平调参中，不同类型选中的持续右手规则参数不同（例如集中式随机采样模型预测控制与优先级时空 A* 规划偏好"始终触发、0.1 m/s"，分布式模型预测控制偏好"按进度触发、20 步、0.35 m/s"）。

所以本轮 critic 必须按 controller 区分三类修法；**controller 信息第一次应当有用**，这是本轮要检验的核心。

# 三、动物园

**开发类型（可以反复使用）：**

| 编号 | 全称（代号） | 实例数 | 来源与说明 |
|---|---|---|---|
| 开发 1–8 | 跨type1 的 8 个学习型类型：环形交换场景流匹配策略（Flow）、高斯（`gaussian`）、确定性（`deterministic`）、混合密度（`mixture`）、扩散（`diffusion`）、动作分块（`chunking`）、循环网络（`recurrent`）、条件变分自编码器（`cvae`）行为克隆策略 | 56 | 跨type1 `ZOO_MANIFEST.csv`，直接复用 checkpoint、探测与跨type1b 的标签 |
| 开发 9 | 参数共享独立近端策略优化强化学习策略（`ring_ippo`） | 6 | 同上 |
| 开发 10 | 分布式模型预测控制（`distributed_shooting`） | 6 | round33 / round37，已有 |
| 开发 11 | 集中式随机采样模型预测控制（`centralized_sampling0–5`） | 6 | round27 / round38，已有 |
| 开发 12 | 交叉熵方法模型预测控制（`cross_entropy0–5`） | 6 | round38，已有 |
| 开发 13 | 优先级时空 A* 规划（`prioritized_astar`） | 6 | 以 round38 的 `corrected_astar_reference.py` 为起点复制到本轮目录（可配合 `fast_astar.py` 的加速核，但必须先做与参考实现的等价性测试）。round38 的 `IMPLEMENTATION_GAP_ANALYSIS_AMENDMENT.json` 写明：须先通过完整单元测试（轴向与斜向空地图直线偏离 ≤ 1 个栅格；后规划的机器人从不进入先规划者占用的时空格；重规划时刻正确），再重新登记、重新探测，不得用 round38 已有的 244 条工程评估证据选参数。6 个实例沿用 round38 `DESIGN_CONFIG.json` 的 `ASTAR` 网格与跟踪设定 |
| 开发 14 | 保守近端策略优化强化学习策略（`ring_ippo_conservative`） | 6 | 与开发 9 的配方、观测、网络、超参数、训练 state、预算（每实例 2,000 万个机器人-步）与 checkpoint 规则完全相同，只改奖励：碰撞惩罚改为 −60（自己）与 −20（其余机器人），并加接近惩罚：与任一机器人或障碍的间距小于 0.2 m 时，每步 −0.05 ×（0.2 − 间距）/ 0.2。目的是得到更容易停住不动的学习型策略。训练环境复用跨type1 已通过等价性测试的 JAX 实现（逐字复制并记录 sha256），开工时复跑一次它的 float64 等价性测试 |

**锁定类型（冻结前完全不能碰：可以实现、训练、过能力门、算探测特征与候选，但只记录能力门通过与否和排除原因，不看任何修法臂的成功率）：**

| 编号 | 全称（代号） | 实例数 | 要点 |
|---|---|---|---|
| 锁定 1 | 分布式交叉熵方法模型预测控制（`distributed_cross_entropy`） | 6 | 每个机器人只优化自己的动作序列，其他机器人按当前速度匀速预测（与 `distributed_shooting` 的预测方式相同，按其源码核对后写入开工记录）；代价逐字复用 round27 的 `shooting_cost`，只对本机器人求和，并保留与预测轨迹之间的避撞项。优化器与 round38 交叉熵方法模型预测控制相同（精英比例 10%，初始标准差 0.8 × 最大速度，标准差下限 0.001，3 个节点）。6 个实例沿用 round38 `DESIGN_CONFIG.json` 的 `CEM` 网格。单元测试：单机器人空地图下与集中式交叉熵方法模型预测控制在同一随机键下的首动作一致（误差 ≤ 1e-9）；预测轨迹确为匀速 |
| 锁定 2 | 集中式模型预测路径积分控制（`centralized_mppi`） | 6 | 依据 Williams et al. 2016, ICRA, "Aggressive Driving with Model Predictive Path Integral Control"。对全部机器人的联合动作序列做采样（初始标准差 0.8 × 最大速度），代价逐字复用 round27 的 `shooting_cost`，按 exp(−代价/λ) 加权平均更新均值序列，执行第一步后平移热启动。6 个实例（规划步长，样本数，λ）：(10, 64, 1.0)、(10, 256, 1.0)、(10, 64, 0.1)、(10, 256, 0.1)、(30, 64, 1.0)、(30, 256, 0.1)。单元测试：λ → 0 时退化为取最低代价样本；同一随机键下动作逐字可复现 |
| 锁定 3（可选） | 隐式能量行为克隆策略（`implicit`） | 6 | 跨type1 未通过两峰合成门。只有在**不放宽原判据**的前提下修好实现并通过原合成门，才训练与纳入；否则记为缺口，不替换。它从未进行 Ring 评估，仍算未见类型 |

**新类型的共同要求：** 全部写在本轮目录，配单元测试；参数网格与训练设置在训练或评估之前写进 config 并留收据，不得根据结果调整。

**能力门（每个新实例，含锁定类型）：** round38 `PILOT_STATES.json` 的 16 个 pilot state × seed 0 × {safety baseline、按进度切换时间表、全程 η10、固定参数的持续右手规则}。四臂最高成功率 < 15% 就排除，不替换；只加安全投影就已 ≥ 98% 的实例**保留**，单独标注为"不需要修正"（它们是"不修正"的正例）。某类型剩下的实例少于 4 个，就记为缺口。锁定类型的能力门只记录通过与否。

**复现门：** 先在本轮复制的 runtime 上逐位复现：跨type1b 的 16 条记录（覆盖至少 3 个学习型类型与强化学习策略，含生成器候选动作）、round38 的 16 条记录（覆盖集中式随机采样与交叉熵方法模型预测控制）、round37 的 16 条记录（覆盖分布式模型预测控制与持续右手规则）。首动作字节、终止类型、步数与完整位置轨迹必须一致；冻结的生成器在跨type1b 的 state 上重算的候选必须与 `CANDIDATES.json` 逐位一致。不一致就把依赖它的部分记为缺口。

# 四、候选动作

**每个 state 的候选集（共 16 个）：**

| 来源 | 候选 | 说明 |
|---|---|---|
| 冻结的生成器 | 5 个 η 提议（变换后的均值与 4 个随机样本）× {全程 η_g，前期 η_g、按进度切换到 η10} = 10 个 | 规则与跨type1b 的 `CANDIDATES.json` / `condition_candidates_v2.py` 完全相同。生成器只看 state，不看 controller，所以同一 state 对所有 controller 给出相同的候选；跨type1b 已算好的 state 直接复用 |
| 锚点 | safety baseline（η = 0）、按比例切换时间表（`S500`）、按进度切换时间表（`Sprog`，p* = 0.9681620884830543，实现只读复用 round35 的 `progress.py`） | 3 个，与跨type1b 相同 |
| 持续右手规则（本轮新增） | 固定参数的持续右手规则（`RHR`，"按进度触发，40 步，0.35 m/s"）与短名单中另外 2 组（`RHR_b`、`RHR_c`） | 3 个；执行一律用 round37 的 `RightHand` |

**持续右手规则短名单（开工时按规则一次算定并留收据，之后不改）：** 汇总 round37 `RIGHT_HAND_SELECTED.json` 与 round38 全部 `pilot_*_FAIR_RULES.json` 中各组公平调参选中的参数，去掉与固定参数相同者，按"被几组选中"降序、再按这些组的调参成功率下界平均值降序排列，取前 2 组为 `RHR_b`、`RHR_c`；平局规则同 round30d。全部来源都是已有的开发数据，不涉及锁定类型。

另报 nominal only，不进候选。开发中可以改变候选的组成，但每次改动都要给全部开发 controller 补齐对应标签，写进迭代日志，并计入追加预算。

# 五、state、标签与探测

**state 划分：**
- **critic 训练 state：** 与跨type1b 相同的 48 个（跨type1 的 16 个调参 state 与 32 个评估 state）。
- **新评估 state（本轮新生成，冻结前不得跑任何闭环 rollout）：** 24 个，用 round10 生成器按开工时写定的种子生成，与 DB 中全部 state（包括跨type1b 的 24 个新评估 state）去重，seed 0–1。为它们计算 h、c 与生成器候选不需要 rollout，可以提前做。

**训练 state 的标签：** 每个开发 controller × 48 个训练 state × 16 个候选 × seed 0（跨type1 评估 state 上已有 seed 0–1 的锚点标签照常复用）。一律先做 DB preflight，只跑真正缺失的：
- 学习型与强化学习的 62 个 controller：只缺 3 个持续右手规则候选；
- 已有的 3 种模型预测控制（18 个）：round38 已有部分锚点与固定参数持续右手规则，缺生成器候选、`RHR_b`、`RHR_c`；
- 新开发类型（开发 13、14）：全部新跑。

**新评估 state 的标签（冻结之后才跑）：** 每个 controller × 24 个新评估 state × seed 0–1 × 16 个候选 + nominal only。范围：
- 全部锁定类型 controller；
- 全部非学习型与保守强化学习的开发 controller（开发 10–14），用没见过它所属类型的留一类型模型选择；
- 学习型开发类型（开发 1–9）每类按开工时写定的顺序取 2 个实例，用来确认加入新类型后 pipeline 在学习型上没有退步（跨type1b 已有全部学习型的结果）。

**机制探测（不看成败）：** 每个 controller × 16 个探测 state × {safety baseline、全程 η13、全程 η10、**固定参数的持续右手规则**} × seed 0 × 300 步，关闭终止判定，保存逐步轨迹。跨type1 已有的三臂探测直接复用；持续右手规则探测臂是本轮新增的，全部 controller 都要补跑。

**controller 特征 C：** 以跨type1 `features.py` 的 24 维机制特征为起点（逐字复用其定义），另加持续右手规则探测臂的同类特征：进度曲线、冻结比例、在动但不前进比例，以及"持续右手规则 − 全程 η13"在第 50、100、200、300 步的进度差。开发中可以增删特征，每次改动记入迭代日志。不得使用只在单一类型上拟合的降维或预处理。

**规模上限（开工时写定）：** 训练 state 补缺（预计 ≤ 35,000 条）；新评估 state（锁定 18 个 + 开发 10–14 共 30 个 + 学习型抽样 18 个，约 66 × 24 × 2 × 17 ≈ 54,000 条）；能力门、探测与强化学习训练另计。先做精确 preflight，再按实际缺口写定上限并记录。开发中若需追加标签，每次写进迭代日志，累计追加不超过 8,000 条。

# 六、critic 与 pipeline

**critic Q(h, C, a)：** 以跨type1b 的 `critic_model.py` 为起点（复制到本轮目录）：
- **输入：** state 分支为 100 维 `conditioning.flat` 与 33 维环境描述，按训练 state 归一化；controller 分支为特征 C；动作分支在跨type1b 的 10 维编码上增加持续右手规则的编码（是否为持续右手规则、触发方式独热、触发步数、速度）。
- **损失与训练：** 与跨type1b 相同：按观测到的成功次数做二项对数似然，权重为 min(实际 seed 数, 16)，unknown 不计入；AdamW，学习率 1e-3，权重衰减 1e-4，梯度范数截断 5；3 个 seed（17、23、41）集成；checkpoint 与超参数只用源类型内部的留一类型（嵌套）来选。
- **跨type1b 的教训，必须处理并报告：**
  1. 留出负对数似然在训练后期上升。报告每折的留出曲线与同一 state 内候选排序准确率，checkpoint 仍只按源内规则选。
  2. 跨type1b 第二版把"完整分支"与"不看 controller 的分支"按概率加权，controller 权重从 {0.1, 0.25, 0.5, 0.75, 1} 中按源内负对数似然选。本轮保留这一结构作为候选之一，并报告各折选中的权重。若 controller 信息真的有用，选中的权重应当变大。
- **选择规则：** 对每个 state 在 16 个候选中取集成均值最高者。另报保守版本：用集成均值减 1 倍标准差打分；若最高者相对按进度切换时间表的优势小于 0，就执行按进度切换时间表。

**对照方法（同表报告）：**

| 编号 | 全称 | 内容 |
|---|---|---|
| 对照 1 | 一律按进度切换时间表 | 学习型上的最佳单一修法 |
| 对照 2 | 一律固定参数的持续右手规则 | 分布式模型预测控制上的最佳单一修法 |
| 对照 3 | 一律不修正 | safety baseline |
| 对照 4 | 每个开发类型里最好的单一候选，按源类型多数决定 | 在源类型上平均最好的那个候选，用于全部留出 controller；"不靠任何输入"的最强固定方法 |
| 对照 5 | 跨type1b 冻结的 pipeline | 直接使用跨type1b `FROZEN_PIPELINE.json` 的全部九类型模型（它的候选中没有持续右手规则），不重训 |
| 对照 6 | 原 pipeline 原样 | 冻结的生成器 + 原 critic（`orthoflow3_generator_critic_v1`），只在 5 个全程固定 η 提议中取最高者 |
| 对照 7 | 不看 controller 的 critic | 去掉 C 输入，其余相同 |
| 对照 8 | 不看 state 的 critic | 去掉 h 与 c 输入，其余相同（生成器候选仍随 state 变化，报告时注明） |
| 少量标签基线 | 每个 controller 用前 k 个训练 state 的标签选一个固定候选 | k ∈ {2, 4, 8}，只对开发类型的留一类型折与锁定类型（锁定类型此时需要额外的训练 state 标签，在新评估 state 标签之后才跑，不影响冻结） |
| oracle（特权） | 按 state 的事后最佳候选 | 新评估 state 上用 seed 0 选、seed 1 打分，反之亦然，取平均；另报样本内与按 controller 的 oracle |

# 七、开发与冻结

**开发前先报告可选空间（只用训练 state 的标签，不看任何模型）：**
- 按 state 的样本内事后最佳候选、按 controller 的事后最佳候选，分别相对一律按进度切换时间表与对照 4；
- 学习型（开发 1–9、14）与非学习型（开发 10–13）分开；类型等权与 controller 等权各报一次；
- 每个开发 controller 的最佳候选属于哪一类（不修正、修正参数类、持续右手规则），形成一张"修法类型 × 类型"的表。

**留一类型开发检验（DEVELOPMENT，14 折）：** 每折留出一个开发类型，其余类型为源；全部预处理、超参数与 checkpoint 只能用源类型内部（嵌套）来选；在留出类型的训练 state 上打分。类型等权合并为主，另报 controller 等权，以及学习型与非学习型两个子集。另做留一族检验（描述性）：三族为学习型（开发 1–9、14）、优化型（开发 10–12）、搜索型（开发 13）。

**开发目标（达到就冻结；达不到也要在预算内冻结当前最好的版本，并如实报告）：**
- 非学习型子集：pipeline − 对照 4 的配对 CI 下界 > 0，且 pipeline − 一律按进度切换的配对 CI 下界 > 0；
- 类型等权合并：pipeline − 一律按进度切换的配对 CI 下界 > 0；
- 学习型子集不比一律按进度切换差 1pp 以上；伤害数（比 safety baseline 低 ≥ 10pp 的 controller 数）为 0；
- pipeline − 对照 7（不看 controller）在非学习型子集上的点估计 > 0，并报告 CI。

**规模曲线（DEVELOPMENT）：** 冻结的结构固定后，源类型数 m ∈ {1, 2, 4, 7, 10, 13}：对每个留出类型，从其余类型中随机抽 m 个（每个 m 抽 10 组，种子开工时写定；m = 13 只有一组），重新训练并记录留出遗憾。另画一条曲线，横轴改为源类型中"最佳修法类型不同"的种数（1、2、3），回答"见过的卡法越多，泛化越好"是否成立。

**冻结：** 把 14 个留一类型模型与用全部开发类型训练的最终模型，连同代码、输入定义、候选规则、选择规则写入 `FROZEN_PIPELINE.json` 并留 UTC 与 sha256 收据。然后为新评估 state 计算候选并用冻结模型选择（开发 controller 用没见过其类型的留一模型，锁定 controller 用最终模型），留选择收据。选择收据的时间必须早于新评估 state 的第一个 rollout 提交。

# 八、新评估 state 上的一次性评估（CONFIRMATORY）

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| 终点 1 | 修正在锁定类型上是否有效 | 每个锁定类型：pipeline − safety baseline 的配对 CI 下界 > 0；锁定类型若多数实例"不需要修正"，如实报告，并检查 pipeline 是否选了不修正或伤害了它们 |
| 终点 2（主） | 零样本 pipeline 是否胜过最好的固定方法 | 锁定类型合并（类型等权）：pipeline − max(一律按进度切换, 一律固定参数的持续右手规则, 对照 4) 的配对 CI 下界 > 0；另报每个锁定类型各自的结果，以及相对三个固定方法各自的差 |
| 终点 3 | 留出的非学习型开发类型 | 开发 10–14 合并（每个 controller 用没见过其类型的模型）：pipeline − 一律按进度切换、pipeline − 对照 4 的配对差与 CI |
| 终点 4 | 是否接近按 state 的事后最佳 | 锁定类型与开发 10–14 上，相对按 state 交叉拟合 oracle 的平均遗憾 ≤ 3pp，伤害数为 0 |
| 终点 5 | controller 信息是否有用 | pipeline − 对照 7（不看 controller）、pipeline − 对照 8（不看 state）在锁定类型与开发 10–14 上的配对差与 CI |
| 终点 6（描述性） | 修法类型选得对不对 | 每个 controller 被选中的候选中三类修法的比例，与它的按 controller 最佳修法类型对照；学习型抽样实例相对跨type1b 是否退步；特征落在开发范围之外的比例；nominal only → safety baseline 的逐条转移表 |

若锁定 3 缺失，终点 2 按现有锁定类型判定并注明。

**CI：** 按共同协议第四节，5000 次配对 bootstrap；类型等权时先在类型层重抽，再在 controller 层，再在 state 层成簇重抽（两个 seed 保留在同簇）；oracle 在每次重抽内重新计算。unknown 按下界计，另报上界。

# 九、执行顺序

1. 完整性审计（开始，含并行任务目录清单与 `squeue`）→ 复现门 → 新类型实现与单元测试（A* 的完整单元测试与加速核等价性测试；两个锁定模型预测控制；保守强化学习的奖励测试与环境等价性复测）→ 开工记录与收据（候选规则、持续右手规则短名单、state 划分、规模上限、锁定类型清单）。
2. 保守强化学习训练（GPU；锁定 3 若修好，也在此时训练）。
3. 开发类型：能力门 → 精确 preflight 与训练 state 补缺 → 机制探测（含新增的持续右手规则探测臂）→ 可选空间报告 → critic 开发与留一类型检验（迭代）→ 留一族检验 → 规模曲线。
4. 冻结。
5. 新评估 state：候选 → 冻结模型选择并留收据 → 标签 → 终点 1–6；之后再跑锁定类型的少量标签基线所需的训练 state 标签。
6. 完整性审计（结束）→ 交付。

任何阶段被挡住时，把受影响的类型记为缺口并写明原因，继续其余部分。只有源码保护、DB 规则或复现门对全部类型都无法满足时才整体停下。

# 十、结论映射

| 观察结果 | 结论 |
|---|---|
| 终点 2 与终点 3 都通过 | 跨类型零样本修正第一次做出来了：生成器加看 state 与 controller 的 critic，对没见过的黑盒类型胜过任何一种固定修法 |
| 终点 5 中不看 controller 的 critic 明显更差 | controller 信息在最佳修法不同的类型之间第一次起作用，这正是"跨 type"需要的 |
| 可选空间报告中非学习型子集有空间，但 critic 没能利用 | 瓶颈在 critic 的跨类型泛化；看留出曲线、controller 权重的选择与留一族检验 |
| critic 选对了持续右手规则这一类，但参数选错 | 修法类型能零样本迁移，参数不能；报告少量标签基线所需的 k |
| 规模曲线随"最佳修法不同的类型种数"下降 | "见过的卡法越多，泛化越好"成立；下一轮继续扩大动物园的卡法覆盖 |
| 锁定类型多数"不需要修正"，pipeline 也选了不修正 | pipeline 能识别"不需要修正"的 controller，不会因加修正而伤害它们 |
| 非学习型子集的可选空间也很小 | 在 Ring 上，最好的固定修法已经接近上限；下一轮应换到更难的场景 |

# 十一、交付

按共同协议第五节交付，放在本轮目录。另外：
- 动物园清单（每个 controller 一行：类型、族、配方、checkpoint 或参数、能力门结果）；新类型的实现、单元测试、训练 config、训练曲线、checkpoint 与收据；复现门报告；持续右手规则短名单的计算与收据；候选清单与收据；
- 探测轨迹清单与特征表；可选空间报告与"修法类型 × 类型"表；`ITERATION_LOG.md`；critic 的代码、config、留出曲线、同一 state 内排序准确率、controller 权重的选择、checkpoint 与收据；`FROZEN_PIPELINE.json` 与选择收据；规模曲线（两种横轴，图与数据）；
- AXIS_TABLE：每个类型一行，另加"开发类型合并""学习型合并""非学习型合并""锁定类型合并"四行；列为 16 个候选中的锚点与持续右手规则、第六节全部方法与 oracle，分"训练 state 留一类型"和"新评估 state"两部分；单元格为成功率 % [95% CI]；另附碰撞率、超时率、平均与最大遗憾、伤害数、证据等级；
- 每个 (controller, 新评估 state) 的选中候选、各候选的成功数、遗憾的明细表；
- 每个类型一张"选中动作分布"图（按三类修法着色）；每个锁定类型一个典型 state 上 safety baseline、按进度切换时间表、持续右手规则与 pipeline 选中动作的轨迹静态图；
- REPORT.md：第一段先写完整性审计是否通过，再给出可选空间（学习型与非学习型分开）、开发目标是否达到、终点 1–5 的结论与关键数字，以及每个新类型的一句话总结（偏好哪种修法、卡住时是严格冻结还是在动但不前进、pipeline 是否选对）；
- HANDOFF.md（不依赖本对话也能读懂），写明下一轮（跨type3）需要的输入与路径。

**资源：** 用户全部任务合计不超过 18 个 CPU shard、2 个 GPU shard。开工前与每次提交前都要用 `squeue` 查看总占用。**不做 git commit。**

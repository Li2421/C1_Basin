你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。本轮是"跨 type"系列的跨type3。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`（已有 jax、flax、optax、distrax；不安装任何新包）
- 本轮目录：`diagnostics/crosstype3_unseen_controller_schedule_v1/`

**写作约定：** 本文件中的 controller、做法和终点都先写全称，括号中给出文件里的代号，代号只用来定位文件与身份。你的报告也照此书写：代号第一次出现时给出全称，之后正文用全称。

# 零、本轮要做什么（用户 2026-10-19 确定）

**本轮只回答一个问题：** 我们的主线 pipeline 能否泛化到**没见过的 controller**。

**主线 pipeline（沿用跨type1c，不改结构）：**
- 动作是一个两 η 时间表 a = (η_前期, η_后期, 切换时刻)；
- 冻结的生成器对每个 state 给出 8 个 η 提议，两个 η 都取自这 8 个提议；
- critic Q(h, C, a) 在候选中选出前期 η、后期 η 与切换时刻；
- η13、η10 不进入候选，也不进入 critic 训练；人定时间表只作对照。

**用户确定的范围：**
- 只做两 η 时间表。**持续右手规则不进入候选**，也不作为对照方法；
- 本轮是**检验轮**，不是开发轮：critic 的结构、输入、候选规则与选择规则全部按跨type1c 冻结的版本执行，只把训练数据扩大到混合动物园并重训。不得为了提高结果改变结构；实现错误的修正写进 `ITERATION_LOG.md`，并说明它不依赖任何锁定类型的结果。

**什么叫"没见过的 controller"：** 本轮新建三个黑盒 controller 类型。它们在冻结之前完全不能碰（第三节），冻结后在全新 state 上只评估一次，作为本轮唯一的确认性结果。已有的 16 个黑盒类型都在以前的轮次里被研究者用过，在本轮全部作为训练类型；它们上的留一类型结果只算开发证据。

**为什么不能用已有的类型：**
- 跨type1c 的留出类型是留一类型检验：模型没见过它们，但研究者在开发中反复用过；
- 跨type2 的两个锁定类型（集中式模型预测路径积分控制、分布式交叉熵方法模型预测控制）已经打开过一次；
- 势场、切向势场是手写闭式规则，属于白盒，不在论文范围内；速度障碍 controller 的归属用户尚未决定，本轮不用。

# 一、先读、边界与源码保护

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）** 第一至六节全部适用，不得放宽。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. **跨type1c（`diagnostics/crosstype1c_two_eta_schedule_v1/`，只读）：本轮 pipeline 的来源。** `REPORT.md`、`HANDOFF.md`、`GAPS.md`、`ITERATION_LOG.md`、`AXIS_TABLE.md`、`PREREGISTRATION.md`、`candidates.py`、`CANDIDATES.json`、`STATE_INPUTS.npz`、`TRAINING_STATES.json`、`TRAINING_ACTIONS.npz`、`TRAINING_BUNDLE.npz`、`BASE_PROTOCOL.json`、`CONTROLLER_FEATURES.npz`、`critic_model.py`、`train_critic.py`、`select_models.py`、`CRITIC_TRAINING_PLAN.json`、`SELECTED_MODELS.json`、`MAIN_METHOD_SELECTION.json`、`inference.py`、`new_choices.py`、`FROZEN_PIPELINE.json`、`statistics_pipeline.py`、`confirmatory_statistics.py`、`NEW_EVAL_STATES.json`、`resource_guard.py`。
4. **跨type2（`diagnostics/crosstype2_mixed_zoo_selector_v1/`，只读）：混合动物园的来源。** `REPORT.md`、`HANDOFF.md`、`GAPS.md`、`ZOO_MANIFEST.csv`、`policies/`、`mixed_controllers.py`（`CentralizedMPPI`、`DistributedCrossEntropy`）、`mixed_runtime.py`、`mixed_execution.py`、`zoo_design.py`、`conservative_reward.py`、`CONTROLLER_FEATURES.npz`、`assemble_features.py`、`probes.py`、`probe_prefix.py`、`NEW_EVAL_BUNDLE.npz`。它的持续右手规则相关部分本轮不用。
5. **跨type1（`diagnostics/crosstype1_learned_zoo_selector_v1/`，只读）：** `ZOO_MANIFEST.csv`、`features.py`（24 维机制特征 `NAMES`、`one_trace`、`aggregate`）、`train_bc.py`、`zoo_models.py`、`TRAINING_CONFIGS.json`。
6. **round38（`diagnostics/blackbox_families_v1/`，只读）：** `spacetime_astar.py`（`SpaceTimeGrid`、`PrioritizedAStar`）、`fast_astar.py`（`FastGrid`、`FastPrioritizedAStar`，C++ 核 `astar_kernel.cpp`）、`corrected_astar_reference.py`（`CorrectedPrioritizedAStar`）、`CORRECTED_ASTAR_ENGINEERING_TESTS.json`、`DESIGN_CONFIG.json`（`ASTAR` 网格与跟踪设定）、`PILOT_STATES.json`、`train_policies.py`、`learned.py`。
7. round27（`diagnostics/other_controller_classes_v1/controllers.py`）：`shooting_cost` 与集中式随机采样模型预测控制 `shooting`。
8. 行为克隆数据：`ring_exchange_stage1/` 下各数据集的 `manifest.json`（观测 [4, 23]，展平 92 维；动作 [4, 2]，展平 8 维；train / dev / test 划分），加载器 `new_benchmark_common/dataset.py` 的 `JointTransitionDataset` 与 `fit_train_normalization`。
9. pipeline 的冻结部件：`diagnostics/orthoflow3_generator_critic_v1/`（冻结生成器 `generator/seed41/`、`normalization.json`）；`diagnostics/orthoflow3_generator_critic_frozen_test_v1/run_frozen_test.py`（`FrozenNewRuntime.conditioning`）。按进度切换的实现：`diagnostics/axis4a_geometry_horizon_v1/progress.py`。

**源码保护：** 新代码与输出只写在本轮目录。已有源码、`docs/`、`AGENTS.md`、`datasets/`、其他全部 diagnostics 目录（包括跨type1、跨type1b、跨type1c、跨type2）、PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004`（不在 git 中）、MAC-Flow 官方仓库与 venv 一律只读；需要改动的实现按共同协议复制到本轮目录并记录来源 sha256。旧目录中有写副作用的构造器不得直接运行（跨type1c 与跨type2 的 HANDOFF 都写明了这一点）。冻结的生成器与旧模型权重只读加载。只做只读 git 操作（两个仓库都有用户未提交的工作）。开始与结束各做一次完整性审计。共享 DB 只通过官方设计阶段的 INSERT OR IGNORE 登记、`append_journal` 默认路径、单进程 merger 写入；导入后断言 `shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/` 之下（PoC/source 中有旧的同名副本）。

**与并行轮次的边界：**
- 另一个对话的分布式线目录：`diagnostics/decentralized_*`（含以后新建的同前缀目录）与 `diagnostics/teacher_student_*`，不得读写。唯一例外是共同协议第二节引用的 `diagnostics/decentralized_pipeline_v1/`，允许按协议只读。
- 多场景线目录 `diagnostics/multiscene_*` 不得读写。
- 以上禁止目录一律按目录名前缀剪枝，不按轮次代号匹配。对禁止目录和并行任务目录，审计只允许记录路径、大小与修改时间（`os.stat`），不得打开文件、不得计算内容哈希。开工审计时，把最近 24 小时内被其他运行中任务写入的目录列为"并行任务目录"，其变化单独列出，不作为本轮审计失败的理由。
- 运行期间用户可能在 `docs/` 新增提示词文件（`docs/prompt_*.md`）。结束审计时，`docs/` 中新增的 `prompt_*.md` 只按 `os.stat` 记录，单独列为"用户新增提示词"，不打开、不哈希，不作为审计失败的理由；`docs/` 中已有文件被修改或删除，仍按差异处理。
- 若因剪枝错误误读了禁止目录的文件（只读、内容未被使用、未写入），记录为边界偏差：写明涉及的文件、修正剪枝，然后继续。写入禁止目录，或在实验中使用了其中的内容，仍然整体停下。

**场景与安全层：** 只用 Ring 场景（round10 生成器）与集中式安全投影（PoC `Runtime.safety`，内部为 `shared_control/hard_projection.py` 的精确二阶锥投影），H700，x64 = False。执行链与跨type1c 相同：环形交换场景流匹配策略用 FF 位置，其余全部用 TT 位置（PoC `Runtime.action` 的 `chain = "old"`），只替换 nominal 动作的来源。OrthoFlow3 修正每步重算；切换时刻之后只把 η 换成后期 η，其余执行链不变。

# 二、论文定位与已有结果

**论文定位（用户确定）：** 修正黑盒 controller 在安全层下出现的死锁，不论死锁成因。黑盒指方法不依赖 controller 的控制律式子，只探查它的行为；学习得到的模型、模型预测控制、A* 一类规划器都算黑盒，手写闭式规则不纳入。死锁是否被解决，用成功率相对 safety baseline（只加安全投影、不加修正）的提升来衡量。timeout 不得直接称为 deadlock。

**跨type1c（主线，学习型 9 类 62 个 controller，24 个新 state，每个 controller 用没见过其类型的留一模型）：**
- 自学两 η 与切换时刻的 pipeline 95.81%，safety baseline 14.75%，提升 81.06pp；
- 两 η 学出、固定第 500 步切换 96.19%；不带时间表（整局一个 η）76.73%；
- 人定时间表：前 500 步 η13、之后 η10 为 95.61%，按进度从 η13 切到 η10 为 96.56%；主方法相对二者中较好者 −0.76pp [−3.86, +2.16]，非劣（界限 −1pp）没有成立；
- 自学切换时刻相对固定第 500 步 −0.38pp [−2.68, +1.57]；选中时刻集中在 200 与 600 两端；
- 选出的后期 η 中位数约 (0.88, −0.10, 0.047)，离 η10 约 0.43，不是 η10；但"后期 η − 前期 η"与 η10 − η13 的方向余弦中位数约 0.9；
- 不看 controller 的 critic 不比完整 critic 差（主方法减它 −1.00pp [−3.22, +1.35]）。

**跨type2（旧设计：后期 η 写死为 η10，另有持续右手规则候选）：**
- 锁定类型上 pipeline 比 safety baseline 高 22.92pp，但输给最强固定修法；
- 训练时排除全部学习型类型，pipeline 在学习型上只有 29.31%：跨族泛化很弱。

**所以：** 两 η 时间表 pipeline 在学习型的留出类型上成立，但还没有在非学习型上训练与检验过，也从来没有在真正未见的 controller 上检验过。这就是本轮要补的。

# 三、动物园

**训练类型（16 个，全部已有，可以使用其已有标签）：**

| 族 | 全称（代号） | 实例数 | 来源 |
|---|---|---|---|
| 学习型 | 环形交换场景流匹配策略（`Flow`） | 14 | 跨type1 / 跨type2 `ZOO_MANIFEST.csv` |
| 学习型 | 高斯（`gaussian`）、确定性（`deterministic`）、混合密度（`mixture`）、扩散（`diffusion`）、动作分块（`chunking`）、循环网络（`recurrent`）、条件变分自编码器（`cvae`）行为克隆策略 | 各 6 | 同上 |
| 学习型 | 参数共享独立近端策略优化强化学习策略（`ring_ippo`）、保守参数共享独立近端策略优化强化学习策略（`ring_ippo_conservative`） | 各 6 | 同上 |
| 优化型 | 分布式模型预测控制（`distributed_shooting`）、集中式随机采样模型预测控制（`centralized_sampling`）、交叉熵方法模型预测控制（`cross_entropy`）、分布式交叉熵方法模型预测控制（`distributed_cross_entropy`）、集中式模型预测路径积分控制（`centralized_mppi`） | 各 6 | 跨type2 `ZOO_MANIFEST.csv` 与 `mixed_controllers.py` |
| 搜索型 | 优先级时空 A* 规划（`prioritized_astar`） | 6 | 跨type2（修正版） |

实例与能力门结论一律以跨type2 `ZOO_MANIFEST.csv` 为准；跨type2 排除的实例本轮也排除。

**未见类型（本轮新建，3 个，冻结前锁定）：**

| 编号 | 全称（代号） | 族 | 设计要点 |
|---|---|---|---|
| 未见 1 | Transformer 行为克隆策略（`transformer_bc`） | 学习型 | 把 92 维观测看成 4 个机器人各 23 维的集合：逐机器人线性嵌入（宽度 128）→ L 层自注意力块（flax `nn.SelfAttention`，4 头，前置 LayerNorm，前馈 256，残差）→ 每个机器人输出 2 维均值与 2 维对数标准差（截断到 [−5, 1.5]），高斯负对数似然训练。训练框架复制跨type1 的 `train_bc.py`：同一数据集、同一归一化（取自对应流匹配 checkpoint 的 `obs_mean / obs_scale / act_mean / act_scale`）、同一 dev 划分与 checkpoint 规则；推理按高斯行为克隆策略的方式采样并做速度截断。6 个实例：(L, 数据集, 种子) = (2, `base_u_dataset`, 0)、(2, `base_u_v3_dataset`, 1)、(2, `base_u_plus_w_v3_dataset`, 2)、(4, `base_u_dataset`, 3)、(4, `base_u_v7_local_dataset`, 4)、(4, `base_u_v8_local_dataset`, 5)；局部坐标数据集按跨type1 的帧规则处理。仓库中已有的 `new_benchmark_common/macflow.py` `SetActorVectorField` 是流匹配向量场，不是本类型，不得替代 |
| 未见 2 | 梯度下降模型预测控制（`gradient_mpc`） | 优化型 | 集中式：对 4 个机器人的联合动作序列（3 个节点，线性插值到规划步长，与 round38 交叉熵方法模型预测控制相同的参数化）用 Adam 直接对代价求梯度做 K 步更新，执行第一步后平移热启动；初值为"朝目标直线"动作。代价把 round27 的 `shooting_cost` 逐行改写为 `jax.numpy`（铰链平方项可微，无分支），惩罚权重沿用 round38 的 `CEM_penalty_weight` = 100；每步更新后把动作截断到速度上限。单元测试：jnp 代价与 numpy 原版在 1,000 组随机输入上相对误差 ≤ 1e-6（float32）且 float64 下 ≤ 1e-12；梯度与有限差分一致；同一随机键下动作逐位可复现。6 个实例（规划步长，迭代步数 K，学习率）：(10, 20, 0.05)、(10, 50, 0.05)、(10, 50, 0.2)、(30, 20, 0.05)、(30, 50, 0.05)、(30, 50, 0.2) |
| 未见 3 | 冲突搜索多机规划（`conflict_based_search`） | 搜索型 | 依据 Sharon et al. 2015, Artificial Intelligence, "Conflict-based search for optimal multi-agent pathfinding"。底层单机搜索复用 round38 的时空栅格（`SpaceTimeGrid` / `FastGrid`）：8 邻域加等待、时间量子 max(dt, 分辨率/0.52)、扫掠圆盘碰撞检查与到达后占位，与优先级时空 A* 规划完全相同；高层在冲突树上按代价之和最小优先展开，冲突为两机器人在同一时间量子的扫掠圆盘重叠，约束加到单机搜索中。高层展开节点超过上限时，退回当前最优节点的路径（冲突未解的部分照常执行，由安全层处理），并记录退回次数。跟踪为连续动作的规则与优先级时空 A* 规划完全相同（前视 0.3 m，速度 0.52 × min(1, 到目标距离/0.3)）；每隔 K 步重规划。6 个实例（分辨率，K，高层节点上限）：(0.05, 10, 64)、(0.05, 10, 512)、(0.05, 25, 512)、(0.1, 10, 64)、(0.1, 10, 512)、(0.1, 25, 512)。单元测试：两机器人对穿走廊的标准算例得到无冲突且代价之和最优的解；节点上限为 1 时退化为各自独立的最短路；与优先级时空 A* 规划在单机器人空地图上路径一致 |

**新类型的共同要求：** 全部代码写在本轮目录并配单元测试；参数网格、训练设置与 checkpoint 规则在训练或评估之前写进 config 并留收据，不得根据任何结果调整。

**锁定规则（冻结前）：** 可以实现、训练、跑单元测试、过能力门、算机制探测特征与候选。能力门只记录通过与否和 safety baseline 的成功率（safety baseline 不是修正方法），不得查看或汇总任何修正动作的成功率；机制探测的轨迹只用于算特征，关闭终止判定，不读取成败。

**能力门（每个未见实例）：** round38 `PILOT_STATES.json` 的 16 个 pilot state × seed 0 × {safety baseline、按进度从 η13 切到 η10、全程 η13}。程序只输出"三者最高成功率是否 ≥ 15%"这一个布尔值与 safety baseline 成功率，不输出另外两者的数值。不通过就排除，不替换。另按 safety baseline 成功率标注：≥ 98% 记为"不需要修正"，照常保留在评估中，用于检查 pipeline 是否伤害它们。某类型剩下的实例少于 4 个，就记为缺口；某类型全部实例都"不需要修正"，它不进入终点 1 的修正有效性判定，只进入"不伤害"检查。

**复现门：** 先在本轮复制的 runtime 上逐位复现：跨type1c 的 16 条记录（覆盖至少 3 个学习型类型，含两 η 时间表动作）与跨type2 的 16 条记录（覆盖集中式模型预测路径积分控制、分布式交叉熵方法模型预测控制与优先级时空 A* 规划）。首动作字节、终止类型、步数与完整位置轨迹必须一致；冻结生成器在跨type1c 的 state 上重算的 8 个提议必须与其 `CANDIDATES.json` 逐位一致。不一致就把依赖它的部分记为缺口。

# 四、候选、state 与标签

**候选（与跨type1c 完全相同）：** 每个 state 由冻结生成器给出 8 个提议（变换后的均值与 7 个确定性随机样本，派生规则沿用跨type1c `candidates.py`）。候选为 64 个有序对 × 切换时刻 {200, 300, 400, 500, 600}；对角元（η_前期 = η_后期）即整局固定一个 η，只保留第 500 步作为规范身份。动作编码沿用跨type1c 的 10 维：η_前期(3)、η_后期(3)、η_后期 − η_前期(3)、切换时刻/700。

**state：**
- **critic 训练 state：** 跨type1c 的 48 个训练 state（`TRAINING_STATES.json`）。
- **新评估 state（冻结前不得跑任何闭环 rollout）：** 48 个，用 round10 生成器按开工时写定的种子新生成，与 DB 中全部 state 去重，包括跨type1b、跨type2（种子 202610182001–202610182024）与跨type1c 的新评估 state。seed 0–1。为它们计算 h、c 与提议不需要 rollout，可以提前做。

**训练 state 的标签（训练类型，seed 0）：** 每个 (controller, 训练 state) 26 个动作，规则与跨type1c 相同：8 个对角元全部标出；从 56 个非对角有序对 × 5 个切换时刻中按种子抽 18 个，保证 5 个切换时刻各至少 3 个。
- 跨type1c 的 62 个学习型 controller：直接复用其 `TRAINING_ACTIONS.npz` 的抽样与 DB 中的标签，不重新抽；
- 其余训练 controller（保守强化学习、5 种模型预测控制、优先级时空 A* 规划，共 42 个）：按跨type1c 的派生规则，用 controller 身份派生抽样种子；
- 一律先做 DB 精确 preflight，只跑真正缺失的；unknown 保留，不重跑。
- 另外，从已有标签中报告训练 state 上 safety baseline 与两种人定时间表的成功率，只作参照，不进入 critic 训练。

**机制探测与 controller 特征 C：** 沿用跨type1 `features.py` 的 24 维定义（探测臂 safety baseline、全程 η13、全程 η10，16 个探测 state，seed 0，300 步，关闭终止判定）。跨type1c 与跨type2 已算的 24 维特征可以复用：开工时各抽 4 个 controller 从轨迹重算，逐位一致才复用，否则全部重算。未见类型的探测在本轮补跑。不使用跨type2 的持续右手规则特征。

**新评估 state 的标签（冻结之后才跑）：** 每个 (controller, 新评估 state) × seed 0–1 × 下列动作（相同物理动作只跑一次，按身份去重）：

| 动作 | 未见类型 | 训练类型 |
|---|---|---|
| 1. 冻结的主方法选中的动作 | 是 | 是 |
| 2. 另一种时间表 pipeline 选中的动作（主方法为自学切换时刻时，这里是固定第 500 步切换；反之亦然） | 是 | 是 |
| 3. 不带时间表的 pipeline 选中的动作（只在 8 个对角元中选） | 是 | 否 |
| 4. 不看 controller 的 critic 选中的动作 | 是 | 是 |
| 5. 跨type1c 冻结的 pipeline 选中的动作（只见过学习型，不重训；用其全部九类型最终模型） | 是 | 是 |
| 6. 主方法 critic 排名第 2–4 的动作（用于判断 critic 是否挑错，见第六节失败归因） | 是 | 否 |
| 7. safety baseline；前 500 步 η13、之后 η10；按进度从 η13 切到 η10 | 是 | 是 |
| 8. nominal only | 是 | 否 |

即未见类型每 (controller, state) 至多 12 个动作，训练类型至多 7 个。

**保守版本的补评（用户 2026-10-19 确定）：** 第五节的保守版本（集成均值减 1 倍标准差）选中的动作若不在上表清单中，允许补评，未见类型与训练类型都适用，seed 0–1。补评动作在冻结后、首次新评估提交前与其他选择一起写进选择收据；精确 preflight 去重后，把实际缺失数量作为单独一项预算封存，上限为 18 × 48 × 2 + 42 × 24 × 2 = 3,744 条，不占其他各项预算。保守版本只作次要报告，不改变主方法与任何终点的判定。

**评估范围：**
- 全部通过能力门的未见实例 × 48 个新评估 state；
- 训练类型中 6 种非学习型类型（5 种模型预测控制与优先级时空 A* 规划）与保守强化学习的全部 controller，共 42 个，× 前 24 个新评估 state，每个 controller 用没见过其类型的留一类型模型；
- 学习型训练类型不在新评估 state 上评估（跨type1c 已有其新 state 结果）；加入非学习型训练类型后学习型是否退步，只在训练 state 的留一类型检验中与跨type1c 对照报告。

**事后诊断标签（未见类型；在终点 1–7 的结果写入并留收据之后才跑，不改变任何确认性结论）：** 每个未见 controller × 新评估 state 中按开工时写定的种子抽出的 8 个 × seed 0 × 26 个动作，抽样规则与训练 state 完全相同（8 个对角元全部标出；56 个非对角有序对 × 5 个切换时刻中抽 18 个，保证 5 个切换时刻各至少 3 个）。用途见第六节"失败归因"。

**规模上限（开工时写定）：**

| 部分 | 估计 | 说明 |
|---|---|---|
| 训练 state 补缺 | 约 42,000 条 | 42 × 48 × 26，扣除与跨type2 生成器整局固定动作同身份的对角元 |
| 未见类型新评估 | 约 21,000 条 | 18 × 48 × 2 × 12 |
| 训练类型新评估 | 约 14,000 条 | 42 × 24 × 2 × 7 |
| 事后诊断 | 约 3,700 条 | 18 × 8 × 26 |

合计约 81,000 条，另加保守版本补评至多 3,744 条（单独封存），扣除精确复用；能力门、探测与新类型训练另计。先做精确 preflight，再按实际缺口写定上限并记录。实现错误导致的重跑写进迭代日志，累计不超过 8,000 条。模型预测控制与规划器的单条 rollout 远慢于学习型 controller：开工后先用每个新类型各 4 条 rollout 测出单条耗时，写进开工记录，并据此安排提交顺序（未见类型的主评估优先于训练类型的新评估）。

# 五、critic 与 pipeline（按跨type1c 冻结版本，不改结构）

- **结构、输入与损失：** 复制跨type1c 的 `critic_model.py`、`train_critic.py`、`select_models.py`。state 分支 100 维 `conditioning.flat` 与 33 维环境描述；controller 分支 24 维机制特征；动作分支 10 维编码。二项对数似然，权重 min(实际 seed 数, 16)，unknown 不计入；AdamW，学习率 1e-3，权重衰减 1e-4，梯度范数截断 5；seed 17、23、41 集成；"完整分支与不看 controller 的分支按概率加权"的结构与权重候选不变。
- **预处理、checkpoint 与权重：** 只用源类型内部的留一类型（嵌套）来选，与跨type1c 的规则相同。冻结推理恢复 h / c 归一化参数 float32、controller 参数 float64，使用 float32 highest 矩阵乘法（与跨type1c、跨type2 相同）。
- **选择规则：** 对每个 state 在候选中取集成均值最高者；精确并列规则沿用跨type1c `new_choices.py`。另报保守版本：集成均值减 1 倍标准差。
- **主方法的确定（开工时写定，沿用跨type1c 的规则）：** 在 16 折留一类型检验中，若自学切换时刻的类型等权成功率（只在已标出动作上）不低于固定第 500 步切换减 0.5pp，且 critic 对"同一有序对、不同切换时刻"的排序准确率高于 0.5，就冻结自学切换时刻为主方法；否则冻结固定第 500 步切换为主方法。两种都冻结并在新评估 state 上评估。

**留一类型检验（DEVELOPMENT，16 折）：** 每折留出一个训练类型，其余为源，在留出类型的训练 state 上只对已标出动作打分。报告 critic 选中的已标出动作的成功率、三种排序准确率（对角元之间、非对角对之间、对角元与非对角对之间）与"同对不同时刻"排序准确率；类型等权与 controller 等权；学习型、优化型、搜索型三族分开。另做留一族检验（描述性）：留出整个学习型族、优化型族或搜索型族。

**冻结：** 把 16 个留一类型模型与用全部 16 个训练类型训练的最终模型，连同代码、输入定义、提议规则、选择规则与主方法判定写入 `FROZEN_PIPELINE.json` 并留 UTC 与 sha256 收据。然后为 48 个新评估 state 计算提议并选择（训练类型的 controller 用没见过其类型的留一模型；未见类型用最终模型），留选择收据。选择收据的时间必须早于新评估 state 的第一个 rollout 提交，也必须早于任何未见类型修正动作的 rollout。

# 六、新评估 state 上的一次性评估（CONFIRMATORY，只看未见类型）

**CI：** 按共同协议第四节，5000 次配对 bootstrap。未见类型只有 3 个，类型层重抽没有意义，所以未见类型合并时**按类型分层、类型等权**：类型固定，在每个类型内重抽 controller，再在 state 层成簇重抽（两个 seed 保留在同簇）。每次重抽重新取较好的人工时间表与已评估动作中的 oracle。unknown 按下界计，另报上界。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| 终点 1（主） | 主线在未见 controller 上是否修好死锁 | 每个"需要修正"的未见类型：主方法 − safety baseline 的配对 CI 下界 > 0；未见类型合并（分层、类型等权）同样判定。另报伤害数：主方法比 safety baseline 低 ≥ 10pp 的 controller 数，以及"不需要修正"的实例上主方法 − safety baseline |
| 终点 2 | 学出的两 η 能否替代人定时间表 | 未见类型合并：主方法 − max(前 500 步 η13 之后 η10, 按进度从 η13 切到 η10) 的配对 CI 下界 > −1pp（非劣）；另报 CI 下界是否 > 0（优于），以及每个未见类型各自的结果 |
| 终点 3 | 时间表是否必要 | 主方法 − 不带时间表的 pipeline 的配对 CI 下界 > 0；报告选中非对角对的比例 |
| 终点 4 | 多见几族是否有用 | 主方法（16 个训练类型）− 跨type1c 冻结的 pipeline（只见过学习型）的配对差与 CI，按未见类型逐个与合并报告 |
| 终点 5 | 切换时刻 | 自学切换时刻 − 固定第 500 步切换的配对差与 CI；按类型列出选中时刻的分布 |
| 终点 6 | controller 信息 | 主方法 − 不看 controller 的 critic 的配对差与 CI |
| 终点 7（描述性） | 选出了什么 | 每个未见类型选中的 η_前期 与 η_后期 的分布（三个分量的散点图与直方图，标出 η13、η10），到 η13、η10 的距离，"η_后期 − η_前期"与 η10 − η13 的方向余弦；相对已评估动作中按 state 交叉拟合 oracle 的平均与最大遗憾；特征落在训练范围之外的比例；nominal only → safety baseline 的逐条转移表 |

**训练类型在新评估 state 上（DEVELOPMENT 级别，研究者见过这些类型）：** 42 个非学习型与保守强化学习 controller，用留一类型模型，报告主方法相对 safety baseline、两种人定时间表、另一种时间表 pipeline、不看 controller 的 critic 与跨type1c 冻结 pipeline 的差，按类型与族分开。

**失败归因（事后诊断，DEVELOPMENT）：** 对每个未见类型，用事后诊断标签把"没有修好"拆成下列原因，逐类型写进 `FAILURE_ATTRIBUTION.md`。确认性结论只来自终点 1–7，归因只解释原因。

| 原因 | 判定依据 |
|---|---|
| ① 该类型在 safety baseline 下基本不卡 | 能力门与新评估 state 上 safety baseline 成功率 ≥ 98% |
| ② 统计功效不足 | 点估计不差于对照，但 CI 跨过判定线 |
| ③ 两 η 时间表这种修法对该类型无效 | 诊断标签中 26 个动作的样本内最佳与两种人定时间表都显著低于"修好"（报告具体数值，不设单一阈值） |
| ④ 生成器提议没有覆盖好的 η | 人定时间表明显成功，但诊断标签中提议构成的全部动作都明显更差 |
| ⑤ 候选中有好的动作，critic 没挑到 | 诊断标签中样本内最佳明显高于主方法选中的动作；同时报告 critic 在这 26 个动作上的四种排序准确率，与训练类型留一检验中的水平对照 |
| ⑤a 其中：controller 特征越界 | 该类型特征落在训练范围之外的比例 |
| ⑤b 其中：critic 没用上 controller 信息 | 主方法与不看 controller 的 critic 选中动作的一致率，以及二者成功率之差 |
| ⑥ 实现或执行链问题 | 复现门、单元测试与执行记录 |

"明显"按配对 bootstrap 的 CI 判定：差的 CI 不含 0 记为明显。诊断标签只有 8 个 state、seed 0，归因结论注明证据等级。

# 七、执行顺序

1. 完整性审计（开始，含并行任务目录清单与 `squeue`）→ 复现门 → 三个未见类型的实现与单元测试 → 开工记录与收据（state 划分与种子、训练动作抽样、规模上限、未见类型清单与网格、评估范围的抽样顺序、主方法判定规则）。
2. Transformer 行为克隆策略训练（GPU）。
3. 未见类型：能力门（只输出布尔值与 safety baseline 成功率）→ 机制探测 → 特征。
4. 训练类型：精确 preflight 与训练 state 补缺 → 特征复用核查 → 可选空间报告（只用训练 state 标签：对角元与非对角对的样本内事后最佳，按族分开）→ 16 折留一类型检验与留一族检验 → 主方法判定 → 冻结。
5. 新评估 state：提议 → 冻结模型选择并留收据 → 标签（未见类型优先）→ 终点 1–7 写入并留收据 → 事后诊断标签 → 失败归因。
6. 完整性审计（结束）→ 交付。

任何阶段被挡住时，把受影响的类型记为缺口并写明原因，继续其余部分。未见类型不得由其他类型顶替。只有源码保护、DB 规则或复现门对全部类型都无法满足时才整体停下。

# 八、结论映射

| 观察结果 | 结论 |
|---|---|
| 终点 1 在每个需要修正的未见类型上都通过，终点 2 也通过 | 主线能泛化到未见的黑盒 controller：两个 η 与切换方式都由 pipeline 学出，修好死锁且不输人定时间表 |
| 终点 1 通过，终点 2 不通过 | 主线能修好未见 controller 的死锁，但学出的两 η 还不能完全替代人定时间表；看终点 7 的遗憾与选中 η 的分布 |
| 终点 1 只在部分未见类型上通过 | 泛化只到部分族；按族报告，并对照留一族检验，看失败的类型与哪个训练族最近 |
| 终点 4 为正 | 训练中见过更多族对未见 controller 有用，支持"见过很多种，再零样本泛化"的主张 |
| 终点 4 约为零或为负 | 加入非学习型训练类型没有帮助未见类型；泛化主要来自生成器与时间表结构本身 |
| 终点 6 为正 | controller 信息第一次在未见类型上起作用 |
| 未见类型多数"不需要修正"，且主方法不伤害它们 | 主线对不需要修正的 controller 是安全的；修正有效性的证据只来自需要修正的类型，如实写明 |
| 失败归因为 ③ | 不是泛化失败，是两 η 时间表这种修法对该类型无效；下一步需要扩大动作空间 |
| 失败归因为 ④ | 生成器只看 state、不看 controller，提不出该类型需要的 η；下一步让生成器以 controller 为条件，或启用时间表生成器 |
| 失败归因为 ⑤ | 修法可行、提议也够，是 critic 跨类型挑错了；下一步改 critic（看 ⑤a、⑤b 决定是扩大训练类型覆盖，还是加强 controller 分支） |
| 失败归因为 ① 或 ② | 不构成泛化失败的证据；分别需要会卡住的未见类型或更多 state |

# 九、交付

按共同协议第五节交付，放在本轮目录。另外：
- 动物园清单（每个 controller 一行：类型、族、训练或未见、配方、checkpoint 或参数、能力门结果、是否需要修正）；三个未见类型的实现、单元测试、训练 config、训练曲线、checkpoint 与收据；复现门报告；
- 训练动作抽样清单与收据；特征复用核查；可选空间报告；`ITERATION_LOG.md`；critic 的代码、config、留出曲线、四种排序准确率、checkpoint 与收据；主方法判定记录；`FROZEN_PIPELINE.json` 与选择收据；
- AXIS_TABLE：每个类型一行，另加"未见类型合并""训练类型合并""学习型""优化型""搜索型"各一行（新评估 state 部分没有学习型训练类型，对应行写 N/A 并注明原因）；列为主方法、另一种时间表 pipeline、不带时间表的 pipeline、不看 controller 的 critic、跨type1c 冻结的 pipeline、两种人定时间表、safety baseline、nominal only、已评估动作中的 oracle，分"训练 state 留一类型"和"新评估 state"两部分；单元格为成功率 % [95% CI]；另附碰撞率、超时率、平均与最大遗憾、伤害数、选中非对角对的比例、证据等级；
- 每个 (controller, 新评估 state) 的选中动作（两个 η 的完整数值与切换时刻）、各已评估动作的成功数与遗憾的明细表；
- 事后诊断标签的清单与收据、`FAILURE_ATTRIBUTION.md`；
- 终点 7 的图；每个未见类型一个典型 state 上 safety baseline、人定时间表与 pipeline 选中动作的轨迹静态图；
- REPORT.md：第一段先写完整性审计是否通过，再一句话回答"主线能否泛化到没见过的 controller"，然后给出终点 1–6 的结论与关键数字，以及每个未见类型的一句话总结（safety baseline 下是否卡住、pipeline 选了什么、是否修好；没修好时归因是 ①–⑥ 中哪一类）；
- HANDOFF.md（不依赖本对话也能读懂），写明下一轮可以直接复用的部件与路径。

**资源：** 用户全部任务合计不超过 18 个 CPU shard、2 个 GPU shard。开工前与每次提交前都要用 `squeue` 查看总占用。**不做 git commit。**

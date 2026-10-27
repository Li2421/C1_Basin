# 跨场景 pipeline 检验 v1（去重场景，生成器 + critic）：实现说明

**写作约定：** 方法、场景、控制器和候选第一次出现时写全称，括号里给出文件中的代号，代号只用于定位文件与身份；报告也照此书写。本说明里"pipeline"只指我们的主方法：生成器 q(η | h, c) 为每个状态提出候选修正，critic Q(h, c, 控制器, η) 预测该状态、该控制器执行该候选的成功概率，取最高者执行。

本文件取代同名的 18 场景版（那一版未运行）；运行目录名不变。

## 零、开工条件、源码与仓库保护（任何步骤之前）

- **开工条件：** `diagnostics/cross_scene_correction_v1/COMPLETE.json` 存在，且其 `INTEGRITY_COMPARE.json` 的 `passed` 为真；否则停下报告，不建目录。
- 本轮所有新代码与输出只放 `diagnostics/cross_scene_pipeline_v1/`。已有源码、`docs/`、`AGENTS.md`、其他 `diagnostics/` 目录（含 `cross_scene_correction_v1/`、`cross_scene_homogeneity_audit_v1/`、`cross_scene_prospective_v1/`、`crosstype1b_generator_critic_pipeline_v1/`、`orthoflow3_generator_critic_v1/`）、`datasets/`、`bottleneck_family/`、MAC-Flow 官方仓库（`../01_MACFlow_Baseline_Reproduction/MACFlow_Official`）、虚拟环境一律只读；不安装、不升级任何包。
- 复用只用只读 import，或复制到本轮目录并记录 sha256；需要不同行为时写新文件、子类或新函数，不在运行时给旧模块打补丁（JAX 精度恢复除外）。旧运行入口有写入副作用的（`cross_scene_correction_v1` 的 `experiment.py` 会在自身目录建 TMPDIR/XDG/CUDA 缓存；`common.py` 每次读证据都会追加该轮的 `EVIDENCE_INPUTS.jsonl`；本地编译的冲突搜索内核写死了查询缓存根目录），一律不在原目录运行：按该轮 `build_backend.py`、`build_extensions.py` 的做法在本轮目录重建隔离副本（只改缓存根目录），通过第五节第 1 步的逐位重放后才能使用。
- git 只用只读命令：不 commit、stash、checkout、reset、restore、clean、add（Basin_C1 与 02_C1_Toy_GiveWay 两个仓库都有用户未提交的工作）。开工和结束各记录两个仓库的 `git status --porcelain`、`git diff` 的 sha256，以及所有复用文件的 sha256，两次必须一致；本轮目录自身那一行、开工时登记的并行任务目录（例如正在运行的 `cross_scene_prospective_v1/`）、`docs/` 下新增的说明文件（`prompt_*.md`）单列，不算失败。
- 共享数据库：允许 (1) 官方 `cache_writer.append_journal` 写默认的 `shared_rollout_db/journals/<本轮实验 UID>/`；(2) 官方单进程 merger；(3) 新身份登记：单进程、官方 `shared_rollout_db.src.rollout_db` 的 connect/transaction，只对新的 experiment/scenario/state/controller_config/eta 行做 INSERT，遇到 UID 内容冲突即停，登记前后已有行的逐表哈希不变。禁止改 `cache_writer.ROOT`、禁止复制数据库。预检与缓存规则见 `shared_rollout_db/README.md`。
- 开工时列出最近 24 小时有写入的其他 diagnostics 目录与他人正在运行的作业目录；它们的变化单列，不算本轮失败（前提是本轮既不 import 也不写它们）。
- 遇到与本说明矛盾、或要求的接口不存在：停下报告，不改源码。下文点名的接口与文件均已核实存在。

## 一、要检验的问题与历史依据（先查后做）

**问题：** 去掉"需要同一组修正"的重复场景后，在剩下的 11 个场景上，pipeline 对完全没见过的场景（留一场景）是否零样本优于"所有场景用同一组最好的修正"；它按状态、按控制器挑，是否优于按场景挑的近邻模型。

**来源（只读，写进 `HISTORY_CHECK.md`：路径、执行语义是否一致、可复用范围、仍缺证据）：**
1. `diagnostics/cross_scene_correction_v1/`：18 个场景 × 49 个候选的全表标签 `LABEL_TABLES.json`（每个场景的 `y[控制器, 状态, 候选]`，种子 0）、`SCENES.json`（场景定义、状态、`features`）、`models.py`（`select`）、`LOSO_CHOICES.json`、`STATISTICS.json`、`REPORT.md`。18 个场景上按场景选的主模型（岭回归）减全局最佳单一修正 +0.5 个百分点 [−4.4, +5.9]，未通过。
2. `diagnostics/cross_scene_homogeneity_audit_v1/`（看过结果之后的探索，只读）：18 个场景在"需要哪组修正"上高度重复；去重后 11 个场景上留一场景，按场景选的近邻比同池全局最佳 +8.0 [+1.8, +15.5]、岭回归 −2.0 [−10.4, +6.1]；同池全局最佳在各折之间于扩展范围候选 sE29 与 sE05 之间来回变化，改与 17 场景池的全局最佳比时近邻只剩约 +4.9。本轮的 pipeline 结果与这些数对照，但它们不是已有结论。
3. pipeline 的已有实现：`diagnostics/orthoflow3_generator_critic_v1/train_evaluate.py`（`Generator`、`dist_params`、`eta_mean`、`eta_from_noise`、`log_prob`、`stable_int`、`train_generator`、`train_critic`、`freeze_generator_proposals`）；`diagnostics/crosstype1b_generator_critic_pipeline_v1/critic_model.py`（看状态、看控制器的 `Critic`，模式 `full` / `no_controller` / `no_state`）。环形交换上的结果（`crosstype1b_generator_critic_pipeline_v1/REPORT.md`）：相对所有控制器共用的按进度切换时间表 +0.72 [−1.14, +2.87]，未通过。
4. 文献线 v2 的按状态 critic（12 维物理状态、留一类型）：在四个文献场景上与训练集最佳固定修正持平或略差（`diagnostics/baseline_literature_scenes_distributed_v2/`）。

本轮只为 pipeline 自己提出的新候选跑新 rollout；凡是从 49 个候选中选的，一律直接用已有标签计分，不重跑。

## 二、统一执行语义（全部沿用 `cross_scene_correction_v1`，不改）

- 单积分器、分布式安全投影、接触门控修正、成功判据（全部机器人在时限内到达且无碰撞；timeout 不等于死锁；碰撞按机器人/静态/移动分类；数值失败记 unknown，保留不重跑）、随机数规则、每个场景的合格控制器实例（`SELECTED_CONTROLLERS.json`），均与上一轮完全相同；只跑种子 0；只用固定修正，不含两段切换。
- 新候选的请求按上一轮同一场景已有请求的构造方式生成（对穿与瓶颈场景用 `scenes.make_new`，文献场景用上一轮 `main_CASES.json` 中该场景请求的构造方式），只换 η。

## 三、场景集、输入、模型与训练（全部预先写定，不按留出结果调整）

**场景集（去重，零 rollout）：**
- 规则（与 `docs/prompt_cross_scene_prospective_v1_20261025.md` 第三节相同）：两个场景 49 维类型等权成功率向量的 Spearman 相关（并列取平均秩）≥ 0.9，且存在一个候选在两个场景上距各自最佳都不超过 2 个百分点，算重复；贪心地每次取相关最高的一对，去掉与其余剩余场景平均相关更高的那个（相等时去掉在 `SCENES.json` 中排在前面的那个），直到没有重复对。
- 必须得到以下 11 个场景：窄通道（`lit_narrow_passage`）、拥挤随机（`lit_crowded_random`）、圆形五机（`lit_circle_five`）、八机静态障碍（`lit_swap8_obstacles`）、对穿 4 机（`xsc_swap_n4`）、对穿 12 机（`xsc_swap_n12`）、对穿 8 机加密（`xsc_swap_n8_dense`）、缺口 4 机（`xsc_gap_n4`）、缺口 6 机（`xsc_gap_n6`）、双瓶颈（`xsc_double_n4`）、错位口（`xsc_offset_n4`）。不同就停下报告。被去掉的 7 个场景不进入主检验的训练与测试。

**输入（只用场景定义与初始构型，不用任何结果）：**
- 环境描述 c：12 维，即上一轮 `scenes.features` 的前 12 项（人数、机器人半径、限速 × dt / 半径、时限 / 最长直线行程时间、密度、墙段数、最小开口 / 直径、串联墙数、是否有移动障碍及其两项比值、是否对径交换）。
- 状态编码 h：6 维，与 `scenes.features` 后 6 项同一公式，但对单个状态计算、不对状态取平均（最近接近距离的最小值与均值、每台机器人的冲突对数、每台机器人的路径相交对数、迎面冲突比例、中点拥挤度，距离除以半径）。
- 控制器：8 类的独热编码——集中式模型预测路径积分控制（`centralized_mppi`）、集中式随机采样模型预测控制（`centralized_sampling`）、冲突搜索多机规划（`conflict_based_search`）、交叉熵方法模型预测控制（`cross_entropy`）、分布式交叉熵方法模型预测控制（`distributed_cross_entropy`）、分布式模型预测控制（`distributed_shooting`，实例 `DM0`）、梯度下降模型预测控制（`gradient_mpc`）、优先级时空 A* 规划（`prioritized_astar`）。
- 标准化只用该折训练场景的训练状态统计量。
- η 的范围：扩展范围盒 [0, 2] × [−1.5, 1.5] × [0, 1.5]（49 个候选全在其中，含 sE29 与 sE05）；生成器与 critic 的 η 归一化都用这个盒子的中心与半径。

**生成器（复制 `orthoflow3_generator_critic_v1` 的 `Generator`，三个场景专用输入层换成一个 96 维输入层，其余不变）：** h → 96，c → 32，拼接后 128 → 64 → 6，压缩高斯（σ = 0.025 + 0.275·sigmoid）。不看控制器。
- 正样本：在某个状态上，类型等权种子 0 成功率达到该状态 49 个候选最大值的候选（并列全收；最大值为 0 的状态不参与）；负样本：该状态上所有类型都失败的候选。
- 损失、优化与停止全部沿用 `train_generator`：正样本负对数似然 + 0.25 × 状态内 softplus(0.5 + log q(负) − log q(正))；AdamW 学习率 5e-4、权重衰减 1e-4、梯度裁剪 5；每步每个训练场景抽 64 个状态；最多 3000 步，每 100 步验证一次，连续 8 次不改善且已过 1200 步就停。

**critic（复制 `crosstype1b_generator_critic_pipeline_v1/critic_model.py` 的 `Critic`，结构不变）：** h → 96、c → 32、η → 32、控制器 → 32，拼接后 128 → 64 → 1（成功概率的对数几率）。
- 训练样本是（状态, 控制器类型, 候选）三元组，标签为种子 0 的成败；每步每个训练场景抽 96 个，先均匀抽状态、再均匀抽该状态的合格类型、再成功与失败各半抽候选。
- 损失、优化与停止沿用 `train_critic`：二元交叉熵；AdamW 学习率 1e-3、权重衰减 1e-4、梯度裁剪 5；最多 4000 步，每 100 步验证一次，连续 10 次不改善且已过 1500 步就停。
- 消融（只在 49 个候选里选，零 rollout）：不看控制器（去掉控制器输入）；只看场景（去掉 h，保留 c、控制器、η）。以新子类实现，不改原类。

**验证与种子：** 每个训练场景中状态序号除以 4 余 3 的状态作内部验证集，只用于早停与选种子；种子 17、23、41 各训一次，按内部验证损失取最小者（并列取较小种子）。留出场景的任何标签都不得进入训练、早停、选种子或标准化。

**pipeline 的选择规则：** 对留出场景的每个（状态, 控制器类型）：候选集 = 生成器均值 + 4 个样本（噪声由 `numpy.random.default_rng(stable_int("cross-scene-pipeline-proposals", 种子, 场景, 状态序号))` 生成，`stable_int` 与抽样方式同 `freeze_generator_proposals`）∪ 49 个候选；取 critic 预测最高者（并列按 49 个候选的规范顺序优先，再按生成器均值、样本 0–3）。

**对照（每折与 pipeline 用同样的训练场景，用复制的 `models.select` 逐字不改地重算）：** 同池全局最佳单一修正（主对照）；同池近邻按场景选择；η13；仅安全过滤；每个场景的样本内最佳（只作乐观上界描述）。另取上一轮 `LOSO_CHOICES.json` 中同一留出场景的全局最佳（用 17 个场景选出，大多为 sE29）作更强的次要对照。

## 四、检验与冻结

1. **留一场景（主）：** 11 个去重场景逐个留出，用其余 10 个训练。
2. **留一族（去重场景内）：** 用文献的窄通道、拥挤随机、圆形五机、八机静态障碍和 3 个对穿场景训练，测 4 个瓶颈场景（缺口 4 机、缺口 6 机、双瓶颈、错位口）；用 3 个对穿场景和 4 个瓶颈场景训练，测 4 个文献场景。
3. **18 场景对照（次要，零 rollout）：** 在全部 18 个场景上做留一场景，只比较 critic 在 49 个候选里的选择与上一轮同折的全局最佳单一修正，用来和上一轮的甲直接对照。
- 每一折训练完成后、生成任何新 rollout 之前，把该折的模型、生成器提案、全部选择写 sha256 进 `FOLD_FROZEN/<折>.receipt.json`；全部折冻结后再统一跑新 rollout。冻结后不改任何选择。

## 五、执行流程

1. 开工条件与完整性记录（第零节）；历史检索与登记（第一节）；逐位重放上一轮已存的双瓶颈（`xsc_double_n4`）、对穿 12 机（`xsc_swap_n12`）、拥挤随机各至少 2 局，结果与轨迹逐位一致才继续。
2. 去重核对与输入构造（第三节），写 `SCENE_SET.json`、`INPUTS.json` 与 sha256。
3. 逐折训练与冻结（第三、四节）。
4. 新 rollout：只跑 (a) pipeline 选中的生成器提案，(b) "只用生成器均值"这一描述性变体在每个留出（状态, 类型）上的执行；种子 0。去重场景共 796 个（状态, 类型）格子，留一场景与留一族合计新 rollout 约 2,700 次以内。官方预检后只跑真正缺失的请求；worker 写 journal，阶段结束单一 merger 合库。先做计时试跑（每个场景 2 个状态 × 2 个提案），写 `PILOT_COST.md` 后照常继续。
5. 统计（第六节），然后数据库核对（所有请求在库、身份与结果一致、轨迹哈希一致、无未合并 journal）与结束完整性记录。

## 六、判定（预登记）

- 指标：留出场景上所选候选的成功率，场景内控制器类型等权，场景之间等权；unknown 计失败（另报上界）。区间用 5000 次成对自助法，按场景 → 控制器类型 → 状态分层重抽。
- **主判定：** 去重 11 个场景留一场景时，pipeline 减同池全局最佳单一修正的场景等权平均差，95% 区间下界 > 0。
- 次要结果（照报，不改变主判定）：
  1. pipeline 减上一轮 17 场景池的全局最佳（更强的对照）。
  2. pipeline 减同池近邻按场景选择（按状态、看控制器，是否比按场景更好）。
  3. 只用 critic 在 49 个候选里选、只用生成器均值，各自减同池全局最佳。
  4. 消融（49 个候选内，成对）：完整 critic 减不看控制器的 critic；完整 critic 减只看场景的 critic。
  5. 留一族两个方向：pipeline 减同池全局最佳。
  6. 18 场景对照：critic 选择减上一轮同折全局最佳。
  7. 稳健版本：逐场景差值的中位数、两端各去 1 个场景后的截尾均值、Wilcoxon 符号秩检验、符号检验。
  8. 描述：每折同池全局最佳是哪个候选；pipeline 选中生成器提案的比例；同一场景内不同状态选到不同候选的比例、不同控制器类型选到不同候选的比例；每个场景的方法对比表。
- 只说"在本修正族、本控制器集与本安全层下，对这 11 个去重场景"的结论；去重是在看过 18 个场景的标签之后定的，报告中写明这一点；不外推到未测拓扑。

## 七、资源与调度

- 用户全部任务合计不超过 18 个 CPU 份额、2 个 GPU 份额（临时上调由用户决定；与正在运行的 `cross_scene_prospective_v1` 合计）。训练是小网络，CPU 即可。
- Slurm worker 时限约 3 小时，在时限前约 55 分钟停止领取新任务组；每次提交后立即确认作业处于 RUNNING，几分钟后再查一次，不能只看到"已排队"。
- 调度程序自查：本人作业排队超过 15 分钟而 CPU 空闲，就撤销并以更短时限重交，写日志。**不设"若干分钟没有新结果就撤销重交"的规则**（用户 2026-10-25 已取消）；长仿真用上一轮的操作检查点在时限交接时续跑。
- 只有当其他用户的排队作业确实缺 CPU 或内存时才让出；对方在等 GPU 份额时不让出。
- 任务组按预计耗时从长到短排队，单组约 10 分钟。

## 八、交付

- `REPORT.md`（中文，第一段给出主判定、pipeline 减同池全局最佳与减 17 场景池全局最佳的差值和区间、减近邻的差值、留一族瓶颈方向的差值）；每个留出场景的方法对比表；`AXIS_TABLE.md` 按泛化框架标注（跨场景；状态；控制器）；`HISTORY_CHECK.md`、`SCENE_SET.json`、`INPUTS.json`、各折冻结记录、训练记录（每个种子的验证曲线与所选步数）、`PILOT_COST.md`、`GAPS.md`、数据库核对、开工与结束完整性记录。
- 无视频要求。

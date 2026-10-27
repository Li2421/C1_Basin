# critic 训练池补充 v1：补 22 个训练场景，用新配方重训 critic，只在 8 个异质场景上验证：实现说明

**写作约定：** 方法、场景、控制器和候选第一次出现时写全称，括号里给代号，代号只用于定位文件；报告照此书写。候选 η 是三个数（朝目标, 侧移, 互斥），提到某个候选时同时写出这三个数。"主方法"只是角色。本轮只训练与评估 critic，不训练生成器、不跑生成器提案。

**名称表（报告一律用这些名称）：**

| 名称 | 含义 |
|---|---|
| 新 critic·近邻输入（本轮主方法） | 第五节的配方：对任意 (状态, 控制器, η) 预测成功概率；输入除场景描述、状态特征、控制器、η 外，另加"场景描述最近的 3 个训练场景在该 η 附近的实测成功率"；场景等权交叉熵、10 个种子取平均；一个场景选"该场景全部状态与 3 个控制器平均预测成功率最高"的那组 η |
| 新 critic·无近邻输入 | 同上，去掉近邻输入 |
| 新 critic·原网络结构 | 上一轮 critic 的网络结构（状态 96、场景 32、η 32、控制器 32 → 128 → 64 → 1），去掉先验锚定与修正压缩，其余按第五节训练 |
| 新 critic·补充前训练池 | 主方法配方，只用第四节的已有 14 个场景版本训练（不含本轮新场景） |
| 新 critic·逐类加入 | 主方法配方，已有 14 个场景版本 + 本轮某一类新场景 |
| 一个全局 η（训练池里最好） | 49 个候选里在训练池场景上平均成功率最高的一组，所有验证场景都用它 |
| 抄最近 k 个训练场景（k = 1, 2, 3） | 按场景描述找最近的 k 个训练场景，取它们 49 个候选平均成功率曲线的平均，选最高的一组 |
| 每个场景自己的最好 η（参照） | 每个核对状态用同场景其余核对状态挑 49 个候选里最好的，再在该状态上计分；用了验证标签，只作上限参照 |
| 照抄对的训练场景（覆盖上界，参照） | 对每个验证场景，在所有训练场景里找"它的最好 η 放到该验证场景上成功率最高"的那个；用了验证标签，只作参照 |
| 学习选 η v2（上一轮，参照） | `hetero_scene_pipeline_v1/REPORT.md` 的已有数值，不重算 |

## 零、开工条件、源码与仓库保护

- **开工条件：** `diagnostics/hetero_scene_pipeline_v1/COMPLETE.json` 存在且其 `INTEGRITY_COMPARE.json` 的 `passed` 为真；否则停下报告。开工时把最近 24 小时内被写过的其他 `diagnostics/` 目录登记为并行任务目录，只看 `os.stat` 元数据，结束时单列其变化，不因它们判失败。
- 本轮新代码与输出只放 `diagnostics/critic_pool_supplement_v1/`（实验名 `exp_critic_pool_supplement_v1`，Slurm 作业名前缀 `xcs_`）。其他所有目录、源码、`docs/`、`AGENTS.md`、`bottleneck_family/`、虚拟环境只读；不安装任何包。
- **复用（复制到本轮目录并记录 sha256，不 import 旧目录模块、不在旧目录运行）：** 执行图、场景构造与特征代码取自 `hetero_scene_pipeline_v1/`（`scenes.py` 的 `make_new` 与 `features`、`scene_environment.py`、`runtime_new.py`、`controllers_new.py`、`backend/`、`descriptors_v2.py`，以及 worker、预检、合库代码），只改实验名、输出目录、作业名前缀和场景清单；执行语义逐字不改。复制后先逐位重放 `hetero_scene_pipeline_v1` 已存的 2 局完整记录（十字路口半宽 0.20 与方形场地 10 机各 1 局，快控制器）；该目录若没有可重放的完整记录，改为重放 `eta_diverse_scene_test_v1` 已存的 2 局（与上一轮相同）。一致才继续。
- git 只用只读命令；开工与结束各记录两个仓库的 `git status --porcelain`、`git diff` 的 sha256 与复用文件 sha256，两次必须一致（本轮目录、并行任务目录、`docs/` 下新增的 `prompt_*.md` 单列）。
- **共享数据库：** 读用只读连接；写只走官方设计阶段的 `INSERT OR IGNORE`、默认路径的 `append_journal`、单进程 merger；只对新身份做 INSERT，UID 冲突即停，已有行逐表哈希不变；开工断言 `shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/` 之下。rollout 前遵守 `shared_rollout_db/README.md` 的预检与缓存规则，只跑真正缺失的请求。
- 遇到与本说明矛盾、或要求的接口不存在：停下报告，不改源码（第八节的非主线失败除外）。

## 一、背景

- **上一轮结论：** 8 个异质场景（`hetero_scene_pipeline_v1`）作为场景集成立（全局缺口 +11.49 个百分点 [+6.95, +18.40]），但学习选 η v2 在 8 折里选的都正好是"一个全局 η"（53.91%）。原因在 critic 的训练方式：每折只有 7 个训练场景；输出 = "训练场景在该候选上的平均成功率"再加有界修正，修正还乘一个由内部留一选的系数（4 折选成 0）；早停和挑种子看的是同一批训练场景的其他状态，从没考过新场景；超范围时退回全局 η。
- **用户决定（2026-10-26）：** 这 8 个场景只作验证集，任何训练、标准化、近邻特征、超参数选择都不用它们；之前的数据集和其他场景有用的都加入 critic 训练；先把 critic 修好，再做"生成器出 16/8 个、critic 只在生成的 η 里选"的主线。
- **对话中助手的零仿真开发（临时目录，非证据）：** 训练池 = 共享数据库里执行语义一致的 14 个场景版本 + 助手早先探索的 15 个场景（十字路口、方形场地、时间紧；与正式重跑逐格吻合 98.7–100%，但不在共享数据库里），共 29 个场景、约 4 万格。在 8 个验证场景上（核对状态）：一个全局 η 48.7%；抄最近 2 个训练场景 64.6%；新 critic·无近邻输入 63.0%；新 critic·近邻输入 64.3%（减全局 η +15.6 [+10.9, +20.6]，减抄最近 2 个 −0.3 [−3.7, +3.1]）；照抄对的训练场景 74.5%；每个场景自己的最好 η（样本内）75.0%。只用官方数据（不含探索场景）时 critic 只有 54.2%。随机抽 7/14/22/29 个训练场景时 critic 为 54.8/57.4/62.9/63.0。
- **仍然错的 4 个验证场景及原因：** 八机静态障碍（`lit_swap8_obstacles`）要 sE22（0.55, −0.20, 1.42），critic 选 sE13（0.45, 1.34, 1.03）只有 31%——训练池没有带静态障碍的对穿；对穿 10 机（`xsc_swap_n10`）与八机无障碍（`lit_swap8_free`）要 sE29（0.34, 0.55, 1.20），critic 选 sE13——训练池只有 4 个对穿场景，最好 η 随机器人数跳变（对穿 4 机 sE29、对穿 12 机 sE13）；圆形五机（`lit_circle_five`）要 sE27（1.04, 0.25, 0.26），场景描述远在训练范围外（"每步移动量 / 半径"的对数为 −4.67，训练池最小约 −2.0），训练池没有大机器人封闭场地。两个十字路口与方形场地 10 机在有同类探索场景时已选对。
- **本轮做什么：** 用正式流程补 22 个训练场景（第三节），其中三类专补上述缺口，另外两类是探索里的十字路口、方形场地的正式版本（替代非正式探索数据）；按第五节配方重训 critic，只在 8 个验证场景上验证。

## 二、执行语义（同 hetero_scene_pipeline_v1，不改）

- 单积分器、分布式安全投影、接触门控修正、成功判据、随机数规则同 `cross_scene_correction_v1`；只跑种子 0；只用固定修正。
- **控制器：** 只用 3 个快控制器：集中式随机采样模型预测控制（`centralized_sampling0`）、梯度下降模型预测控制（`gradient_mpc0`）、分布式模型预测控制（`DM0`），载荷取自 `cross_scene_correction_v1/SELECTED_CONTROLLERS.json` 的对穿 6 机行（同上一轮）。
- **候选：** 全部 49 个（`best_correction_search_v1/ACTIONS.json` 的精确值，顺序 `safety`、`eta0`–`eta15`、`sE00`–`sE31`）。
- 新场景名必须以 `xsc_` 开头（本轮用 `xsc_tr_`）：快控制器只有在 `xsc_` 场景才走带几何的名义控制器（`scene_environment.NewNominal`）。用复制的 `scenes.make_new` 构造请求（版本 `cross_scene_si_v1`）。有墙的场景每个状态都带 `bottleneck_config`（新场景的墙体只能来自 `bottleneck_config`；`NewEnv` 不读场景定义里的 `walls` 字段）；无墙场景的状态不带它。五层报告与能力门（状态 0–3 的单机器人版本，种子 0 与 1，3 个控制器都 ≥ 95%）同上一轮。

## 三、新训练场景（22 个，构造必须逐项照做）

记 `SEED_TR = 2026102661`。`LiteratureConfig(名称, N, 机器人半径, 0.05, 速度, max_steps)`，其余字段取默认值；`max_steps = ceil(f × 最长直线行程时间 / 0.05)`，最长直线行程时间 = 本场景全部状态里"机器人到目标距离"的最大值 / 速度。速度除 D 类另行说明外为 0.5。每个场景 8 个状态（序号 0–7），速度为 0。场景定义的 `walls` 字段取 `Geometry(Config(...)).walls`（无墙场景为空表；只用于特征计算），场景特征字段 `serial_walls = 0`、`moving = False`，其余按各类给出。助手已在临时目录核对：下面每一类的 `Config` 都能通过校验，每个场景都能采满 12 个有效状态，单机几何路径都存在。

**A. 无障碍对穿（4 个）：** `xsc_tr_swap_n05`、`xsc_tr_swap_n07`、`xsc_tr_swap_n09`、`xsc_tr_swap_n11`。同 `cross_scene_correction_v1/scenes.py` 的对穿写法：N 台机器人，半径 0.05；未扰动位置为半径 N/8 的圆上均匀 N 点，目标为对径点；`rng = numpy.random.default_rng(numpy.random.SeedSequence([SEED_TR, N]))`，状态 0 不扰动，其余依次加 `rng.uniform(-0.1, 0.1, (N, 2))`；f = 4.0；无墙；`free_area = None`，`min_opening_over_diameter = 10`，`diametric = True`。对穿 8 机故意不建（与验证场景八机无障碍几乎相同）。

**B. 带静态障碍的对穿（5 个，补八机静态障碍）：** `xsc_tr_obst_n06_k3`、`xsc_tr_obst_n08_k4`、`xsc_tr_obst_n10_k5`、`xsc_tr_obst_n06_k6`、`xsc_tr_obst_n08_k8`（N 台机器人、K 个障碍）。机器人、目标与扰动同 A，但 `rng = default_rng(SeedSequence([SEED_TR, N, K]))`；f = 4.0。记 R0 = N/8，H = max(2.0, round(R0 + 0.9, 2))，间隙角 θ_i = 2πi/N + π/N（i = 0, …, N−1）；K = N 时每个间隙放一个障碍，K = N/2 时只放 θ_0, θ_2, θ_4, …。障碍是半边长 a = 0.07 的正方形，中心在半径 ρR0（ρ = 0.75）、角度 θ 处，`extra_rectangles` 每项为 (cx − a, cy − a, cx + a, cy + a)。`bottleneck_config = Config(num_agents=N, half_length=H, half_height=H, barrier_x=(H − 0.3,), openings=(((0.0, 2H − 0.4),),), barrier_thickness=0.1, extra_rectangles=上述正方形, agent_radius=0.05, dt=0.05, max_speed=0.5, max_steps=400).to_dict()`（屏障藏在角落，只为通过 `Config` 校验）。`free_area = (2H)² − 0.04 − K(2a)²`；`min_opening_over_diameter = min(10, (2ρR0·sin(π/K) − 2a) / 0.1)`；`diametric = True`。设计仿照八机静态障碍：障碍在相邻机器人之间、起点圈内侧，不挡直线路径（核对：直线路径离墙与障碍至少 0.186）；测试 v1 里中央大障碍让快控制器单机能力不过，这里避开了那种放法。**能力门不过时**，改用 a = 0.05、ρ = 0.85 重建一次（场景名加后缀 `_mild`，其余同），再不过就记"控制器不适用"并去掉。

**C. 方形场地（6 个，正式版，夹住验证场景方形场地 10 机）：** `xsc_tr_box_n05_r012_f25`、`xsc_tr_box_n05_r018_f25`、`xsc_tr_box_n07_r019_f25`、`xsc_tr_box_n08_r015_f25`、`xsc_tr_box_n08_r015_f30`、`xsc_tr_box_n12_r011_f25`（N, 半径 r, f）。照 `hetero_scene_pipeline_v1/scenes.py` 方形场地 10 机的写法（同一 `Config`：半场 1.2，屏障 0.95，开口 (0.0, 2.0)，`Config` 半径字段 0.05；`clear = r + 0.035`；起点两两间距 > 2r + 0.1、最多尝试 2000 次；第 i 个目标与第 i 个起点距离 ≥ 1.0、目标两两间距 > 2r + 0.1、最多尝试 4000 次；起点与目标都凑满才收下该状态），`rng = default_rng(SeedSequence([SEED_TR, N, round(100r), round(10f)]))`，凑满 8 个状态。`free_area = 2.4² − 0.04`，`min_opening_over_diameter = 10`，`diametric = False`。

**D. 大机器人方形场地（3 个，向圆形五机靠拢）：** `xsc_tr_bigbox_n05_r025_v050`（N = 5, r = 0.25, f = 2.4, 速度 0.5）、`xsc_tr_bigbox_n05_r025_v020`（同，速度 0.2）、`xsc_tr_bigbox_n04_r030_v012`（N = 4, r = 0.30, f = 2.4, 速度 0.12）。构造同 C，`rng = default_rng(SeedSequence([SEED_TR, N, round(100r), round(10f), round(100v)]))`；速度写入 `LiteratureConfig` 与 `max_steps` 公式，`Config` 的 `max_speed` 字段保持 0.5（只用于几何）。字段同 C。核对：三者的"每步移动量 / 半径"对数约为 −2.3、−3.2、−3.9，密度 0.17–0.20（圆形五机 −4.67、0.17）。新墙形（圆形围墙）本轮不做：现有执行代码不支持。能力门不过就记"控制器不适用"并去掉（不替换）。

**E. 十字路口（4 个，正式版，夹住验证场景的两种宽度）：** `xsc_tr_plus_w016`、`xsc_tr_plus_w025`、`xsc_tr_plus_w032`、`xsc_tr_plus_w055`（半宽 w）。照 `hetero_scene_pipeline_v1/scenes.py` 十字路口的写法（4 台机器人，半径 0.05，f = 4.0，同一 `Config` 只换 w），`rng = default_rng(SeedSequence([SEED_TR, 4, round(100w)]))`。`free_area = 3.2² − 4(1.6 − w)²`，`min_opening_over_diameter = w / 0.05`，`diametric = True`。半宽 0.20、0.40 是验证场景，不建；0.10 在探索中全部失败，不建。

**规模：** 22 个场景 × 8 个状态 × 3 个控制器 × 49 个候选 = 25,872 次新 rollout；能力门约 3,700 次单机器人运行。D 类速度低，单局较长（约 470–780 步）。

## 四、训练池与验证集

**已有数据（零新 rollout，只读抽取）：** 从共享数据库抽取下列 14 个场景版本（场景名以数据库 `scenario.name` 为准，只取 `metadata_json.config.num_agents` 与下表一致的版本；同名下 `num_agents = 1` 的版本是单机能力检查，不要；同名同机器人数有多个场景身份时合并，按筛选后的数据为准）：

| 场景版本 | 机器人数 | 预期格数（3 个快控制器） |
|---|---:|---:|
| 窄通道 `lit_narrow_passage_flow` | 3 | 3,882 |
| 二维拥挤随机 `lit_crowded_random_flow` | 2 / 4 / 6 / 8 / 10 / 12 / 14 | 96 / 96 / 96 / 3,185 / 96 / 96 / 769 |
| 八机移动障碍（修正版）`lit_swap8_moving_swap8_si_user_fix_20261023` | 8 | 7,296 |
| 对穿 4 机 `xsc_swap_n4_cross_scene_si_v1` | 4 | 1,841 |
| 对穿 12 机 `xsc_swap_n12_cross_scene_si_v1` | 12 | 1,800 |
| 八机密集对穿 `xsc_swap_n8_dense_cross_scene_si_v1` | 8 | 1,836 |
| 大机器人 `xsc_div_size_r014_cross_scene_si_v1` | 6 | 900 |
| 时间紧 `xsc_div_time_f20_cross_scene_si_v1` | 6 | 900 |

- 筛选：`controller_config.config_json` 中 `safety_layer == "distributed"`、`correction_basis == "orthoflow3_local_contact_gated"`、`correction_semantics == "contact_conflict_relative_gate_20261021"`，`nominal.controller` 为 3 个快控制器之一；`rollout.conflict_quarantined = 0`；η 取 `eta` 表的精确值（含 49 个菜单外的 η，例如往轮生成器提案与最佳修正搜索）。文献"flow"版本的 `sensing_radius`、`restoration` 字段为空，这与验证场景圆形五机标签所用的版本相同，按兼容处理；`paper_dynamics` 版本（修正方式不同）、集中式安全层版本、未修正的八机移动障碍旧版（`lit_swap8_moving_swap8_si`）一律不用。
- 一格 = (场景版本, 状态, 控制器, 精确 η)，标签为该格全部兼容 rollout 的平均成功（unknown 计失败），并记录次数。上表"预期格数"为助手开发时的计数，对不上只报告、不停。
- 场景描述与状态特征：同上一轮（复制的 `scenes.features` + `descriptors_v2.descriptor`）。数据库场景的静态字段（墙、`free_area`、`min_opening_over_diameter`、`serial_walls`、`diametric`、`moving`）取 `cross_scene_correction_v1/SCENES.json` 或 `eta_diverse_scene_test_v1/SCENES.json` 的同名定义（拥挤随机各机器人数版本共用 `lit_crowded_random` 的定义，只换 N）；配置字段（N、半径、dt、速度、max_steps）取数据库场景元数据；状态取数据库 `state` 表的 positions 与 goals（该版本全部状态）；`moving` 字段缺失按 False。
- 不进训练池：8 个验证场景的任何版本、任何控制器的数据；助手在临时目录的探索数据；门口场景（`lit_doorway_flow`，全部成功，没有排序信息）；其他控制器的数据（开发中作为辅助任务试过，没有帮助）。

**新场景：** 第三节通过能力门的场景，全部 8 个状态、3 个控制器、49 个候选。

**训练池清单：** 写 `TRAINING_POOL.json`：每个场景的来源、状态数、格数、不同 η 数、c′、平均成功率、是否本轮新场景；断言没有任何验证场景（按场景名前缀 `lit_swap8_obstacles`、`lit_swap8_free`、`lit_circle_five`、`xsc_swap_n6`、`xsc_swap_n10`、`xsc_het_` 检查）。

**验证集：** `hetero_scene_pipeline_v1` 的 8 个场景（八机静态障碍、对穿 10 机、八机无障碍、对穿 6 机、圆形五机、十字路口半宽 0.20 与 0.40、方形场地 10 机）。输入（c′、h、状态序号）取其 `INPUTS_V2.json`，标签取 `LABEL_TABLES.json`（3 个快控制器 × 49 个候选，全部已标注状态），核对状态取 `EVALUATION_STATES.json`（新场景 8–23、对穿 12–27、文献 16 个）。验证标签只在全部选择冻结之后用于计分。

## 五、critic 训练（预登记，配方固定，不在验证集上调参）

- **输入：** c′（12 维，尺度无关场景描述）；h（6 维，该状态的状态特征）；控制器独热（3 维，顺序 `centralized_sampling0`、`gradient_mpc0`、`DM0`）；η 归一化 η̃ = (η − (1, 0, 0.75)) / (1, 1.5, 0.75)。c′ 用训练池场景（每个场景计一次）的均值与标准差标准化（标准差 < 1e-6 的维度记 1）；h 与下面三个近邻输入用训练行的均值与标准差标准化。
- **近邻输入（主方法与"逐类加入""补充前训练池"）：** 对训练池每个场景 s、每个控制器 k，把该场景全部状态上同一 η 的结果合并，得到 (η̃_j, 平均成功 ȳ_j, 次数 n_j)。响应 f_{s,k}(η) = Σ_j w_j ȳ_j / Σ_j w_j，w_j = n_j·exp(−‖η̃ − η̃_j‖² / (2 × 0.2²))（场景里没有控制器 k 的数据时，用该场景全部控制器合并后的表）。场景距离 d(t, s) = ‖(c′_t − c′_s) / σ‖，σ 为训练池场景 c′ 的逐维标准差（< 1e-9 记 1）。取 d 最小的 3 个训练场景（并列按场景名排序），对训练池里的场景 t 只在其他训练场景中取（排除 t 自己）。g_k(η) = 3 个近邻的 f_{s,k}(η) 的平均；g_all(η) = 3 个近邻"各控制器 f 的平均"的平均。三个输入：logit(clip(g_k, 0.02, 0.98))、logit(clip(g_all, 0.02, 0.98))、到最近近邻的距离。
- **网络（主方法与"无近邻输入""逐类加入""补充前训练池"）：** 输入拼接 → Dense(128) → SiLU → Dropout(0.1) → Dense(128) → SiLU → Dropout(0.1) → Dense(1)，输出成功的 logit。
- **新 critic·原网络结构：** h → Dense(96)、c′ → Dense(32)、η̃ → Dense(32)、控制器 → Dense(32)，各自 SiLU 后拼接 → Dense(128) → SiLU → Dense(64) → SiLU → Dense(1)；无近邻输入、无 Dropout、4,000 步，其余同下。
- **损失与优化：** 每格以平均成功为软标签的 sigmoid 交叉熵；每批 1,024 格，按"每个场景总权重相同、同一场景内各格权重相同"的概率有放回抽取；AdamW，学习率 1e-3 按余弦衰减到 0（3,000 步），权重衰减 1e-4，全局梯度范数裁剪 5。种子 0–9（初始化 `jax.random.PRNGKey(种子)`，抽样 `numpy.random.default_rng(种子)`），10 个模型的 logit 取平均作为预测。
- **选择规则：** 对一个场景，在 49 个候选上计算"该场景全部状态 × 3 个控制器的平均预测成功率"，取最高者（并列取候选顺序靠前者）。另报"每个 (状态, 控制器) 各自取最高"的结果。
- **基线：** 一个全局 η（训练池里最好）：每个训练场景在每个菜单候选上的平均成功率（只用该候选覆盖了该场景 ≥ 50% 的 (状态, 控制器) 格时），场景之间平均后取最高。抄最近 k 个训练场景：同上的场景距离与成功率曲线，k 个最近场景平均后取最高。两个参照方法按名称表定义。
- **冻结：** 全部方法对 8 个验证场景的预测与选择写 sha256 进 `FROZEN.receipt.json` 之后，才读取验证标签计分；之后不改任何选择。
- **训练池内交叉验证（描述用，不用于改配方）：** 二维拥挤随机的 7 个版本归为一组，其余每个场景各为一组；组名排序后用 `numpy.random.default_rng(2026102674)` 打乱，第 i 折取第 i, i+6, i+12, … 组，共 6 折。对主方法、新 critic·无近邻输入、抄最近 1/2/3 个、一个全局 η，报告留出场景上的遗憾（只在该场景 ≥ 40 个候选有 ≥ 50% 覆盖时计算：最好候选的成功率减所选候选的成功率）。
- **学习曲线：** 从完整训练池随机抽 25%、50%、75% 的场景（各 5 次，`numpy.random.default_rng(2026102673)`），主方法每次 3 个种子；同时算抄最近 2 个训练场景与一个全局 η。另做"新 critic·逐类加入"：已有 14 个场景版本 + A、B、C、D、E 中的一类（每类 5 个种子）。
- **规模：** 约 120 次小网络训练，CPU 上每次约 15–20 秒，并行不超过 14 个进程。

## 六、统计与判定（预登记）

成功率一律为 3 个控制器等权、状态等权；unknown 计失败（另报计成功时的上界）。主口径在核对状态上计分，另报全部已标注状态。区间：5000 次自助法，两种（固定 8 个场景、只重抽每个场景的核对状态，`numpy.random.default_rng(2026102671)`；场景 → 状态，`numpy.random.default_rng(2026102672)`），各方法共用抽样索引。

1. **主比较表：** 名称表中全部方法在 8 个验证场景上的平均成功率，以及各自减一个全局 η、减抄最近 2 个训练场景的差与两种区间。
2. **逐场景表：** 每个验证场景每种方法选出的 η（写出三个数）与成功率；单列上一轮仍错的 4 个场景（八机静态障碍、对穿 10 机、八机无障碍、圆形五机）。
3. **补场景的作用：** 主方法减新 critic·补充前训练池；新 critic·逐类加入相对补充前训练池，在每个验证场景上的变化（哪一类新场景改善了哪个验证场景）；补充前后的覆盖上界。
4. **新场景本身：** 每个新场景的能力门结果、49 个候选成功率、最好的 3 个候选（写出三个数）；特别报告：带静态障碍的对穿是否偏好负侧移、大互斥（如 sE22），大机器人方形场地是否偏好小互斥、大朝目标（如 sE27、eta1 = (1.02, 0.22, 0.07)）。
5. **训练池内交叉验证表与学习曲线表。**

**判定（分开写，不合成一条；读未四舍五入的 JSON 数值）：**
- **有用：** 主方法减一个全局 η 的"只重抽状态"95% 区间下界 > 0。
- **超过抄近邻：** 主方法减抄最近 2 个训练场景，点估计 > 0 且"只重抽状态"区间下界 > 0 记"明显超过"；点估计 ≥ 0 记"持平或略好"；否则记"不如"。
- **补场景有用：** 主方法减新 critic·补充前训练池，点估计 > 0 且"只重抽状态"区间下界 > 0。
- **不伤：** 没有任何验证场景上主方法比一个全局 η 低 10 个百分点以上（点估计）。
- **critic 是否修好（本验证集上）：** "有用""不伤""补场景有用"三条都成立，且"超过抄近邻"至少为"持平或略好"。报告写明：8 个验证场景已在开发中用于比较配方，最终宣称需要另一批未碰过的场景做测试。

## 七、执行流程

1. 开工条件与完整性记录；复制执行与特征代码；逐位重放 2 局。
2. 历史检索，写 `HISTORY_CHECK.md`（`hetero_scene_pipeline_v1`、`cross_scene_correction_v1`、`eta_diverse_scene_test_v1`、`cross_scene_pipeline_v1/v2` 的报告与标签，第一节的开发结论）。
3. 构造 22 个新场景，五层报告，能力门（B 类不过时按 `_mild` 重建一次），新身份登记；写 `SCENES.json`、`GEOMETRY_REPORTS.json`、`CAPABILITY_TABLE.json`。
4. 新 rollout（官方预检后只跑真正缺失的请求；worker 写 journal，结束后单一 merger 合库）；写新场景的 `LABEL_TABLES.json`。
5. 抽取已有数据、合并新场景，写 `TRAINING_POOL.json` 与训练格文件；验证集输入准备。
6. 训练全部 critic 变体与基线，交叉验证与学习曲线；冻结验证集上的全部预测与选择。
7. 读取验证标签计分，统计与判定；数据库核对；结束完整性记录；写 `COMPLETE.json`。

## 八、失败处理、无人值守与资源

- **解释器：** 主解释器 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`（有 jax、flax、optax、distrax、numpy、scipy、PIL；没有 matplotlib、pandas、scikit-learn）；画图用 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python`（matplotlib、numpy）；`/usr/bin/python3` 没有 numpy。本轮不要求图；若画学习曲线图，放在最后一步、从已封存的 JSON 读取。
- **三类失败：** (1) 整轮停止只限：违反源码或仓库保护、违反共享数据库规则、逐位重放失败。(2) 主线作业失败（能力门、rollout、合库、抽取、训练、冻结、统计）：先读日志；基础设施原因按原清单、原种子补交，自动处理，每阶段最多 3 次；实现错误修正后只重跑受影响部分，写 `ITERATION_LOG.md`（修正不得改变已冻结的选择；会改变就停下报告）；仍不能完成的场景记为缺口，写 `GAPS.md`，其余照常。能力门不过的场景按第三节处理并单列。(3) 非主线（图、排版、可选表）放最后，失败写进 `GAPS.md`，不影响交付。
- **无人值守：** 每 30 分钟更新 `STATUS.md`；执行代理至少每 30 分钟检查一次，直到交付，不等用户发现。
- **资源：** 用户全部任务合计不超过 18 个 CPU 份额、2 个 GPU 份额（临时上调由用户决定）；全程只用 CPU。新 rollout 约 2.6 万次，按以往实测（16 进程约 12 次/秒，D 类单局更长）约 45 分钟；训练约 10 分钟。新 rollout（不含能力门）超过 32,000 次就停下报告。Slurm worker 时限约 1 小时；提交后立即确认作业处于 RUNNING；本人作业排队超过 15 分钟而 CPU 空闲就撤销并以更短时限重交；只有其他用户的排队作业确实缺 CPU 或内存时才让出。

## 九、交付

- `REPORT.md`（中文；第一段：critic 是否修好，以及"有用 / 超过抄近邻 / 补场景有用 / 不伤"四条各自的结论与数值，主方法减一个全局 η、减抄最近 2 个训练场景的差与两种区间）；主比较表；逐场景表（选出的 η 写出三个数）；补场景作用表（含逐类加入）；新场景表；交叉验证表；学习曲线表。
- `HISTORY_CHECK.md`、`SCENES.json`、`GEOMETRY_REPORTS.json`、`CAPABILITY_TABLE.json`、新场景 `LABEL_TABLES.json`、`TRAINING_POOL.json`、训练记录与模型、`FROZEN.receipt.json`、`STATISTICS.json`（全部未四舍五入的数值）、`STATUS.md`、`GAPS.md`、`ITERATION_LOG.md`、数据库核对、开工与结束完整性记录、`COMPLETE.json`。
- 无视频要求。

## 附：近邻输入与训练循环的参考写法（与第五节等价，以第五节文字为准）

```python
# 归一化 eta：E = (eta - [1, 0, 0.75]) / [1, 1.5, 0.75]
def response(table, E, k, bw=0.2):
    # table[k] = (X: 该场景控制器 k 的不同 eta（已归一化）, Y: 平均成功, N: 次数)；没有 k 时用全部控制器合并
    X, Y, N = table[k] if k in table else merge_all(table)
    w = N[None] * np.exp(-((E[:, None, :] - X[None]) ** 2).sum(-1) / (2 * bw * bw))
    return (w @ Y) / np.maximum(w.sum(1), 1e-12)

def neighbour_inputs(c_target, E, ctl, pool, tables, sigma, k_nn=3):
    X = np.array([c_prime[s] for s in pool]); d = np.sqrt((((X - c_target) / sigma) ** 2).sum(1))
    near = [pool[i] for i in np.argsort(d, kind='stable')[:k_nn]]      # pool 已按场景名排序；训练场景自己不在 pool 里
    g = np.mean([[response(tables[s], E[i:i+1], ctl[i])[0] for i in range(len(E))] for s in near], 0)
    g_all = np.mean([np.mean([response(tables[s], E, k) for k in tables[s]], 0) for s in near], 0)
    lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
    return np.stack([lg(g), lg(g_all), np.full(len(E), d.min())], 1)

# 训练：每步按 p ∝ 1 / 该格所在场景的格数 有放回抽 1024 格；
# loss = mean(optax.sigmoid_binary_cross_entropy(logit, y_soft))；
# optax.chain(optax.clip_by_global_norm(5.), optax.adamw(optax.cosine_decay_schedule(1e-3, 3000), weight_decay=1e-4))
```

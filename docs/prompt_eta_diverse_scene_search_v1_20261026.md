# 补场景 v1：寻找"几个 η 解决不了"的场景（粗筛 → 扩展）：实现说明

**写作约定：** 方法、场景、控制器和候选第一次出现时写全称，括号里给代号，代号只用于定位文件；报告照此书写。候选 η 是三个数（朝目标, 侧移, 互斥），提到某个候选时同时写出这三个数。本轮只找场景、出标签，不训练任何选择模型。

**名称表（报告一律用这些名称）：**

| 名称 | 含义 |
|---|---|
| 每个场景自己的最好 η | 在该场景一半状态上选成功率最高的 η，在另一半状态上计分；随机对半划分 200 次取平均 |
| 最好的 k 个固定 η | 在全部场景上挑 k 个 η，每个场景用这 k 个里对它最好的一个；挑选只用各场景的同一半状态，计分用另一半 |
| 覆盖缺口(k) | 每个场景自己的最好 η 减最好的 k 个固定 η（场景等权平均） |
| 交叉使用损失 | 把场景 i 自己的最好 η 用到场景 j 上，比 j 自己的最好 η 低多少（选择与计分同样对半划分） |
| 邻近插值 | 某个旋钮取值不看自己的标签，只用旋钮上相邻两个取值的标签选 η |
| 快控制器 | 集中式随机采样模型预测控制（`centralized_sampling0`）、梯度下降模型预测控制（`gradient_mpc0`）、分布式模型预测控制（`DM0`）、交叉熵方法模型预测控制（`cross_entropy0`） |

## 零、开工条件、源码与仓库保护（任何步骤之前）

- **开工条件：** `diagnostics/cross_scene_pipeline_v2/COMPLETE.json` 存在且 `status` 为 `COMPLETE`，其 `INTEGRITY_COMPARE.json` 的 `passed` 为真；否则停下报告，不建目录。若 `diagnostics/cross_scene_pipeline_v2_learning_tests/` 正在运行，登记为并行任务目录。
- 本轮新代码与输出只放 `diagnostics/eta_diverse_scene_search_v1/`。已有源码、`docs/`、`AGENTS.md`、其他 `diagnostics/` 目录（含 `cross_scene_correction_v1/`、`cross_scene_pipeline_v1/`、`cross_scene_pipeline_v2/`、`cross_scene_homogeneity_audit_v1/`、`baseline_literature_scenes_supplement_eta_swap8_v1/`、`baseline_literature_scenes_distributed_v2/`、`controller_state_geometry_transfer_v1/`、`transferable_state_conditioning_v1/`）、`datasets/`、`bottleneck_family/`、MAC-Flow 官方仓库（`../01_MACFlow_Baseline_Reproduction/MACFlow_Official`）、虚拟环境一律只读；不安装、不升级任何包。`cross_scene_prospective_v1/` 只看 `os.stat` 元数据，不打开。
- 复用只用只读 import，或复制到本轮目录并记录 sha256；不在运行时给旧模块打补丁（JAX 精度恢复除外）。旧目录里的运行入口有写入副作用（`experiment.py` 在自身目录建缓存目录；`common.py` 的 `evidence` 追加自身目录的 `EVIDENCE_INPUTS.jsonl`；本地编译的冲突搜索内核写死查询缓存根目录），一律不在旧目录运行、不 import 旧目录模块。以 `cross_scene_pipeline_v1/build_isolated.py` 为模板，从 `cross_scene_correction_v1` 在本轮目录重建隔离的执行图（实验名 `exp_eta_diverse_scene_search_v1`，只改缓存根目录）；`scenes.py`（`make_new`、`geometry_report`、`features`）与 `stages.py`（`capability_cases`、`capability_select`）复制后按本说明改写；Slurm 作业名前缀用 `xds_`，撤销作业只认这个前缀。
- git 只用只读命令（两个仓库都有用户未提交的工作）。开工与结束各记录两个仓库的 `git status --porcelain`、`git diff` 的 sha256 与所有复用文件的 sha256，两次必须一致；本轮目录、开工时登记的并行任务目录、`docs/` 下新增的 `prompt_*.md` 单列，不算失败。
- 共享数据库：允许 (1) 官方 `cache_writer.append_journal` 写默认的 `shared_rollout_db/journals/<本轮实验 UID>/`；(2) 官方单进程 merger；(3) 单进程、官方 `shared_rollout_db.src.rollout_db` 的 connect/transaction，只对新的 experiment/scenario/state/controller_config/eta 行做 INSERT，遇到 UID 内容冲突即停，登记前后已有行逐表哈希不变。禁止改 `cache_writer.ROOT`、禁止复制数据库；开工时断言 `shared_rollout_db.__file__` 与 `cache_writer.ROOT` 都在 `/home/zhihan/research/Basin_C1/` 之下。预检与缓存规则见 `shared_rollout_db/README.md`。
- 遇到与本说明矛盾、或要求的接口不存在：停下报告，不改源码。下文点名的接口与文件均已核实存在。

## 一、要回答的问题与已有证据

**问题：** 能否找到一组场景，使"几个 η 解决不了所有场景"：即使用最好的 3 个固定 η，覆盖缺口(3) 仍不少于 10 个百分点。

**已有数据（2026-10-25/26 对共享数据库全量只读统计，零 rollout）都达不到：**

| 场景集 | 每个场景自己的最好 η | 最好的 1 / 2 / 3 个固定 η |
|---|---:|---|
| C1 原四场景（Toy 让行、双瓶颈、四路口、环形交换；16 个 η 的旧库，同一协议） | 93.2 | 61.5 / 87.0 / 93.3 |
| 18 个跨场景场景（49 个候选，全部合格控制器） | 71.2 | 71.1 / 74.9 / 77.0 |
| 其中有至少 2 个快控制器的 11 个场景（只算快控制器） | 67.0 | 60.9 / 67.8 / 70.6 |

`controller_state_geometry_transfer_v1` 在 Toy 宽度族里也发现：少数几个 η 几乎处处成功，加深让行湾、扩大初始状态都没有打破，只有收紧时限开始减少饱和。

**可能让最好 η 漂移的旋钮（已有标签中的线索）：** 圆形五机（机器人大、机器人面积占自由面积 16.5%）的最好 η 是 η9（0.94, 0.43, 0.43），成功率与"互斥"的 Spearman 相关 −0.63，而小机器人的对穿场景偏好大互斥；八机对穿加入静态障碍后，最好 η 从扩展范围候选 sE29（0.34, 0.55, 1.20）变为 sE22（0.55, −0.20, 1.42），侧移方向反转；对穿人数从 4 增到 12，最好 η 从 sE29 移到 sE13（0.45, 1.34, 1.03）。

**只用快控制器的原因（用户要求：最耗时的几个不跑）：** `cross_scene_pipeline_v1` 的实测中，冲突搜索与优先级 A* 占 12.7 CPU 小时里的 11.4 小时（最慢的一成要 88–191 秒，最长 16 分钟与 46 分钟），其余控制器中位 1.6–3.2 秒、最长 23 秒。因此本轮不跑冲突搜索与优先级 A*；只有这两类能过能力门的瓶颈类场景不在本轮范围内，报告写明。

## 二、统一执行语义（沿用 `cross_scene_correction_v1`，不改）

- 单积分器、分布式安全投影、接触门控修正、成功判据（全部机器人在时限内到达且无碰撞；timeout 不等于死锁；碰撞按机器人/静态/移动分类；数值失败记 unknown，保留不重跑）、随机数规则；只跑种子 0；只用固定修正。
- 候选 49 个，精确 float64 值取自 `baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json`（与已有记录同一身份）。
- 控制器：只用 4 个快控制器实例（与 `cross_scene_correction_v1/SELECTED_CONTROLLERS.json` 中对穿场景的选择相同）。新场景必须 4 个都通过能力门才收录（能力门同 `cross_scene_correction_v1`：标签状态的单机器人版本，每台机器人单独，种子 0 与 1，成功率 ≥ 95%）；不通过就整场景排除并单列，不换控制器、不替换场景。
- 新请求用复制的 `scenes.make_new` 构造（版本 `cross_scene_si_v1`）。新场景名必须以 `xsc_` 开头（`scene_environment.NewNominal` 按这个前缀把模型预测控制类控制器接到 `controllers_new.Nominal`），统一写作 `xsc_div_<族>_<取值>`。
- 每个新场景用复制的 `scenes.geometry_report` 出五层报告（几何有效、单机器人有路、联合可解写"未认证"、控制器成功、死锁解决）；不合法的状态照实记录，不替换。

## 三、场景族（全部预先写定）

**基础场景：** 与 `cross_scene_correction_v1/SCENES.json` 中对穿 6 机（`xsc_swap_n6`）完全相同：同样的 12 个状态（位置、目标、速度），`LiteratureConfig`（机器人半径 0.05、dt 0.05、限速 0.5、`max_steps = ceil(4 × 最长直线行程时间 / 0.05)`）。各族只改下面写明的旋钮。

| 族 | 旋钮与 5 个取值 | 第一阶段用的 3 个取值 |
|---|---|---|
| A 机器人尺寸族 | 机器人半径 0.05（即已有对穿 6 机，复用标签）、0.08、0.11、0.14、0.17 | 0.05、0.11、0.17 |
| B 时间余量族 | `max_steps = ceil(f × 最长直线行程时间 / 0.05)`，f = 1.6、2.0、2.5、3.2、4.0（即已有对穿 6 机） | 1.6、2.5、4.0 |
| C 中心障碍族 | 以原点为中心的正方形障碍，半边长 s = 0、0.08、0.16、0.24、0.32 | 0、0.16、0.32 |
| D 偏心障碍族 | 半边长 0.16 的正方形障碍，中心在 (dx, 0)，dx = 0（即 C 族 s = 0.16，同一场景）、0.1、0.2、0.3、0.4 | 0、0.2、0.4 |

- **C、D 族的墙体：** 由 `bottleneck_family.scenario.Geometry(Config(...))` 生成，状态里带 `bottleneck_config`（同 v1 瓶颈场景的做法）：`num_agents=6, half_length=2.0, half_height=2.0, barrier_x=(1.7,), openings=(((0.0, 3.6),),), barrier_thickness=0.1, extra_rectangles=` 该障碍（s = 0 时为空）`, agent_radius=0.05, dt=0.05, max_speed=0.5, max_steps` 同基础场景，其余字段取默认值。屏障放在 x = 1.7、开口覆盖几乎全高，只是为满足 `Config` 的校验（屏障右侧需留出空间），离机器人所在的圆（半径约 0.85 以内）很远。已核实：这种构造对 s = 0、0.16、0.32 与 dx = 0.2、0.4 都能通过 `Config` 校验，按基础场景方式生成的 8 个状态全部有效，每台机器人都有单机器人路径。C 族 s = 0 是带场地边界的新场景，不等同于已有对穿 6 机。若实际构造仍被拒绝，停下报告，不自行换做法。
- **已有族（零 rollout）：** E 人数族：对穿 4 机、6 机、10 机、12 机（`xsc_swap_n4/6/10/12`，圆半径 N/8）与对穿 8 机加密（`xsc_swap_n8_dense`）。F 文献锚点：窄通道、拥挤随机、圆形五机、八机无障碍、八机静态障碍、八机移动障碍（`lit_narrow_passage`、`lit_crowded_random`、`lit_circle_five`、`lit_swap8_free`、`lit_swap8_obstacles`、`lit_swap8_moving`）。都取 `cross_scene_correction_v1/LABEL_TABLES.json` 的已有标签，只用 4 个快控制器那几行。

## 四、两阶段与预算

**第一阶段（粗筛）：** 新场景共 9 个（A 族 0.11、0.17；B 族 1.6、2.5；C 族 0、0.16、0.32；D 族 0.2、0.4），状态 0–7 × 4 个快控制器 × 49 个候选 × 种子 0，至多 14,112 次新 rollout，另加能力门。先做计时试跑（每个新场景 1 个状态 × 4 个控制器 × 4 个候选），写 `PILOT_COST.md` 后照常继续。

**第一阶段判定（每个族，只用状态 0–7；E 族同样计算，零成本）：**
- 有信息：该族 3 个场景中至少 2 个的样本内最好成功率在 [0.20, 0.95] 之间；
- 有漂移：该族场景两两之间，至少有一对的交叉使用损失 ≥ 15 个百分点（取两个方向中较大者）。
- 两条都满足的族进入第二阶段。没有任何族满足：写 `STAGE1_DECISION.json` 与报告后结束，不跑第二阶段。

**第二阶段（扩展，只对进入的族）：** 补齐 5 个取值，每个取值用状态 0–11（第一阶段场景补状态 8–11）。新 rollout 上限 20,000 次；超出时按第一阶段交叉使用损失从大到小依次纳入族，直到上限为止。

## 五、分析（零额外 rollout）

**候选场景集：** 进入第二阶段的族的全部场景，加 E 族与 F 锚点，再只保留样本内最好成功率在 [0.20, 0.95] 之间的场景（有信息的场景）。成功率一律为 4 个快控制器等权、状态等权，unknown 计失败。

1. **覆盖缺口（主）：** 覆盖缺口(k)，k = 1–5，用名称表的对半划分算法（200 次），并给出按场景 → 状态分层自助法的 95% 区间（5000 次，`numpy.random.default_rng(2026102601)`）；另报固定 η 在全部状态上挑选（对固定 η 更有利）的保守版本。分别对候选场景集、只算新场景、每个进入第二阶段的族单独报告。
2. **漂移描述：** 每个场景自己的最好 η（写出三个数）与前 3 名；每个族的"旋钮取值 → 最好 η 三个分量"表。
3. **邻近插值（可学性的线索，不进判定）：** 每个进入第二阶段的族、每个内部取值，只用相邻两个取值的标签平均选 η，在本取值的状态上计分；与该取值自己的最好 η、与最好的 3 个固定 η 比较。
4. **按控制器检查：** 每个（场景, 控制器）的最好 η，看 4 个快控制器的漂移方向是否一致，以区分"场景造成的漂移"与"控制器混合造成的漂移"。

## 六、判定（预登记）

- **找到"几个 η 解决不了"的场景集**：候选场景集的覆盖缺口(3) ≥ 10 个百分点，且候选场景集至少 8 个场景、其中至少 4 个是本轮新场景。满足就写 `VALUABLE_SCENE_SET.json`（场景清单、标签位置、五层报告、能力门结果），供下一轮使用。
- 不满足：照实报告达到的最大覆盖缺口(3) 以及每个族的漂移大小，不再追加 rollout。
- 一切判定读未四舍五入的 JSON 数值。

## 七、执行流程

1. 开工条件与完整性记录（第零节）；隔离重建；按 `cross_scene_pipeline_v1/replay.py` 的做法逐位重放 `cross_scene_correction_v1` 已存的对穿 12 机与拥挤随机各 2 局，结果与轨迹逐位一致才继续。
2. 历史检索，写 `HISTORY_CHECK.md`（第一节各项的路径与可复用范围）。
3. 构造 A–D 族场景与状态，五层报告，能力门，新身份登记，写 `SCENES.json`、`GEOMETRY_REPORTS.json`、`CAPABILITY_TABLE.json`。
4. 第一阶段 rollout（官方预检后只跑真正缺失的请求；worker 写 journal，阶段结束单一 merger 合库）；第一阶段判定，写 `STAGE1_DECISION.json`。
5. 第二阶段（若有）。
6. 分析（第五节）、数据库核对（所有请求在库、身份与结果一致、无未合并 journal）、结束完整性记录，写 `COMPLETE.json`。

## 八、失败处理与无人值守

- **解释器：** 主解释器 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`（有 jax、flax、optax、distrax、numpy、scipy、PIL；没有 matplotlib、pandas、scikit-learn）；画图用 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python`（matplotlib 3.11、numpy）；`/usr/bin/python3` 没有 numpy。本轮不要求图。
- **三类失败：** (1) 整轮停止只限：违反源码或仓库保护、违反共享数据库规则、逐位重放失败、C/D 族构造被拒。(2) 主线作业失败（能力门、rollout、合库、分析）：先读日志；基础设施原因按原清单、原种子补交未完成部分，协调程序自动处理，每阶段最多 3 次；实现错误修正后只重跑受影响部分，写 `ITERATION_LOG.md`；仍不能完成的场景记为缺口，写 `GAPS.md`，其余照常。(3) 非主线（排版、可选附表、任何图）放在最后单独一步，只读已封存的 JSON，失败写进 `GAPS.md`。
- **无人值守：** 协调程序自己执行上面的处理；每 30 分钟更新 `STATUS.md`（阶段、完成数、失败与补交、预计剩余时间）；执行代理至少每 30 分钟检查一次，直到交付。

## 九、资源与调度

- 用户全部任务合计不超过 18 个 CPU 份额、2 个 GPU 份额（临时上调由用户决定）。全程只用 CPU。
- 不跑冲突搜索与优先级 A*。按快控制器的实测成本（中位约 2 秒），第一阶段约 8–10 CPU 小时，第二阶段至多约 12–14 CPU 小时；试跑后在 `PILOT_COST.md` 中更新。
- Slurm worker 时限约 3 小时，在时限前约 55 分钟停止领取新任务组；每次提交后立即确认作业处于 RUNNING，几分钟后再查一次。本人作业排队超过 15 分钟而 CPU 空闲，就撤销并以更短时限重交，写日志；不设"若干分钟没有新结果就撤销重交"的规则。只有当其他用户的排队作业确实缺 CPU 或内存时才让出。任务组按预计耗时从长到短排队，单组约 10 分钟。

## 十、交付

- `REPORT.md`（中文；第一段：判定结论、候选场景集与新场景的覆盖缺口(1)、(2)、(3) 及区间、哪些族有漂移）；每个场景的方法表（每个场景自己的最好 η 写出三个数、最好的 1/2/3 个固定 η 是哪几个）；每个族的漂移表；邻近插值表；按控制器检查表。
- `HISTORY_CHECK.md`、`SCENES.json`、`GEOMETRY_REPORTS.json`、`CAPABILITY_TABLE.json`、`PILOT_COST.md`、`STAGE1_DECISION.json`、`COVERAGE.json`（全部未四舍五入的数值）、`VALUABLE_SCENE_SET.json`（若判定通过）、`STATUS.md`、`GAPS.md`、`ITERATION_LOG.md`、数据库核对、开工与结束完整性记录、`COMPLETE.json`。
- 无视频要求。

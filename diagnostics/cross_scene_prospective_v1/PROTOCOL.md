# 跨场景修正前瞻检验 v1：实现说明

**写作约定：** 方法、场景和候选第一次出现时写全称，括号里给出文件中的代号，代号只用于定位文件与身份；报告也照此书写。

## 零、开工条件、源码与仓库保护（任何步骤之前）

- **开工条件：** 只有当 `diagnostics/cross_scene_correction_v1/COMPLETE.json` 存在，且其 `INTEGRITY_COMPARE.json` 的 `passed` 为真时才开工。那一轮的结束审计会比对整个 Basin_C1 的 `git status`，在它完成前新建本轮目录会使其审计失败。条件不满足就停下报告，不建目录。
- 本轮所有新代码与输出只放 `diagnostics/cross_scene_prospective_v1/`。已有源码、`docs/`、`AGENTS.md`、其他 `diagnostics/` 目录（含 `cross_scene_correction_v1/`、`cross_scene_homogeneity_audit_v1/`、`baseline_literature_scenes_distributed_v2/`、`baseline_literature_scenes_supplement_eta_swap8_v1/`）、`datasets/`、`bottleneck_family/`、MAC-Flow 官方仓库（`../01_MACFlow_Baseline_Reproduction/MACFlow_Official`）、虚拟环境一律只读；不安装、不升级任何包。
- 复用只用只读 import，或复制到本轮目录并记录 sha256；需要不同行为时写新文件、子类或新函数，不在运行时给旧模块打补丁。旧运行入口有写入副作用的（`cross_scene_correction_v1` 的 `experiment.py` 会在自身目录建 TMPDIR/XDG/CUDA 缓存；本地编译的冲突搜索内核写死了查询缓存根目录），按该轮 `build_backend.py`、`build_extensions.py` 的做法在本轮目录重建隔离副本（只改缓存根目录），并通过第五节第 2 步的逐位重放后才能使用。
- git 只用只读命令：不 commit、stash、checkout、reset、restore、clean、add（Basin_C1 与 02_C1_Toy_GiveWay 两个仓库都有用户未提交的工作）。开工和结束各记录两个仓库的 `git status --porcelain`、`git diff` 的 sha256，以及所有复用文件的 sha256，两次必须一致；本轮目录自身那一行、开工时登记的并行任务目录（例如 `cross_scene_pipeline_v1/`）、`docs/` 下新增的说明文件（`prompt_*.md`）单列，不算失败。
- 共享数据库：允许 (1) 官方 `cache_writer.append_journal` 写默认的 `shared_rollout_db/journals/<本轮实验 UID>/`；(2) 官方单进程 merger；(3) 新身份登记：单进程、官方 `shared_rollout_db.src.rollout_db` 的 connect/transaction，只对新的 experiment/scenario/state/controller_config/eta 行做 INSERT，遇到 UID 内容冲突即停，登记前后已有行的逐表哈希不变。禁止改 `cache_writer.ROOT`、禁止复制数据库。预检与缓存规则见 `shared_rollout_db/README.md`。
- 开工时列出最近 24 小时有写入的其他 diagnostics 目录与他人正在运行的作业目录；它们的变化单列，不算本轮失败（前提是本轮既不 import 也不写它们）。
- 遇到与本说明矛盾、或要求的接口不存在：停下报告，不改源码。下文点名的接口与文件均已核实存在。

## 一、要检验的问题与历史依据（先查后做）

**问题：** 训练池先按预先写定的规则去掉"需要同一组修正"的重复场景，再用场景特征做零样本选择（近邻），在**从未参与任何标签或分析的新场景**上，是否优于"所有场景用同一组最好的修正"？

**来源（只读，写进 `HISTORY_CHECK.md`：路径、执行语义是否一致、可复用范围、仍缺证据）：**
1. `diagnostics/cross_scene_correction_v1/`：18 个有标签场景 × 49 个候选的全表标签（`LABEL_TABLES.json`）、场景定义与特征（`SCENES.json`、`FEATURES.json`）、模型代码（`models.py`）、留一场景与留一族结果（`LOSO_CHOICES.json`、`LOFO_CHOICES.json`、`STATISTICS.json`）、`REPORT.md`。预登记结果：主模型（岭回归）减全局最佳单一修正 +0.5 个百分点 [−4.4, +5.9]，未通过；留一族测瓶颈族 −32 [−58, −12]，未通过。外推（Gap1 N=10）完成 107/120 次，其余 13 次按用户要求停止，只作不完整描述（`USER_STOP_AMENDMENT.md`）。
2. `diagnostics/cross_scene_homogeneity_audit_v1/`（看过结果之后的探索性分析，只读）：全局最佳单一修正在 9/18 个场景就是该场景最佳、12/18 个场景差距 ≤5 个百分点。按第三节的去重规则剩 11 个场景，留一场景时次模型（近邻）比同池全局最佳 +8.0 [+1.8, +15.5]；但同池全局最佳在各折之间来回变化，改与全 18 场景的全局最佳比时只剩约 +4.9，且主要来自窄通道、拥挤随机、圆形五机、缺口 6 机。本轮就是对这一事后发现的前瞻检验，不得把它当作已有结论。
3. `diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_REVISED_20261021_lit_crowded_random.json`（拥挤随机 14 机的状态来源）。
4. `bottleneck_family/scenario.py`（`Config`、`Geometry`、`build_instance`）与 `bottleneck_family/__main__.py`（`preset`）。

能精确复用的结果一律复用，只补真正缺失的请求。

## 二、统一执行语义（全部沿用 `cross_scene_correction_v1`，不改）

- 单积分器、分布式安全投影、接触门控修正、49 个候选的精确身份（`best_correction_search_v1/ACTIONS.json`）、限速与 dt、成功判据（全部机器人在时限内到达且无碰撞；timeout 不等于死锁；碰撞按机器人/静态/移动分类；数值失败记 unknown，保留不重跑）、随机数规则，均与 `cross_scene_correction_v1` 完全相同；新场景的请求按该轮 `scenes.make_new` 的方式构造，瓶颈布局在状态里带 `bottleneck_config`，由该轮 `scene_environment.NewEnv` 接入墙体。
- 控制器：8 类，参数不改；每类取文献线 v2 训练池中第一个通过本场景能力门的实例（与上一轮相同的规则）。
- 只跑种子 0；只用固定修正，不含两段切换。

## 三、训练池与冻结（零 rollout）

- **训练标签：** 只读 `cross_scene_correction_v1/LABEL_TABLES.json`，每个场景压成"49 个候选的类型等权成功率"向量（unknown 计失败）。场景特征取该轮 `SCENES.json` 的 `features`。本轮不为训练场景新增任何标签。
- **去重规则（预登记，零 rollout）：**
  - 两个场景算重复的条件：二者 49 维成功率向量的 Spearman 相关（并列取平均秩）≥ 0.9，且存在一个候选，使它在两个场景上距各自最佳都不超过 2 个百分点（即 min_a max(遗憾_i(a), 遗憾_j(a)) ≤ 0.02）。
  - 贪心去重：每次取相关最高的一对重复场景，去掉二者中与其余剩余场景平均相关更高的那个（相等时去掉在 `SCENES.json` 中排在前面的那个），直到没有重复对。
  - 必须得到以下 11 个场景（去重训练池）：窄通道（`lit_narrow_passage`）、拥挤随机（`lit_crowded_random`）、圆形五机（`lit_circle_five`）、八机静态障碍（`lit_swap8_obstacles`）、对穿 4 机（`xsc_swap_n4`）、对穿 12 机（`xsc_swap_n12`）、对穿 8 机加密（`xsc_swap_n8_dense`）、缺口 4 机（`xsc_gap_n4`）、缺口 6 机（`xsc_gap_n6`）、双瓶颈（`xsc_double_n4`）、错位口（`xsc_offset_n4`）。得到的列表不同就停下报告。
- **方法（复制 `cross_scene_correction_v1/models.py` 的 `select`，逐字不改）：**
  - 主方法：次模型同构的近邻选择器（`knn`），在去重训练池上拟合，k ∈ {1, 2, 3, 5} 用训练池内部留一场景选（应得 k = 2）。
  - 次方法：岭回归选择器（`ridge`），同池拟合，λ ∈ {0.1, 1, 10, 100}（应得 λ = 100）；以及二者在全 18 场景池上的版本（应得 k = 3、λ = 100）。
  - 主对照：全局最佳单一修正（`global_fixed`）＝全 18 场景上场景等权平均成功率最高的候选，应为扩展范围候选 sE29（朝目标 0.34、侧移 0.55、互斥 1.20）。去重池上的全局最佳也应为 sE29。
  - 参照：η13（`eta13`）、仅安全过滤（`safety`）。
- 上述任一期望值（场景列表、k、λ、全局最佳）对不上，就停下报告。

## 四、新场景（测试集）

规则：每个测试场景在布局或密度上与 18 个训练场景都不同；不加"只换人数而密度不变"的版本（例如按半径 = N/8 的对穿新人数）。共 12 个，场景之间等权。下表参数之外一律取 `bottleneck_family.scenario.Config` 默认值（机器人半径 0.16、限速 0.5、dt 0.05、半长 8、半高 4、墙厚 0.6），`max_steps = max(2000, 800N)`，`split='train'`；每个瓶颈布局 16 个状态，第 k 个状态用 `build_instance(replace(cfg, seed=2026102600+k))`，k = 0…15，不合法的状态不替换、照实记录。开口写作（中心 y，宽度）。

| 代号 | 全称 | N | 与默认配置的不同 |
|---|---|---|---|
| `xsp_long_tunnel_n4` | 长隧道 | 4 | `barrier_thickness=3.0`，开口 (0, 0.5) |
| `xsp_clutter_n4` | 杂物 | 4 | `preset('clutter', 4)` |
| `xsp_wall_side_door_n4` | 贴墙门 | 4 | 开口 (3.0, 0.5) |
| `xsp_zigzag_three_n4` | 三墙之字 | 4 | `barrier_x=(-3, 0, 3)`，开口依次 (1, 0.5)、(−1, 0.5)、(1, 0.5) |
| `xsp_chicane_n4` | 小弯道 | 4 | `barrier_x=(-0.6, 0.6)`，开口依次 (−0.5, 0.5)、(0.5, 0.5) |
| `xsp_narrow_door_n4` | 窄门 | 4 | 开口 (0, 0.4) |
| `xsp_low_room_n6` | 矮房间 | 6 | `half_height=2.0`，开口 (0, 0.5) |
| `xsp_chamber_pillars_n6` | 双墙夹柱 | 6 | `barrier_x=(-1.5, 1.5)`，两墙开口均 (0, 0.5)，`extra_rectangles=((-0.4,0.6,0.4,1.4),(-0.4,-1.4,0.4,-0.6))` |
| `xsp_wide_wall_door_n6` | 贴墙宽门 | 6 | 开口 (2.8, 1.2) |
| `xsp_unequal_doors_n6` | 不等宽三门 | 6 | 一道墙三个开口 (−2, 0.4)、(0, 0.6)、(2, 0.9) |
| `xsp_offset_tunnel_n6` | 偏心隧道 | 6 | `barrier_thickness=2.0`，开口 (−2.0, 0.5) |
| `lit_crowded_random_n14` | 拥挤随机 14 机 | 14 | 状态取第一节第 3 项文件 `training` 中 N = 14 的前 16 个（按文件顺序）；配置与上一轮拥挤随机 8 机相同（半径 0.15、dt 0.15、限速 1.0、`max_steps=333`），只换人数；请求按上一轮 `main_CASES.json` 中拥挤随机 8 机请求的构造方式生成 |

- 场景特征用上一轮 `scenes.features` 的同一代码（复制，逐字不改）计算，只用场景定义与初始构型，不用任何结果。按 `cross_scene_correction_v1/scenes.py` 中瓶颈场景的写法填 `free_area`、`min_opening_over_diameter`、`serial_walls`；拥挤随机 14 机沿用 8 机场景的墙体与开口描述。
- **冻结（任何测试场景的 rollout 之前，包括能力门）：** 对每个测试场景算出第三节全部方法的选择，连同特征、去重结果、模型参数、全部统计代码一起写 sha256 进 `MODEL_FROZEN.receipt.json`。按本说明构造时，应得到：11 个瓶颈布局上近邻选扩展范围候选 sE05（朝目标 0.11、侧移 0.32、互斥 0.53；全 18 场景池版本在偏心隧道上为 sE22），岭回归选 sE13；拥挤随机 14 机上近邻选 sE01、岭回归选 sE19。对不上就停下报告（说明场景构造与本说明不一致），不得调整场景去凑。冻结后不再改任何选择。

## 五、执行流程

1. 开工条件与完整性记录（第零节）；历史检索与登记（第一节）。
2. 逐位重放：用本轮代码重放上一轮已存的双瓶颈（`xsc_double_n4`）与拥挤随机 8 机各至少 2 局，结果与轨迹逐位一致才继续。
3. 新场景合法性：每个瓶颈布局按 AGENTS.md 出五层几何报告（几何有效、单机器人有路、联合可解（不认证则写未认证）、控制器成功、死锁解决）；容量不足或通道封死照实报告，不缩小机器人，不把几何不可达记作死锁。不合法的布局整体排除并单列，不替换。
4. 特征、选择与冻结（第四节）。
5. 能力门：每个测试场景、每个控制器实例，在测试状态的单机器人版本上（每台机器人单独、种子 0 与 1）成功率 ≥ 95% 才进入该场景；未过门的类型在该场景排除并单列。
6. 计时试跑：每个测试场景 × 每类控制器，2 个状态 × 2 个候选；单局耗时与预计总机时写进 `PILOT_COST.md`，然后照常继续。
7. 评估：每个测试场景的 16 个状态 × 通过能力门的类型 × 该场景冻结选择的并集（近邻两版、岭回归两版、全局最佳、η13、仅安全过滤，去重后约 5–6 个候选）× 种子 0。官方预检后只跑真正缺失的请求；worker 写 journal，阶段结束单一 merger 合库。
8. 统计（第六节），然后数据库核对（所有请求在库、身份与结果一致、轨迹哈希一致、无未合并 journal）与结束完整性记录。

## 六、判定（预登记）

- 指标：每个测试场景上所选候选的成功率，场景内控制器类型等权，场景之间等权；unknown 计失败（另报 unknown 计成功的上界）。区间用 5000 次成对自助法，按场景 → 控制器类型 → 状态分层重抽。
- **主判定：** 近邻（去重训练池）减全局最佳单一修正 sE29，12 个测试场景等权平均差的 95% 区间下界 > 0。
- 次要结果（照报，不改变主判定）：
  1. 岭回归（去重池）减全局最佳；近邻与岭回归的全 18 场景池版本减全局最佳（看去重是否起作用）。
  2. 分层：11 个瓶颈布局、拥挤随机 14 机分别报告。
  3. 稳健版本（预登记）：逐场景差值的中位数、两端各去 1 个场景后的截尾均值、Wilcoxon 符号秩检验、符号检验。
  4. 描述性：η13、仅安全过滤的成功率；每个场景"已评估候选中的事后最佳"（只作描述，不作判定）；近邻与全局最佳选择不同的场景数。
- 只说"在本修正族、本控制器集与本安全层下，对这 12 个新场景"的结论；瓶颈布局上近邻对 11 个布局选的是同一组修正，结论应写成"瓶颈族修正对全局修正"，不写成"逐场景自适应"；不外推到未测拓扑。

## 七、资源与调度

- 用户全部任务合计不超过 18 个 CPU 份额、2 个 GPU 份额（临时上调由用户决定）。
- Slurm worker 时限约 3 小时，在时限前约 55 分钟停止领取新任务组；每次提交后立即确认作业处于 RUNNING，几分钟后再查一次，不能只看到"已排队"。
- 调度程序自查：本人作业排队超过 15 分钟而 CPU 空闲，就撤销并以更短时限重交，写日志。**不设"若干分钟没有新结果就撤销重交"的规则**（用户 2026-10-25 已取消）；长仿真用上一轮的操作检查点在时限交接时续跑。
- 只有当其他用户的排队作业确实缺 CPU 或内存时才让出；对方在等 GPU 份额时不让出。
- 任务组按预计耗时从长到短排队，单组约 10 分钟。

## 八、交付

- `REPORT.md`（中文，第一段给出主判定、近邻与全局最佳的差值和区间，以及次要结果 1 的两个差值）；每个测试场景的方法对比表；`AXIS_TABLE.md` 按泛化框架标注（跨场景·前瞻；跨因素·几何新布局、密度）；`HISTORY_CHECK.md`、`MODEL_FROZEN.receipt.json`、各布局五层几何报告、能力门表、`PILOT_COST.md`、`GAPS.md`、数据库核对、开工与结束完整性记录。
- 无视频要求。

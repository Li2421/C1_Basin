# 新场景测试 v1：这几个新场景是否带来"现有几个 η 覆盖不了"的需求（小规模测试）：实现说明

**写作约定：** 方法、场景、控制器和候选第一次出现时写全称，括号里给代号，代号只用于定位文件；报告照此书写。候选 η 是三个数（朝目标, 侧移, 互斥），提到某个候选时同时写出这三个数。本轮只是小规模测试：只出标签与统计，不训练任何模型，不做第二阶段，耗时大的东西一律不跑。

**名称表（报告一律用这些名称）：**

| 名称 | 含义 |
|---|---|
| 每个场景自己的最好 η | 在该场景一半状态上选成功率最高的 η，在另一半状态上计分；随机对半划分 200 次取平均 |
| 现有场景最好的 k 个固定 η | 只用现有 11 个场景（第三节）的标签，挑 k 个 η，使"每个场景用这 k 个里对它最好的一个"的平均成功率最高 |
| 新场景的缺口(k) | 新场景自己的最好 η 减"现有场景最好的 k 个固定 η 里对它最好的一个"（后者在新场景同一组计分状态上计分） |
| 交叉使用损失 | 把场景 i 自己的最好 η 用到场景 j 上，比 j 自己的最好 η 低多少（同样对半划分） |
| 覆盖缺口(k) | 在一组场景上：每个场景自己的最好 η 减"这组场景上最好的 k 个固定 η"（固定 η 的挑选与计分同样只用各场景的同一半/另一半状态） |

## 零、开工条件、源码与仓库保护（任何步骤之前）

- **开工条件：** `diagnostics/cross_scene_pipeline_v2/COMPLETE.json` 存在且 `status` 为 `COMPLETE`；否则停下报告，不建目录。若 `diagnostics/cross_scene_pipeline_v2_learning_tests/` 正在运行，登记为并行任务目录。
- 本轮新代码与输出只放 `diagnostics/eta_diverse_scene_test_v1/`。已有源码、`docs/`、`AGENTS.md`、其他 `diagnostics/` 目录（含 `cross_scene_correction_v1/`、`cross_scene_pipeline_v1/`、`cross_scene_pipeline_v2/`、`baseline_literature_scenes_supplement_eta_swap8_v1/`）、`datasets/`、`bottleneck_family/`、MAC-Flow 官方仓库、虚拟环境一律只读；不安装、不升级任何包。`cross_scene_prospective_v1/` 只看 `os.stat` 元数据，不打开。
- 复用只用只读 import，或复制到本轮目录并记录 sha256；不在运行时给旧模块打补丁（JAX 精度恢复除外）。旧目录里的运行入口有写入副作用（`experiment.py` 在自身目录建缓存目录；`common.py` 的 `evidence` 追加自身目录的 `EVIDENCE_INPUTS.jsonl`），一律不在旧目录运行、不 import 旧目录模块。以 `cross_scene_pipeline_v1/build_isolated.py` 为模板，从 `cross_scene_correction_v1` 在本轮目录复制执行图（实验名 `exp_eta_diverse_scene_test_v1`）；本轮不用冲突搜索与优先级 A*，不需要重新编译冲突搜索内核（只有在导入时确实需要该内核才按模板编译，几秒钟）。`scenes.py`（`make_new`、`geometry_report`）与 `stages.py`（`capability_cases`、`capability_select`）复制后按本说明改写。Slurm 作业名前缀用 `xdt_`，撤销作业只认这个前缀。
- git 只用只读命令（两个仓库都有用户未提交的工作）。开工与结束各记录两个仓库的 `git status --porcelain`、`git diff` 的 sha256 与所有复用文件的 sha256，两次必须一致；本轮目录、并行任务目录、`docs/` 下新增的 `prompt_*.md` 单列，不算失败。
- 共享数据库：允许 (1) 官方 `cache_writer.append_journal` 写默认的 `shared_rollout_db/journals/<本轮实验 UID>/`；(2) 官方单进程 merger；(3) 单进程、官方 `shared_rollout_db.src.rollout_db` 的 connect/transaction，只对新的 experiment/scenario/state/controller_config/eta 行做 INSERT，遇到 UID 内容冲突即停，登记前后已有行逐表哈希不变。禁止改 `cache_writer.ROOT`、禁止复制数据库；开工时断言 `shared_rollout_db.__file__` 与 `cache_writer.ROOT` 都在 `/home/zhihan/research/Basin_C1/` 之下。预检与缓存规则见 `shared_rollout_db/README.md`。
- 遇到与本说明矛盾、或要求的接口不存在：停下报告，不改源码。下文点名的接口与文件均已核实存在。

## 一、要回答的问题

用户要的场景集：**几个 η 解决不了所有场景**，即用最好的 3 个固定 η，覆盖缺口(3) 仍不少于 10 个百分点。现有数据都达不到（2026-10-25/26 对共享数据库的全量只读统计）：C1 原四场景用 3 个 η 就覆盖（93.3 对 93.2）；18 个跨场景场景用 1–2 个 η 就覆盖（最好的 1/2/3 个固定 η：71.1 / 74.9 / 77.0，每个场景自己的最好 η 71.2）。

**本轮只测一件事：** 下面 4 个新场景，是否各自带来"现有场景最好的 3 个固定 η 覆盖不了"的需求，而且彼此的需求不同。符合，再另写 prompt 扩展；不符合，就到此为止，不再追加。

**旋钮的来历（已有标签中的线索）：** 圆形五机（机器人大、面积占比高）的最好 η 是 η9（0.94, 0.43, 0.43），成功率与"互斥"负相关（Spearman −0.63），小机器人对穿则偏好大互斥；八机对穿加入静态障碍后，最好 η 从扩展范围候选 sE29（0.34, 0.55, 1.20）变为 sE22（0.55, −0.20, 1.42），侧移方向反转；以前 Toy 宽度族的研究中，只有收紧时限开始打破"少数几个 η 处处成功"。

## 二、执行语义（沿用 `cross_scene_correction_v1`，不改）

- 单积分器、分布式安全投影、接触门控修正、成功判据（全部机器人在时限内到达且无碰撞；timeout 不等于死锁；碰撞按机器人/静态/移动分类；数值失败记 unknown，保留不重跑）、随机数规则；只跑种子 0；只用固定修正。
- **控制器（只用最快的 3 个）：** 集中式随机采样模型预测控制（`centralized_sampling0`）、梯度下降模型预测控制（`gradient_mpc0`）、分布式模型预测控制（`DM0`）。在 `cross_scene_pipeline_v1` 的实测中它们单局中位 1.6–2.3 秒、最长约 20 秒。**冲突搜索、优先级 A* 及其他控制器一律不跑。**
- **候选（25 个，精确 float64 值取自 `baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json`）：** 仅安全过滤（`safety`，0, 0, 0）；历史 16 个 `eta0`–`eta15`；扩展范围候选 `sE29`（0.34, 0.55, 1.20）、`sE13`（0.45, 1.34, 1.03）、`sE22`（0.55, −0.20, 1.42）、`sE05`（0.11, 0.32, 0.53）、`sE01`（0.97, 0.94, 1.41）、`sE10`（0.14, −1.29, 1.22）、`sE20`（1.21, −1.13, 0.97）、`sE24`（1.72, −0.78, 1.35）。这 25 个包含现有每个场景的最好 η，另加 3 个侧移为负的角落点。
- 新请求用复制的 `scenes.make_new` 构造（版本 `cross_scene_si_v1`）。新场景名必须以 `xsc_` 开头（`scene_environment.NewNominal` 按这个前缀把模型预测控制类控制器接到 `controllers_new.Nominal`）。
- 每个新场景用复制的 `scenes.geometry_report` 出五层报告（几何有效、单机器人有路、联合可解写"未认证"、控制器成功、死锁解决）；不合法的状态照实记录，不替换。
- **能力门（缩短版）：** 状态 0–3 的单机器人版本（每台机器人单独，种子 0 与 1），每个控制器成功率 ≥ 95%。3 个控制器都通过才测这个场景；否则该场景记为"控制器不适用"，不换控制器、不替换场景。

## 三、场景

**基础：** 与 `cross_scene_correction_v1/SCENES.json` 中对穿 6 机（`xsc_swap_n6`）完全相同的 12 个状态（位置、目标、速度）和配置（`LiteratureConfig`：机器人半径 0.05、dt 0.05、限速 0.5、`max_steps = ceil(4 × 最长直线行程时间 / 0.05)`，最长直线行程时间的算法同该轮 `scenes.prepare`）。每个新场景只改一个旋钮，取中等偏强的值：

| 新场景 | 代号 | 只改的旋钮 |
|---|---|---|
| 大机器人 | `xsc_div_size_r014` | 机器人半径 0.14 |
| 紧时限 | `xsc_div_time_f20` | `max_steps = ceil(2.0 × 最长直线行程时间 / 0.05)` |
| 中心障碍 | `xsc_div_center_s024` | 以原点为中心、半边长 0.24 的正方形障碍 |
| 偏心障碍 | `xsc_div_offset_dx030` | 半边长 0.16 的正方形障碍，中心在 (0.30, 0) |

- **障碍场景的墙体：** 由 `bottleneck_family.scenario.Geometry(Config(...))` 生成，状态里带 `bottleneck_config`：`num_agents=6, half_length=2.0, half_height=2.0, barrier_x=(1.7,), openings=(((0.0, 3.6),),), barrier_thickness=0.1, extra_rectangles=` 该障碍`, agent_radius=0.05, dt=0.05, max_speed=0.5, max_steps` 同基础场景，其余字段取默认值。屏障放在 x = 1.7、开口覆盖几乎全高，只为满足 `Config` 的校验，离机器人所在的圆很远。已核实这种构造对半边长 0.16–0.32、偏移 0–0.4 都能通过校验，按基础方式生成的状态全部有效、每台机器人都有单机器人路径（x = 1.85 会被拒绝，不要用）。若实际构造仍被拒绝，停下报告。
- **现有场景（零 rollout，只读已有标签）：** 对穿 6 机、对穿 4 机、对穿 10 机、对穿 12 机、对穿 8 机加密（`xsc_swap_n6/n4/n10/n12/n8_dense`，各 12 个状态），以及窄通道、拥挤随机、圆形五机、八机无障碍、八机静态障碍、八机移动障碍（`lit_narrow_passage`、`lit_crowded_random`、`lit_circle_five`、`lit_swap8_free`、`lit_swap8_obstacles`、`lit_swap8_moving`，各 16 个状态）。取 `cross_scene_correction_v1/LABEL_TABLES.json`，只用这 3 个控制器、这 25 个候选。

**新 rollout 总量：** 4 个场景 × 12 个状态 × 3 个控制器 × 25 个候选 × 种子 0 = 3,600 次（另加能力门约 576 次单机器人运行）。按实测成本约 2–3 CPU 小时，并行后约 15–20 分钟。超过 4,500 次就停下报告。

## 四、统计（零额外 rollout）

成功率一律为 3 个控制器等权、状态等权；unknown 计失败（另报计成功时的上界）。

1. **每个新场景：** 25 个候选的成功率表；自己的最好 η（写出三个数）与前 3 名；样本内最好成功率；按控制器分别的最好 η。
2. **新场景的缺口(k)**，k = 1、2、3：现有场景最好的 k 个固定 η 是哪几个（写出三个数），它们在每个新场景上的成功率，以及缺口(k) 与 95% 区间（新场景内按状态重抽 5000 次，`numpy.random.default_rng(2026102611)`，每次重抽内部仍做对半划分）。
3. **新场景之间、新场景与对穿 6 机之间的交叉使用损失**（两两、两个方向）。
4. **覆盖缺口(k)**，k = 1、2、3：分别在 (a) 4 个新场景 + 对穿 6 机，(b) 4 个新场景 + 全部 11 个现有场景上计算。

## 五、判定（预登记）

- **单个新场景"带来新需求"：** 样本内最好成功率在 [0.20, 0.95] 之间，且新场景的缺口(3) 点估计 ≥ 10 个百分点。最好成功率低于 0.20 或高于 0.95 的写"本测试无法判断"，不算"不符合"。
- **这批新场景"符合需求，值得扩展"：** 至少 2 个新场景带来新需求，且其中至少有一对的交叉使用损失 ≥ 10 个百分点（它们需要的 η 彼此也不同）。
- 结论写进 `TEST_DECISION.json`（全部未四舍五入的数值）。本轮无论结论如何都不追加 rollout；符合时在报告里写明哪些旋钮值得扩展，由用户决定下一步。

## 六、执行流程

1. 开工条件与完整性记录（第零节）；复制执行图；按 `cross_scene_pipeline_v1/replay.py` 的做法，逐位重放 `cross_scene_correction_v1` 已存的对穿 6 机或对穿 12 机中由这 3 个控制器跑的 2 局（不重放冲突搜索、A* 或拥挤随机），结果与轨迹逐位一致才继续。
2. 历史检索，写 `HISTORY_CHECK.md`（第一、二节所列来源的路径与可复用范围）。
3. 构造 4 个新场景，五层报告，能力门，新身份登记；写 `SCENES.json`、`GEOMETRY_REPORTS.json`、`CAPABILITY_TABLE.json`。
4. 新 rollout：官方预检后只跑真正缺失的请求；worker 写 journal，结束后单一 merger 合库。
5. 统计与判定（第四、五节）；数据库核对（所有请求在库、身份与结果一致、无未合并 journal）；结束完整性记录；写 `COMPLETE.json`。

## 七、失败处理与无人值守

- **解释器：** 主解释器 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`（有 jax、flax、optax、distrax、numpy、scipy、PIL；没有 matplotlib、pandas、scikit-learn）；`/usr/bin/python3` 没有 numpy。本轮不要求图。
- **三类失败：** (1) 整轮停止只限：违反源码或仓库保护、违反共享数据库规则、逐位重放失败、障碍场景构造被拒、新 rollout 总量超过 4,500。(2) 主线作业失败：先读日志；基础设施原因按原清单、原种子补交未完成部分，自动处理，最多 3 次；实现错误修正后只重跑受影响部分，写 `ITERATION_LOG.md`；仍不能完成的场景记为缺口，写 `GAPS.md`，其余照常。(3) 非主线（排版、可选附表）放在最后，失败写进 `GAPS.md`。
- **无人值守：** 每 30 分钟更新 `STATUS.md`（阶段、完成数、失败与补交、预计剩余时间）；执行代理至少每 30 分钟检查一次，直到交付。

## 八、资源与调度

- 用户全部任务合计不超过 18 个 CPU 份额、2 个 GPU 份额（临时上调由用户决定）。全程只用 CPU。
- Slurm worker 时限约 1 小时（本轮总量小）；每次提交后立即确认作业处于 RUNNING。本人作业排队超过 15 分钟而 CPU 空闲，就撤销并以更短时限重交，写日志。只有当其他用户的排队作业确实缺 CPU 或内存时才让出。

## 九、交付

- `REPORT.md`（中文；第一段：判定结论，每个新场景是否带来新需求、它的缺口(3) 与区间、自己的最好 η，以及两组覆盖缺口(3)）；每个新场景的 25 个候选成功率表；现有场景最好的 1/2/3 个固定 η 表；交叉使用损失表；按控制器的最好 η 表。
- `HISTORY_CHECK.md`、`SCENES.json`、`GEOMETRY_REPORTS.json`、`CAPABILITY_TABLE.json`、`TEST_DECISION.json`、`STATUS.md`、`GAPS.md`、`ITERATION_LOG.md`、数据库核对、开工与结束完整性记录、`COMPLETE.json`。
- 无视频要求。

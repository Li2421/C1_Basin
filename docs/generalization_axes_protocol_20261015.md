# 泛化轴实验共同协议（round32–36）

本文件是五个泛化轴 prompt 的共同规则：

| 轮次 | 泛化轴 | prompt |
|---|---|---|
| round32 | 1. State | `docs/prompt_round32_axis1_state_20261015.md` |
| round33 | 2. Controller | `docs/prompt_round33_axis2_controller_20261015.md` |
| round34 | 3. Cross-Scene | `docs/prompt_round34_axis3_cross_scene_20261015.md` |
| round35 | 4. Cross-Factor：几何与时间预算 | `docs/prompt_round35_axis4a_geometry_horizon_20261015.md` |
| round36 | 4. Cross-Factor：机器人数量 | `docs/prompt_round36_axis4b_agent_count_20261015.md` |

各轴 prompt 与本文件冲突时，以各轴 prompt 为准；但第一节（源码保护）与第二节（DB 规则）不得放宽。

---

## 一、源码保护（最高优先级）

用户明确要求：**不得改动现有源码。** 每一轮的全部新代码与输出，只能写在该轮 prompt 指定的新目录（下称"本轮目录"）中。

**1. 只读，不得修改、删除、移动、重命名或重新格式化：**
- Basin_C1 中所有已有文件，包括 `new_benchmark_common/`、`ring_exchange/`、`shared_control/`、`shared_rollout_db/`（代码只读；DB 写入见第二节）、`bottleneck_family/`、`toy_giveway/`、`double_bottleneck/`、`four_way_intersection/`、`datasets/`、`docs/`、`AGENTS.md`，以及 `diagnostics/` 下除本轮目录外的所有子目录；
- PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`（不在 git 中）；
- MAC-Flow 官方仓库 `/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/`，它带有用户尚未提交的改动；
- Python 环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/`：不安装、不升级任何包。缺少依赖时，在本轮目录内另建环境，或把该子项记为缺口。

**2. 复用已有代码的方式：**
- 只读 import；
- 或复制到本轮目录后再修改，并记录源文件路径与 sha256；
- 需要不同行为时，写子类或新函数。不得在运行时替换已有模块的函数或全局状态来改变它的行为；唯一的例外是按协议显式恢复 JAX 精度。

**3. git：** 两个仓库都只允许只读命令（status、diff、log、rev-parse、ls-files）。严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。

**4. 完整性审计（交付必需）：** 开始与结束时各记录一次：
- Basin_C1 的 `git status --porcelain`，以及 `git diff` 输出的 sha256；
- MACFlow_Official 的 HEAD、`git status --porcelain`，以及 `git diff` 输出的 sha256；
- PoC 目录 `poc/` 下全部 .py 文件的 sha256 清单；
- 本轮 import 或复制的每个已有源文件的 sha256。

除本轮目录外，结束时的记录必须与开始时完全一致。不一致就在 REPORT.md 第一段写明，并列出改动。

**5. 被挡住时怎么办：**
- 某个子项被挡住（缺接口、缺能用的 controller、能力门不通过）：把该子项记为缺口，写明原因与证据，继续其余子项。
- 只有当源码保护或 DB 规则无法满足时，才整体停下报告。
- 任何情况下都不要靠改源码绕过。

---

## 二、DB 规则

新目录之外唯一允许的写入，是共享 DB 的官方流程，共三处：
- **design 阶段登记新身份：** 按 round28 `diagnostics/decentralized_pipeline_v1/design.py` 的做法，用官方的 `shared_rollout_db.src.rollout_db.connect()` 与 `transaction()`，对 experiment、state、controller_config、eta 表（需要时还有 scenario 表）执行 `INSERT OR IGNORE`，只追加本轮的新身份。要求：
  - 不得 UPDATE 或 DELETE 任何已有行，也不得直接插入 rollout 行；rollout 只能经 journal 与官方 merger 入库；
  - 同一 uid 已存在且内容不同时，把相关子项记为缺口并报告；
  - 记录插入的全部行（JSON）及其 sha256；
  - 插入前后，各登记表中已有行的内容哈希必须相同。
- **journal：** worker 用 `shared_rollout_db.src.cache_writer.append_journal` 写到默认位置 `shared_rollout_db/journals/<experiment_uid>/`。
- **合并：** 用 `python -m shared_rollout_db.src.merger --experiment <experiment_uid>` 单进程合并。

`rollout.sqlite` 与 journal 目录都已被 git 忽略，不影响完整性审计。不要通过修改 `cache_writer.ROOT` 等模块全局变量来重定向 journal。

**流程：** design → preflight（在提交作业之前封存）→ 只跑 `genuinely_missing` → journal → 单进程官方 merger → postflight。已有的同身份记录一律复用，不重跑。

---

## 三、复用历史代码：规则与已知的坑

**复制规则：** 若要复用的历史执行代码在 import 时有副作用（改环境变量、往自己的目录写文件、替换全局变量），就复制到本轮目录：
- 只允许修改路径常量与环境变量，其余逻辑逐字保持不变；
- 提交与原文件的 diff，以及原文件的 sha256。

新的臂（新 η、新切换规则、nominal only 等）写成单独文件中的新函数，不改复制来的逻辑，并配单元测试。

**复现门：** 使用任何历史执行代码（原样 import 或复制版）跑新 rollout 之前，先复现 16 条该代码的历史记录。首动作字节、终止类型、步数与完整位置轨迹必须逐位一致；不一致就把依赖该代码的子项记为缺口。

**已知的坑（前几轮踩过）：**
- round28 的 `core.py` 在 import 时，会把 TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME 指向 round28 的目录。不要 import，按复制规则处理。
- round27 的 `probes.py`、`freeze_choices.py` 等脚本含全局变量替换和向其他目录的写入，不能直接运行，复制后改写。
- **PoC 目录 `source/` 下有一份旧版 `shared_rollout_db` 副本。** 如果 sys.path 中 PoC 的 `source/` 排在 Basin_C1 前面，`import shared_rollout_db` 会加载这份旧副本，journal 会写进 PoC 目录。round30 因此误写了 448 个 journal。
  - 每个入口（包括 worker）在 import 之后，都要断言 `shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/`；
  - 断言写入 journal 的路径位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/journals/`。
- PoC 的 `field_controller.py` 在 import 时开启 JAX x64。全部导入完成之后，必须显式恢复协议精度（x64 = False），并记录。
- **坐标系一致：** global 坐标系 controller 的动作在世界坐标系中，local 坐标系 controller 的动作在径向/切向坐标系中。任何用一个 controller 的输出去训练另一个模型（蒸馏、BC）时，输入观测与目标动作必须在可互相推出的坐标系中。round30 用旋转不变的 local 观测去拟合 global 教师的世界坐标动作，10 个学生因此学不会，其中 6 个停在初始化。
- PoC `nominal(env, key, traced=True)` 的内置断言容差按 float64 设定，在 x64 = False 下会因约 1e-8 的舍入差报错。不要用它做检查，也不要为通过它而改用 x64 = True。
- PoC 的 `make_env` 只接受 t0 state；需要非 t0 state 时，直接构造 env 并 reset。
- `diagnostics/orthoflow3_tt_ff_matched_learnability_v1/execution.py` 中的 `key_for` 在 import 时没有副作用，可以只读 import。
- PoC `Runtime.action(env, key, eta, chain)` 中：
  - `chain = "old"` 是 TT 位置：nominal → 投影 → 加修正 → 再投影；
  - `chain = "field"` 是 FF 位置：修正加在 flow 积分内部。
- 已有链路都没有 nominal only。需要时写新函数：直接执行 `nominal(env, key)` 返回的动作，不投影、不修正，碰撞仍由原 simulator 判定。
- safety baseline 就是对应链路中取 η = (0, 0, 0)。
- round28 的 Ring Flow 教师中，`base_u_v10_local_macflow` 用 PoC 原生 runtime，其余 13 个用 `diagnostics/controller_q_transfer_round3_v1/flow_adapter.py` 中的 `RingFlowAdapter`。

---

## 四、标准方法与指标（各轴统一）

**标准方法：** 能做的都要报告，做不了的标 N/A 并说明原因。

| 代号 | 内容 |
|---|---|
| nominal only | 只执行 controller 动作，不加安全层，不加修正 |
| safety baseline | 安全投影，η = 0 |
| F10 | 全程 η10，即旧的默认做法 |
| S500 类 | 不看 controller 的时间表，按各轴定义（Ring H700 下为 S500） |
| A3 零样本 | round16 冻结的 A3 做 controller 级选择 |
| A0 | round16 冻结的、不看 controller 的 critic |
| G_E | 规则：探测 → Ê → 选择 |
| 最佳固定（oracle） | 每个 controller 在固定菜单中事后最佳的臂 |
| 完整菜单 oracle | 每个 controller 在完整菜单中事后最佳的臂 |
| 专家 | 特权参照，只在有专家的场景报告，不作为竞争方法 |

**冻结选择：**
- A3、A0、G_E 的选择，都必须在读取任何评估标签之前完成并留收据（UTC 与 sha256）。
- 需要新的 controller 表示时，按 R5 协议探测：16 个 probe state × {η10, η13} × 200 步，关闭终止，预处理与 PCA 用 round16 冻结的版本。
- round16 的冻结模型与收据在 `diagnostics/ring_controller_critic_v1/`；用法参考 round26（`diagnostics/method_benchmark_v1/`）与 round27 的冻结选择。

**指标：**
- 成功率：全体无碰撞到达，在 horizon 内完成；
- 碰撞率、超时率、数值失败（unknown）率；
- 相对完整菜单 oracle 的平均遗憾与最大遗憾；
- 灾难数：比 F10 低 ≥ 10pp 的 controller 数；
- 伤害数：比 safety baseline 低 ≥ 10pp 的 controller 数，用来发现"加了修正反而更差"；
- 从 nominal only 到 safety baseline 的逐条转移表：nominal 下碰撞的 episode，加安全层后变成成功、timeout 与 unknown 的比例；
- 停滞诊断只作描述，不得称为 deadlock。

**oracle 的估计：** 用同一批标签既选又评估会高估。同时报告：
- 交叉拟合值（下界）：按 state 或 seed 分半，一半选、另一半评估，再交换；
- 样本内值（上界）。

**CI：** 先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次，使用配对比较。unknown 按下界计，另报上界。

**证据等级：**
- 全新标签、预注册、选择先于标签冻结的终点，标为 CONFIRMATORY；
- 其余标为 DEVELOPMENT。

---

## 五、标准交付（每一轮都要有）

- 完整性审计：开始与结束两次记录，以及二者的比对；
- PREREGISTRATION.md 及收据（UTC 与 sha256），必须早于任何新 rollout 与任何结果计算；
- 历史审计：已有标签的来源、可复用性、缺口；
- 复现门报告；
- DB 的 preflight 与 postflight 记录，以及登记行的清单；
- results.json；
- **AXIS_TABLE.md 与 AXIS_TABLE.csv**，所有轮次用同一格式：
  - 每行：一个子项（例如"新 seed""H = 1000""N = 10"）；
  - 每列：第四节的标准方法，单元格为成功率 % [95% CI]；
  - 另附：碰撞率、超时率、平均与最大遗憾、灾难数、伤害数、证据等级；
  - 没有数据的子项保留一行，标"缺口"并写明原因；
- 缺口清单：每个缺口的原因与补齐所需的条件；
- REPORT.md：第一段先写完整性审计是否通过，再直接给出该轴各子项的主要结论；
- HANDOFF.md（不依赖本对话也能读懂）。

---

## 六、其他规则

- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **精度：** 全部导入完成之后，显式恢复协议规定的精度，并记录 x64 状态。
- **不得改动核心组件：** 不修改物理、安全投影、事件判定、成功标准与已有 controller。各轴需要的新环境参数（几何、horizon、机器人数量）通过构造参数传入，或在本轮目录中写子类，原代码不改。
- **参数写死：** 设计、阈值、能力门、state 区间与种子，全部写入预注册，不得根据结果调整。
- **视频：** 不要求。若用户另行要求，按 `docs/video_delivery_optional.md` 执行。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。round32–36 可能并行，开工前与提交作业前都要用 `squeue` 查看当前总占用。
- **不做 git commit。**

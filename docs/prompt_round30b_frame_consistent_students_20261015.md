你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/teacher_student_macflow_v1b/`

# 零、源码保护（最高优先级，先于本文件其他所有要求）

用户明确要求：**不得改动现有源码。** 本轮的全部新代码与输出，只能写在本轮目录中。

**1. 只读，不得修改、删除、移动、重命名或重新格式化：**
- Basin_C1 中所有已有文件，包括 `new_benchmark_common/`、`ring_exchange/`、`shared_control/`、`shared_rollout_db/`（代码只读；DB 写入见第六节）、`docs/`、`AGENTS.md`，以及 `diagnostics/` 下除本轮目录外的所有子目录。其中包括 round30 的目录 `diagnostics/teacher_student_macflow_v1/`：它的代码、checkpoint 与结果只能读取或复制，不得改动。
- PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`（不在 git 中）；
- MAC-Flow 官方仓库 `/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/`，它带有用户尚未提交的改动；
- Python 环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/`：不安装、不升级任何包。绘图可用已有的 `.venv-c1-plots`。

**2. 复用已有代码的方式：**
- 只读 import；
- 或复制到本轮目录后再修改，并记录源文件路径与 sha256；
- 需要不同行为时，写子类或新函数。不得在运行时替换已有模块的函数或全局状态；唯一的例外是按协议显式恢复 JAX 精度。

**3. git：** 两个仓库都只允许只读命令（status、diff、log、rev-parse、ls-files）。严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。

**4. 防止写错 DB（round30 的教训）：**
- PoC 目录的 `source/` 下有一份旧版 `shared_rollout_db`。round30 的 worker 加载了它，把 448 个 journal 写进了 PoC 目录；这些文件已由用户确认删除。
- 本轮每个入口与 worker，在 import 之后都必须断言：`shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/`；写入 journal 的路径位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/journals/`。断言失败就停止该进程，不写任何文件。

**5. 完整性审计（交付必需）：** 开始与结束时各记录一次：
- Basin_C1 的 `git status --porcelain`，以及 `git diff` 输出的 sha256；
- MACFlow_Official 的 HEAD、`git status --porcelain`，以及 `git diff` 输出的 sha256；
- PoC 目录下全部文件的路径与 sha256 清单，不只 .py；
- round30 目录下全部文件的路径与 sha256 清单；
- 本轮 import 或复制的每个已有源文件的 sha256。

除本轮目录外，结束时的记录必须与开始时完全一致。git 状态中其他并行任务新增的条目，单独列出并注明不是本轮所写。

**6. 被挡住时停下报告。** 如果已有代码缺少需要的接口，或者只有改源码才能继续，就停下报告，不要绕过。

# 一、背景与问题

**round30**（`diagnostics/teacher_student_macflow_v1/`）效仿 MAC-Flow 第二阶段，把 14 个冻结的联合 MACFlow 教师各蒸馏成一个分散式一步学生，在不通信、只知道自己目标的设定下评估。

**它的设计有错误：**
- 学生的输入是旋转不变的 local 观测行。
- 但 14 个教师中有 10 个是 global 坐标系，它们的动作 `raw` 在世界坐标系中。学生从旋转不变的输入，推不出世界坐标系中的方向，因此学不会：
  - 6 个按 dev 误差选中了初始化；
  - 另外 4 个与教师动作的余弦只有 0.10–0.48；
  - 这 10 个学生在 nominal only 与 safety baseline 下成功率都是 0%。
- 4 个 local 坐标系教师（v7、v8、v9、v10_local）的学生是有效的：余弦约 0.98，98% 的样本误差 ≤ 0.05 m/s。

**本轮：**
- 用坐标系一致的方式，重做这 10 个 global 教师的学生；
- 与 round30 的 4 个有效学生合起来，得到 14 个有效学生；
- 用 round30 预注册的同一组终点重新检验。

# 二、学生的构造（只改坐标系，其余与 round30 相同）

**与 round30 相同的部分：**
- 10 个 global 教师（以 round30 的 TEACHERS.json 为准），冻结，先核对 hash；
- 学生网络与训练设置：官方 `actor_onestep_flow` 形式，输入为机器人编号 one-hot、自身 local 观测行、自身 2 维噪声；超参数、步数、batch、dev 选择规则、seed 规则，全部与 round30 的 TRAINING_CONFIG.json 相同；
- 训练数据：每个教师自己训练集的 state，标签由教师生成；
- 教师动作与噪声的取法：教师动作取 `native_fn(obs, k)` 的输出，噪声 ξ 取 `trace_fn(obs, k)` 的第 0 步。若本轮使用的代码与 round30 噪声一致性门所用代码的 hash 相同，就直接引用 round30 的门结果（1,008/1,008 通过）；否则重跑该门。

**坐标系一致的做法（只用机器人自己的位置，不需要通信）：**
- 记 R_i 为机器人 i 的径向/切向坐标系，按 `ring_exchange/local_frame.py` 中 `frames` 的定义；该文件中的转换都是逐行的，每行只用机器人自己的位置。
- **目标动作：** u_i^loc = `world_actions_to_local`(u^world, positions) 的第 i 行。
- **噪声输入：** z_i^loc = R_iᵀ ξ_i，即把教师在世界坐标中的噪声分量，旋转到机器人 i 的坐标系。
- **部署：**
  - 机器人 i 直接抽 z_i^loc ~ N(0, I)，随机键用 `jax.random.fold_in(key_for(state, seed, step), i)`，与 round30 相同。各向同性的高斯分布旋转后不变，所以分布与训练一致。
  - 学生输出经 `local_actions_to_world` 转回世界坐标，再按教师的方式限速。
- 记录每个教师的动作归一化尺度 `act_scale` 是否各向同性。若不是，噪声的旋转只是近似，如实报告。

**离线诊断（训练前，描述性）：**
- 对每个 global 教师，在 256 个数据集 state 上，比较原 state 与随机旋转后 state 的平均动作（每个 state 用 16 个噪声样本取平均），计算旋转等变误差相对动作大小的比例。
- 它用来解释哪些教师难以分散化，不作为门。

**蒸馏有效性门（预注册，在任何闭环运行之前判定）：** 以下三条全部满足才算有效学生：
- 选中的 checkpoint 不是 step 0；
- 在 round30 的独立 dev 检查（2,048 个固定的 state 与噪声样本，转换到同一坐标系后比较）上，与教师动作的平均余弦 ≥ 0.90；
- 误差 ≤ 0.05 m/s 的样本占比 ≥ 80%。

未通过的学生记为无效并报告，不重训、不调参、不替换。

**单元测试：**
- 坐标转换逐行只依赖机器人自己的位置；
- 改变其他机器人的目标，学生输出不变；
- 给定观测与噪声时，输出是确定的；
- 把本轮的转换流程用在一个 local 教师上，应与 round30 的 local 学生设定一致。

# 三、执行与臂（与 round30 主评估相同）

- **管线：** round28 的 D-local 安全层与本地修正。按 round30 的做法，把 round28 的 `pipeline.py`、`run.py` 的 `episode` 及所需的 `core.py` 部分复制到本轮目录，只改路径与环境变量，并提交 diff。复现门：复现 round28 的 16 条 D-local 记录，必须逐位一致。
- **学生接入：** 包装成 runtime 对象，实现 `nominal(env, key)`。复制版的 `decentralized_action` 一行不改；nominal only 与 B-RHR 的新函数，从 round30 复制。
- **state 与 seed：** round28 的 32 个评估 state，seed 0–1，与 round30 及教师逐条配对。
- **臂：** nominal only、safety baseline、F10、F13、S300、S500、B-RHR（D-local），共 7 个。
- **能力门：** 用 round30 的 16 个 pilot state，safety baseline 与 S500 各跑 seed 0；规则与 round30 相同。
- **探测与冻结选择（在任何评估标签之前完成并留收据）：** R5 探测（D-local）；A3、A0、G_E、一律 S500、一律 F10。代码从 round30 复制。
- **复用：** 教师的补充臂，以及 round30 的 4 个有效学生的全部标签与冻结选择，都已在 DB 与 round30 目录中，直接复用，不重跑。
- **规模：** 10 × 32 × 2 × 7 = 4,480 条主评估，另有能力门 320 条与探测。

# 四、分析与预注册终点

终点定义与阈值都沿用 round30 的预注册，另加 TS6。CI 先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次，使用配对比较。unknown 按下界计，另报上界。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| TS1 | 问题是否存在 | S500 − safety baseline 的 CI 下界 > 0 |
| TS2 | 时间表优势 | 最佳时间表 − 最佳固定 η 的 CI 下界 > 0，且 ≥ 70% 的学生满足 S(最佳) ≥ max(F10, F13) − 1pp |
| TS3a | 方法层面的零样本 | A3 相对四臂 oracle 的平均遗憾 ≤ 3pp，且 A3 − safety baseline 的 CI 下界 > 0 |
| TS3b | critic 是否利用了 controller 信息（固定菜单） | 前提：至少 3 个学生 F10 比 F13 高 ≥ 10pp，且至少 3 个学生反过来；否则记为 INCONCLUSIVE。判定：A3 的遗憾低于 A0，配对 CI 上界 < 0 |
| TS6（新增） | 是否优于经典的分散式死锁处理 | S500 − B-RHR 的 CI 下界 > 0 |
| TS4、TS5（描述性） | 继承与代价 | 同 round30 |

**两套分析集：**
- **主分析（CONFIRMATORY）：** 本轮通过蒸馏有效性门的学生，标签全新。
- **合并分析（论文用，DEVELOPMENT）：** 主分析的学生，加上 round30 的 4 个有效 local 学生。后者的标签此前已打开过。

**另报（描述性）：** 4 个有效 local 学生的分析显示，其中 2 个在完全不加安全层时也全部超时，而且没有碰撞。所以每个学生都要报告：
- nominal only 下的结果构成：成功、碰撞、timeout；
- 从 nominal only 到 safety baseline 的逐条转移表；
- 失败来源的拆分：
  - **自身停滞：** nominal only 下已经 timeout 且无碰撞；
  - **安全层导致的停滞：** nominal only 下碰撞，加安全层后 timeout；
- 每类失败被修正（F13、S500）与 B-RHR 各救回了多少。

round30 中那 10 个无效学生的结果，在本轮报告中注明"已被取代"，不删除原文件。

# 五、结论映射

| 观察结果 | 结论 |
|---|---|
| TS1、TS6 通过 | 在专家无法部署、不通信的分散式学生上，修正是必要的，并优于经典的右手规则 |
| TS2 不通过 | 阶段结构是联合 Flow 教师的特性，分散式学生用固定的 η13 就够；一律 S500 仍是安全的默认选择 |
| TS3a 不通过，且 A3 系统性地选错 | 从 Flow 学到的关系在这一类型变化下负迁移，零样本时应优先用稳健的默认动作 |
| 多个学生未通过蒸馏有效性门 | 这些 global 教师不够旋转等变，分散化需要给学生全局坐标的自身观测；结合离线等变诊断报告 |
| "自身停滞"占失败的大多数 | 分散式学生的主要问题是彼此协调不起来，而不是被安全层卡住；论文中的问题陈述要相应调整 |

# 六、其他硬性规则

- **DB 写入：** 新目录之外唯一允许的写入，是共享 DB 的官方流程，共三处：
  - **design 阶段登记新身份：** 按 round28 `design.py` 的做法，用官方的 `connect()` 与 `transaction()` 执行 `INSERT OR IGNORE`，只追加本轮的新身份。要求：
    - 不得 UPDATE 或 DELETE 任何已有行，也不得直接插入 rollout 行；
    - 同一 uid 已存在且内容不同时，停下报告；
    - 记录插入的全部行及其 sha256；
    - 插入前后，各登记表中已有行的内容哈希必须相同。
  - **journal：** 用 Basin_C1 的 `shared_rollout_db.src.cache_writer.append_journal` 写到默认位置，并满足第零节第 4 条的断言。
  - **合并：** 用 `python -m shared_rollout_db.src.merger --experiment <experiment_uid>` 单进程合并。

  流程：design → preflight（在提交作业之前封存）→ 只跑缺失 → journal → 单进程官方 merger → postflight。
- **不得改动核心组件：** 不修改物理、事件判定、成功标准、round28 的安全层与修正实现。教师冻结。
- **参数写死：** 坐标转换、蒸馏有效性门、臂、能力门与阈值，全部写入预注册，不得根据闭环结果调整。
- **冻结时序：** 学生训练、checkpoint 选择、蒸馏有效性门、探测与冻结选择，都必须在任何评估标签之前完成并留收据。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock；停滞诊断只作描述。
- **精度：** 全部导入完成之后，显式恢复协议规定的精度（x64 = False），并记录。
- **视频：** 本轮不要求视频。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。其他轮次（例如 round32）可能正在运行，提交作业前用 `squeue` 查看当前总占用。

# 七、交付

全部交付物放在本轮目录：
- 源码完整性审计：开始与结束两次记录及比对；
- PREREGISTRATION.md 及收据；
- 坐标转换的实现、单元测试与离线等变诊断；
- 学生训练代码、config、checkpoint、训练曲线；
- 蒸馏报告：含有效性门的逐学生结果；
- 复现门、能力门、探测特征、冻结选择及收据；
- DB 的 preflight 与 postflight 记录，以及登记行清单；
- results.json；
- 论文用总表：主分析与合并分析各一张，列出 nominal only、safety baseline、F10、F13、S300、S500、B-RHR、A3 零样本、一律 S500、oracle 的成功率与 CI，另附碰撞率、超时率；
- 失败来源拆分表（自身停滞与安全层导致的停滞）；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过（含第零节第 4 条的 DB 路径断言）；
- 10 个新学生中有几个通过蒸馏有效性门；
- TS1、TS2、TS3a、TS3b、TS6 在主分析与合并分析中的结论；
- 分散式学生的失败主要来自自身停滞，还是安全层导致的停滞。

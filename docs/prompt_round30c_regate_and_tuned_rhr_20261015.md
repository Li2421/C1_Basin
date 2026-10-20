你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/teacher_student_macflow_v1c/`

# 零、源码保护（最高优先级，先于本文件其他所有要求）

用户明确要求：**不得改动现有源码。** 本轮的全部新代码与输出，只能写在本轮目录中。

**1. 只读，不得修改、删除、移动、重命名或重新格式化：**
- Basin_C1 中所有已有文件，包括 `new_benchmark_common/`、`ring_exchange/`、`shared_control/`、`shared_rollout_db/`（代码只读；DB 写入见第六节）、`docs/`、`AGENTS.md`，以及 `diagnostics/` 下除本轮目录外的所有子目录。其中包括 round30 的 `diagnostics/teacher_student_macflow_v1/` 与 round30b 的 `diagnostics/teacher_student_macflow_v1b/`：它们的代码、checkpoint 与结果只能读取或复制。
- PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`（不在 git 中）；
- MAC-Flow 官方仓库 `/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/`，它带有用户尚未提交的改动；
- Python 环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/`：不安装、不升级任何包。绘图可用已有的 `.venv-c1-plots`。

**2. 复用已有代码的方式：**
- 只读 import；
- 或复制到本轮目录后再修改，并记录源文件路径与 sha256；
- 需要不同行为时，写子类或新函数。不得在运行时替换已有模块的函数或全局状态；唯一的例外是按协议显式恢复 JAX 精度。

**3. git：** 两个仓库都只允许只读命令（status、diff、log、rev-parse、ls-files）。严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。

**4. 防止写错 DB：**
- PoC 目录的 `source/` 下有一份旧版 `shared_rollout_db`。
- 每个入口与 worker 在 import 之后都必须断言：`shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/`；写入 journal 的路径位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/journals/`。
- 断言失败就停止该进程，不写任何文件。

**5. 完整性审计（交付必需）：** 开始与结束时各记录一次：
- Basin_C1 的 `git status --porcelain`，以及 `git diff` 输出的 sha256；
- MACFlow_Official 的 HEAD、`git status --porcelain`，以及 `git diff` 输出的 sha256；
- PoC 目录、round30 目录、round30b 目录下全部文件的路径与 sha256 清单；
- 本轮 import 或复制的每个已有源文件的 sha256。

除本轮目录外，结束时的记录必须与开始时完全一致。git 状态中其他并行任务新增的条目，单独列出并注明不是本轮所写。

**6. 被挡住时停下报告。** 如果已有代码缺少需要的接口，或者只有改源码才能继续，就停下报告，不要绕过。

# 一、背景与问题

**前两轮的经过：**
- **round30**（`teacher_student_macflow_v1`）：把 14 个联合 MACFlow 教师各蒸馏成分散式一步学生。其中 10 个 global 教师的学生因坐标系错误而无效，只有 4 个 local 学生有效。
- **round30b**（`teacher_student_macflow_v1b`）：用坐标系一致的方式重做了那 10 个学生。坐标系修正有效：余弦升到 0.76–0.95，误差 ≤ 0.05 m/s 的样本占比升到 90–98%。但只有 2 个学生（base_u、v2）通过蒸馏有效性门，其余 8 个都卡在"平均余弦 ≥ 0.90"这一条上。

**round30b 门的问题：**
- 平均余弦是在全部样本上算的，其中包括接近零的动作，例如已到达目标的机器人。这些动作的方向没有意义，会把平均值拖低。
- 例如 plus_w_v5：动作 RMSE 为 0.0136 m/s，与 round30 的有效 local 学生（0.011–0.012）相当，98.3% 的样本误差在 0.05 m/s 以内，但平均余弦只有 0.85。
- 这 8 个学生没有进入闭环评估，**至今没有任何闭环标签**。所以现在修正门的定义，不会受到它们闭环结果的影响。

**round30b 的另一个问题：** 经典的右手规则（R28 的参数：0.15 m/s，2 秒触发）在全部 6 个有效学生上，一条失败都没救回。审稿人会认为这个基线没调好。

**本轮做四件事：**
1. 用修正后的门，重新判定这 8 个学生；
2. 对通过的学生，按与 round30b 相同的协议做闭环评估；
3. 加一个调过参的右手规则基线；
4. 若坐标系一致的新学生达到 8 个，就做确认性检验。

# 二、修正后的蒸馏有效性门

**只用于这 8 个学生，在计算任何新指标之前写入预注册。**
- **数据：** round30b 的独立 dev 检查样本：2,048 个 state 索引与噪声键，生成方式见 round30b 预注册第 11 行。目标动作与噪声按 round30b 的方式转到机器人自己的坐标系。
- **checkpoint：** 直接使用 round30b 已冻结的 `students/<teacher>/best.pkl`，先核对 hash。不重训、不换 checkpoint。
- **判定（三条全部满足）：**
  1. 选中的 checkpoint 不是 step 0（与 round30b 相同）；
  2. **按动作大小筛选后的平均余弦 ≥ 0.90：** 只在教师动作大小 ≥ 0.10 m/s 的样本上计算（逐机器人判断，约为最大速度 0.52 m/s 的五分之一）；
  3. 误差 ≤ 0.05 m/s 的样本占比 ≥ 80%（与 round30b 相同，在全部样本上计算）。
- **只改了第 2 条的定义，阈值都不变。**
- **描述性报告：**
  - 对全部 14 个学生（含已有效的 4 个 local 与 2 个 round30b 学生）报告新指标；
  - 另报动作阈值取 0.05 与 0.20 m/s 时的结果，作为敏感性分析，不改变判定。
- 已经有效的 6 个学生，维持原判定不变。

# 三、闭环评估

## 3.1 与 round30b 相同的部分

- **管线与代码：** 从 round30b 复制学生 runtime、D-local 管线、episode、nominal only 与 B-RHR 的实现，只改路径并提交 diff。
  - 若复制版与 round30b 已验证的版本 hash 相同，就引用 round30b 的 round28 复现门结果；
  - 否则重新复现 round28 的 16 条记录，必须逐位一致。
- **能力门：** 对通过修正门的学生，在 round30 的 16 个 pilot state 上跑 safety baseline 与 S500，seed 0；规则与 round30 相同。
- **探测与冻结选择：** 在任何评估标签之前完成并留收据：R5 探测（D-local）；A3、A0、G_E、一律 S500、一律 F10。
- **state 与 seed：** round28 的 32 个评估 state，seed 0–1。
- **原有的 7 个臂：** nominal only、safety baseline、F10、F13、S300、S500、B-RHR（R28 参数，D-local）。

## 3.2 新增：调过参的右手规则（B-RHR-tuned）

- **为什么要改触发条件：** 原右手规则只在机器人连续 2 秒速度低于 0.01 m/s 时才触发。但在有效学生的失败中，机器人几乎从不停住：round30 中 4 个有效 local 学生在 nominal only 下的 145 次 timeout 没有一次出现这样的低速段，在 safety baseline 下的 150 次 timeout 中只有 3 次。它们仍在移动，只是到不了目标，所以原规则几乎不会启动。只调速度大小与触发时长，解决不了这个问题。
- **参数网格（共 20 组）：**

  | 触发方式 | 触发时长 | 切向速度大小 | 组数 |
  |---|---|---|---|
  | 一直开启：每一步都叠加，相当于"全体按同一方向绕行"的约定 | — | {0.10, 0.15, 0.25, 0.35} m/s | 4 |
  | 低速触发：连续低于 0.05 m/s | {1, 2} s | 同上 | 8 |
  | 无进展触发：在该时长内，到目标的距离减少不足 0.05 m | {1, 2} s | 同上 | 8 |

  触发型的规则在机器人恢复速度或恢复进展后停止叠加；同时报告原 R28 规则（0.01 m/s、2 s、0.15 m/s），作为"B-RHR"。
- **固定不变：** 顺时针方向（绕中心障碍的切向 (p_y, −p_x)/‖p‖）、"未到达目标"条件、D-local 安全层；不加 OrthoFlow3 修正。叠加后再经该机器人自己的安全投影。
- **调参用的 state：** 16 个新的调参 state，种子区间写入预注册，并与 DB、round28 评估 state、round30 pilot state 去重。
- **调参用的学生：** 全部有效学生，即已有的 6 个加本轮通过的；seed 0。
- **选择规则：** 选各学生平均成功率最高的一组；平局时依次取速度更小、时长更短的。选定后冻结并留收据，之后才跑主评估中的这个臂。
- **主评估：** 对全部有效学生（含已有的 6 个），在 32 个评估 state × seed 0–1 上跑 B-RHR-tuned。
- **说明：** 右手规则是在学生身上调参的，而我们的 η 是在联合 Flow 教师上确定的，没有针对学生调过。所以这个比较对基线是偏宽松的。

## 3.3 复用与规模

- **复用：** 已有 6 个有效学生在 round30、round30b 中的全部标签与冻结选择；教师的补充臂。
- **规模（上限）：**
  - 新学生主评估：8 × 32 × 2 × 8 = 4,096 条；
  - 已有 6 个学生的 B-RHR-tuned：6 × 32 × 2 = 384 条；
  - 调参：20 × 最多 14 × 16 = 4,480 条；
  - 能力门 256 条，另有探测。

# 四、分析与预注册终点

**分析集：**
- **主分析（CONFIRMATORY）：** "坐标系一致的新学生"，即 round30b 的 2 个，加本轮通过修正门与能力门的学生。
  - 它们都在同一组预注册终点下评估；本轮只是在修正门之后补入学生，这一点要如实写明。
  - 少于 8 个时，全部终点记为 INCONCLUSIVE，只做描述。
- **敏感性分析：** 只用本轮新增的学生。
- **合并分析（论文用，DEVELOPMENT）：** 主分析的学生，加上 round30 的 4 个有效 local 学生。

**终点：** 定义与阈值都沿用 round30 与 round30b，另加 TS6b。CI 先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次，使用配对比较。unknown 按下界计，另报上界。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| TS1 | 问题是否存在 | S500 − safety baseline 的 CI 下界 > 0 |
| TS2 | 时间表优势 | 最佳时间表 − 最佳固定 η 的 CI 下界 > 0，且 ≥ 70% 的学生满足 S(最佳) ≥ max(F10, F13) − 1pp |
| TS3a | 方法层面的零样本 | A3 相对四臂 oracle 的平均遗憾 ≤ 3pp，且 A3 − safety baseline 的 CI 下界 > 0 |
| TS3b | critic 是否利用了 controller 信息（固定菜单） | 前提：至少 3 个学生 F10 比 F13 高 ≥ 10pp，且至少 3 个反过来；否则 INCONCLUSIVE。判定：A3 的遗憾低于 A0，配对 CI 上界 < 0 |
| TS6 | 是否优于原右手规则 | S500 − B-RHR 的 CI 下界 > 0 |
| TS6b（新增） | 是否优于调过参的右手规则 | S500 − B-RHR-tuned 的 CI 下界 > 0 |
| TS4、TS5（描述性） | 继承与代价 | 同 round30 |

**另报（描述性）：**
- 与 round30b 相同的失败来源拆分：自身停滞（nominal only 下已 timeout 且无碰撞），以及安全层导致的停滞（nominal only 下碰撞、加安全层后 timeout）；
- 每类失败被 F13、S500、B-RHR、B-RHR-tuned 各救回了多少；
- 一律 S500 与 A3 的配对差。

# 五、结论映射

| 观察结果 | 结论 |
|---|---|
| TS1、TS6b 通过 | 在专家无法部署、不通信的分散式学生上，修正是必要的，而且优于调过参的经典右手规则 |
| TS6 通过但 TS6b 不通过 | 原来的优势部分来自右手规则没调好；论文要用调参后的版本做比较，并强调我们的 η 不需要针对学生调参 |
| TS2 不通过 | 阶段时间表是联合 Flow 教师的特性；分散式学生用固定的 η13 就够，一律 S500 仍是安全的默认选择 |
| TS3a 不通过，且 A3 系统性选错 | 从联合 Flow 学到的 critic 不能迁移到分散式学生；零样本时应使用稳健的默认动作 |
| 自身停滞占多数 | 分散式学生的主要问题是彼此协调不起来，安全层只是另一部分原因；论文中的问题陈述要相应调整 |
| 修正门下仍有多个学生不合格 | 报告这些 global 教师难以分散化的证据（结合 round30b 的等变诊断） |

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
- **不得改动核心组件：** 不修改物理、事件判定、成功标准、round28 的安全层与修正实现。教师与学生 checkpoint 都冻结。
- **参数写死：** 修正门的定义与阈值、右手规则的调参网格与选择规则、臂、能力门，全部写入预注册，不得根据闭环结果调整。
- **冻结时序：**
  - 修正门的判定，先于这 8 个学生的任何闭环运行；
  - 探测与冻结选择，先于任何评估标签；
  - 右手规则的调参选择，先于主评估中 B-RHR-tuned 的任何标签。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock；停滞诊断只作描述。
- **精度：** 全部导入完成之后，显式恢复协议规定的精度（x64 = False），并记录。
- **视频：** 本轮不要求视频。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 20 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。其他轮次（例如 round33）可能正在运行，提交作业前用 `squeue` 查看当前总占用。

# 七、交付

全部交付物放在本轮目录：
- 源码完整性审计：开始与结束两次记录及比对；
- PREREGISTRATION.md 及收据；
- 修正门的计算与逐学生结果（含敏感性分析）；
- 右手规则调参的完整网格结果与冻结的选择；
- 复现门（或引用）、能力门、探测特征、冻结选择及收据；
- DB 的 preflight 与 postflight 记录，以及登记行清单；
- results.json；
- 论文用总表：主分析、敏感性分析与合并分析各一张，列出 nominal only、safety baseline、F10、F13、S300、S500、B-RHR、B-RHR-tuned、A3 零样本、一律 S500、oracle 的成功率与 CI，另附碰撞率、超时率；
- 失败来源拆分与救回计数表；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过（含第零节第 4 条的 DB 路径断言）；
- 8 个学生中有几个通过修正门，主分析共有几个学生；
- TS1、TS2、TS3a、TS3b、TS6、TS6b 在三套分析中的结论；
- 调参后的右手规则与 S500 相比如何。

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/teacher_student_macflow_v1d/`

# 零、源码保护（最高优先级，先于本文件其他所有要求）

用户明确要求：**不得改动现有源码。** 本轮的全部新代码与输出，只能写在本轮目录中。

**1. 只读，不得修改、删除、移动、重命名或重新格式化：**
- Basin_C1 中所有已有文件，包括 `new_benchmark_common/`、`ring_exchange/`、`shared_control/`、`shared_rollout_db/`（代码只读；DB 写入见第六节）、`docs/`、`AGENTS.md`，以及 `diagnostics/` 下除本轮目录外的所有子目录。特别是 round30、30b、30c 的目录（`teacher_student_macflow_v1`、`_v1b`、`_v1c`）：只能读取或复制。
- PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`（不在 git 中）；
- MAC-Flow 官方仓库 `/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/`，它带有用户尚未提交的改动；
- Python 环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/`：不安装、不升级任何包。绘图可用已有的 `.venv-c1-plots`。

**2. 复用已有代码的方式：**
- 只读 import；
- 或复制到本轮目录后再修改，并记录源文件路径与 sha256；
- 需要不同行为时，写子类或新函数。不得在运行时替换已有模块的函数或全局状态；唯一的例外是按协议显式恢复 JAX 精度。

**3. git：** 两个仓库都只允许只读命令（status、diff、log、rev-parse、ls-files）。严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。

**4. 防止写错 DB：** 每个入口与 worker 在 import 之后都必须断言：
- `shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/`；
- 写入 journal 的路径位于 `/home/zhihan/research/Basin_C1/shared_rollout_db/journals/`。

PoC 的 `source/` 下有一份旧副本，不得加载。断言失败就停止该进程，不写任何文件。

**5. 完整性审计（交付必需）：** 开始与结束时各记录一次：
- Basin_C1 的 `git status --porcelain`，以及 `git diff` 输出的 sha256；
- MACFlow_Official 的 HEAD、`git status --porcelain`，以及 `git diff` 输出的 sha256；
- PoC 目录与 round30、30b、30c 目录下全部文件的路径与 sha256 清单；
- 本轮 import 或复制的每个已有源文件的 sha256。

除本轮目录外，结束时的记录必须与开始时完全一致。git 状态中其他并行任务新增的条目，单独列出并注明不是本轮所写。

**6. 被挡住时：** 某个阶段被挡住（缺接口、缺数据），就把该阶段记为缺口并写明原因，继续其余阶段。只有源码保护或 DB 规则无法满足时，才整体停下报告。任何情况下都不要靠改源码绕过。

# 一、背景与问题

round30、30b、30c 把 14 个联合 MACFlow 教师蒸馏成分散式一步学生，在不通信的设定下评估，得到 14 个有效学生（round30c 合并分析）：

| 臂 | 成功率 |
|---|---|
| nominal only | 11.7% |
| safety baseline | 11.8% |
| 右手规则 | 11.8% |
| F13 | 89.1% |
| S500 | 94.2% |

其中 10 个 global 学生在 nominal only 与 safety baseline 下都是 0%。

**还没有回答的三个问题：**
1. **修正本身能做到多少？** 学生自己完全完成不了任务，而加上修正就有约 94%。需要检验"根本不用学生"（nominal 动作取 0）和"最简单的直奔目标 controller"加上同样的修正后，能做到多少。否则无法区分：是修正修好了学生，还是修正本身就能完成 Ring 任务。
2. **公平的右手规则：** round30c 实际用的是旧的 12 组调参网格（低速阈值 0.01 m/s）。12 组的调参成功率完全相同（12.05%），说明规则从未触发；"调参后的右手规则"实际就等于 safety baseline。原因是学生失败时机器人几乎从不停住：一直在动，只是到不了目标。需要按改正后的触发方式重做。
3. **整套 pipeline 有没有空间：** 用户希望在分散式下试一下"生成器给候选、critic 挑选"的整套方法。在现有菜单中，14 个学生的 oracle 为 94.4%，一律 S500 为 94.2%，几乎没有挑选空间。需要先在生成器的全部 16 个候选（K16）上测空间，再决定是否值得在分散式数据上训练 critic。

# 二、阶段 A：修正本身能做到多少（最先做）

**两个不含学习的对照 controller**（写在本轮目录中，配单元测试）：
- **Z：零动作。** 每个机器人的 nominal 动作恒为 0。这时机器人的运动完全来自安全投影与修正。
- **G：直奔目标。** u_i = 指向自己目标的向量，长度上限为 0.52 m/s，即 `bounded_rows(goal_i − p_i, max_speed)`。这是最简单的分散式 controller。

**管线：**
- **D-local：** 与学生相同的分散式安全层与本地修正。从 round30c 复制，只改路径，提交 diff。
- **集中式 TT：** 与 round28 中教师的集中式 TT 相同（PoC `Runtime.action` 的 `chain = "old"`）。写一个 runtime 子类，只替换 nominal。用来与教师在集中式下的结果对比。

**臂：** safety baseline（η = 0）、F10、F13、S300、S500。

**state 与 seed：** round28 的 32 个评估 state，seed 0–1，与学生和教师逐条配对。Z 与 G 都是确定性的，两个 seed 的结果应当相同，照跑两个 seed 并核对。

**规模：** 2 个 controller × 2 条管线 × 5 个臂 × 32 × 2 = 1,280 条。

**预注册的解读规则：** 每条管线分别判定，并报告配对差的 CI。

| 比较 | 结果 | 结论 |
|---|---|---|
| D-local：Z + S500 vs 14 个学生 + S500 | Z ≥ 学生 − 5pp | 修正本身就能完成 Ring 任务，学生的贡献不显著 |
| | Z ≤ 学生 − 20pp | 学生与修正有协同：修正是在修学生 |
| | 介于两者之间 | 部分协同 |
| 集中式 TT：Z + S500 vs 教师 + S500（round28：93.1%） | 同上三条规则 | 同上 |
| G 加修正与学生加修正 | 描述性 | 学到的 controller 比最简单的 controller 多了什么 |

# 三、阶段 B：公平的右手规则

**参数网格（共 20 组）：** 方向固定为顺时针，即绕中心障碍的切向 (p_y, −p_x)/‖p‖；条件"未到达目标"不变。

| 触发方式 | 触发时长 | 切向速度大小 | 组数 |
|---|---|---|---|
| 一直开启：每一步都叠加，相当于"全体按同一方向绕行"的约定 | — | {0.10, 0.15, 0.25, 0.35} m/s | 4 |
| 低速触发：连续低于 0.05 m/s | {1, 2} s | 同上 | 8 |
| 无进展触发：在该时长内，到目标的距离减少不足 0.05 m | {1, 2} s | 同上 | 8 |

- 触发型规则在机器人恢复速度或恢复进展后停止叠加。
- 叠加后再经该机器人自己的安全投影。不加 OrthoFlow3 修正。

**调参：**
- 用 round30c 的 16 个调参 state，seed 0。
- 分两组分别调参：一组用全部 14 个有效学生，一组用直奔目标 controller G。G 加分散式 CBF 加右手规则，正是经典文献中的分散式设定。
- 各自选平均成功率最高的一组；平局时依次取速度更小、时长更短的。选定后冻结并留收据，之后才跑主评估。
- 规模：20 × 15 × 16 = 4,800 条。

**主评估：** 选定的规则（记为 RHR-fair）在 32 个评估 state × seed 0–1 上，分别用于 14 个学生与 G。规模：(14 + 1) × 64 = 960 条。

# 四、阶段 C：K16 上的空间（整套 pipeline 的第一步）

**候选：**
- 从历史资产中找到 controller 研究使用的冻结 K16 列表，例如 round2（`diagnostics/controller_eta_selection_db_audit_v1`）与 round15 的协议。
- 核对其中第 10、13 个与共享 DB 中 η10、η13 的精确 float64 值一致。
- 找不到唯一一致的列表时，把阶段 C 记为缺口，继续其余阶段。

**评估：**
- 每个候选都作为全队固定的 η（与现有 F10、F13 的用法相同），用于 14 个有效学生与零动作 controller Z。
- 已有的 F10、F13、η = 0 直接复用。
- 32 个评估 state × seed 0–1。
- 规模：约 (14 + 1) × 14 × 64 = 13,440 条，减去可复用的部分。

**空间的计算（交叉拟合，同时报告样本内上界）：**
- **按 controller 选：** 每个学生在 K16 ∪ {S300, S500} 中选最好的一臂，减去一律 S500；
- **按 state 选：** 每个 (学生, state) 在 K16 ∪ {S300, S500} 中选最好的一臂（按 seed 交叉拟合），减去一律 S500。

**预注册的决策规则：**
- 若任一种空间的平均值 ≥ 2pp 且 CI 下界 > 0：下一轮在 D-local 下统一测量 controller 表示，并用分散式数据训练 critic。
- 否则：结论为"在分散式执行下，固定的 S500 已接近最优，生成器与 critic 没有可发挥的空间"。

# 五、交付要求中的结论映射

| 观察结果 | 结论 |
|---|---|
| 阶段 A 中 Z + S500 接近学生 + S500 | 在 Ring 中，修正本身就是一个可用的分散式 controller。论文不能写成"修正修好了学习型策略"，而应写成"修正本身能完成任务"，并重新审视集中式的结果 |
| 阶段 A 中 Z 远低于学生 | 学生与修正有协同，"修正修好学生"的说法成立 |
| S500 − RHR-fair 的 CI 下界 > 0 | 修正优于公平调参的经典右手规则 |
| RHR-fair 接近 S500 | 经典规则在调好之后同样有效，我们的优势只剩"无需调参" |
| 阶段 C 的空间 < 2pp | 整套 pipeline 在分散式下没有必要，固定 S500 即可 |

# 六、其他硬性规则

- **DB 写入：** 新目录之外唯一允许的写入，是共享 DB 的官方流程，共三处：
  - **design 阶段登记新身份：** 按 round28 `design.py` 的做法，用官方的 `connect()` 与 `transaction()` 执行 `INSERT OR IGNORE`，只追加本轮的新身份。要求：
    - 不得 UPDATE 或 DELETE 任何已有行，也不得直接插入 rollout 行；
    - 同一 uid 已存在且内容不同时，该阶段记为缺口并报告；
    - 记录插入的全部行及其 sha256；
    - 插入前后，各登记表中已有行的内容哈希必须相同。
  - **journal：** 用 Basin_C1 的 `shared_rollout_db.src.cache_writer.append_journal` 写到默认位置，并满足第零节第 4 条的断言。
  - **合并：** 用 `python -m shared_rollout_db.src.merger --experiment <experiment_uid>` 单进程合并。

  流程：design → preflight（在提交作业之前封存）→ 只跑缺失 → journal → 单进程官方 merger → postflight。
- **不得改动核心组件：** 不修改物理、事件判定、成功标准、安全层与修正的实现。教师与学生 checkpoint 都冻结，不重训。
- **参数写死：** 对照 controller 的定义、解读规则、右手规则的网格与选择规则、K16 列表、空间的决策规则，全部写入预注册，不得根据结果调整。
- **冻结时序：** 右手规则的选择，先于它的主评估标签。预注册先于任何新 rollout。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **精度：** 全部导入完成之后，显式恢复协议规定的精度（x64 = False），并记录。
- **视频：** 本轮不要求视频。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 20 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。其他轮次可能正在运行，提交作业前用 `squeue` 查看当前总占用。
- **执行顺序：** 先 A，再 B，最后 C。各阶段互不依赖；资源允许时可以并行。

# 七、交付

全部交付物放在本轮目录：
- 源码完整性审计：开始与结束两次记录及比对；
- PREREGISTRATION.md 及收据；
- 对照 controller 与右手规则变体的实现和单元测试；
- 阶段 A 的结果表与解读；
- 阶段 B 的完整调参网格、冻结的选择与主评估结果；
- 阶段 C 的 K16 列表核对、空间表与决策；
- DB 的 preflight 与 postflight 记录，以及登记行清单；
- results.json；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过（含第零节第 4 条的 DB 路径断言）；
- 零动作与直奔目标 controller 加修正后能做到多少，与学生、教师相比如何；
- 公平调参的右手规则选了哪一组，与 S500 相比如何；
- K16 上还有多少空间，是否值得继续做分散式的 critic。

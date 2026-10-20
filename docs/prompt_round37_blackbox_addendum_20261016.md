你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/blackbox_addendum_v1/`

# 零、先读

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）。** 源码保护、DB 规则、复用规则、标准方法与指标、标准交付，对本轮全部适用。
2. 本轮直接依赖的已有目录（一律只读，不得补写或修改其中任何文件）：
   - round32 `diagnostics/axis1_state_v1/`
   - round33 `diagnostics/axis2_controller_v1/`（`controllers.py`、`statistics_core.py`、`LABELS.npz`、`FROZEN_CHOICES.json`、各类型的 `*_RESULTS.json`）
   - round35 `diagnostics/axis4a_geometry_horizon_v1/`（`geometry.py`、`progress.py`、`prefix.py`、`CAPABILITY_GATE.json`、`PSTAR_*`）
   - round36 `diagnostics/axis4b_agent_count_v1/`（`ASSET_AUDIT*`、`CAPABILITY_GATE.json`、`FIVE_LAYER_EVIDENCE.json`、执行链代码）
   - round30d `diagnostics/teacher_student_macflow_v1d/`（`fair_rhr.py`、`RHR_GRID.json`、`RHR_SELECTED.json`、`PREREGISTRATION.md`）

## 0.1 名词对照（本文件中，新写的内容一律用全称；代号只用来对应已有文件）

| 代号 | 全称 |
|---|---|
| F10 / F13 / F15 | 全程固定用 η10 / η13 / η15 的修正 |
| S500 | 前 500 步用 η13、之后用 η10 的时间表 |
| 按比例切换时间表（S-rel） | 前 round(5/7·H) 步用 η13、之后用 η10 |
| 按进度切换时间表（S-prog） | 平均进度第一次达到门槛 p* 时从 η13 切到 η10 |
| safety baseline | 只加安全投影、不加修正（η = 0） |
| nominal only | 只执行 controller 的动作，不加安全层与修正 |
| FF 位置 | 修正加在 flow 积分内部（PoC `Runtime.action` 的 `chain = "field"`） |
| TT 位置 | nominal → 投影 → 加修正 → 再投影（`chain = "old"`） |
| A3 | round16 冻结的、看 controller 特征的选择模型 |
| A0 | round16 冻结的、不看 controller 的选择模型 |
| RHR | 持续右手规则：触发时叠加一个绕场景中心的顺时针切向速度，再投影一次 |
| DM | 分散式模型预测控制 controller：每个机器人只为自己做随机采样规划，把其他机器人预测为匀速运动 |
| 高斯行为克隆策略（类型 G） | round33 训练的单峰高斯 MLP 策略 |

## 0.2 本轮范围：只做明确的黑盒 controller

论文基调：**修正在安全层下运行的黑盒 controller 出现的死锁，不论死锁的成因。** 黑盒指：方法不依赖 controller 的控制律式子，但可以探查它的行为。

| 本轮纳入（明确黑盒） | 本轮不纳入 |
|---|---|
| Ring Flow（round10 的 14 个）；高斯行为克隆策略（round33 的 6 个）；分散式模型预测控制 controller（round33 的 6 个）；Gap 场景的分散式模型预测控制 controller（本轮移植） | 闭式公式的 controller：朝目标直走、势场、带切向项的势场、互惠避碰基线；以及暂不判定的速度障碍式反应 controller |

不纳入的 controller，已有结果保持原样，本轮不重算、不重跑，也不在本轮结论中出现。

## 0.3 资源与并行

- 用户全部任务合计不超过 18 个 CPU shard、2 个 GPU shard。开工前与每次提交前都要查 `squeue`，按其他作业的占用动态限制本轮的并发。
- 官方 merger 单进程运行，不与其他轮次的 merge 同时运行。
- 其他作业若在本轮期间运行，造成的完整性差异照常逐项记录，并在 REPORT 第一段注明来源。

# 一、本轮的六个部分

| 部分 | 问题 | 能否立刻执行 |
|---|---|---|
| A | round35 几何能力门缺阳性对照；并对门中的失败逐条分类 | 能 |
| B | round35 按进度切换时间表未执行（p* 的定义已由用户决定为候选 1） | 能 |
| C | round36 Gap 场景 N=2 Flow 边缘未过能力门：只读审计 | 能 |
| D | 持续右手规则作为对照方法，套在黑盒 controller 上 | 能 |
| E | 用与机器人数量无关的黑盒 controller（分散式模型预测控制）补做机器人数量轴 | 能 |
| F | round33 新类型 controller 的结论，在黑盒子集上离线重算 | 能，不跑 rollout |

执行顺序：F、C（只读，最快）→ A → B → D → E。各部分互不依赖，一个部分被挡住时记为缺口，继续其余部分。

**预注册：** 所有部分的设计、阈值、种子区间、门规则写在同一份 PREREGISTRATION.md 中，附 UTC 与 sha256 收据，早于任何新 rollout 与任何新结果计算。B 部分在预注册中写明用户选定的候选 1 及其数值。

# 二、A：几何能力门的阳性对照与失败分类

**背景：**
- 基准半径 r = 0.90 下，主评估的 32 个 state 上，S500 成功率约 90%；
- round35 的能力门在 r ∈ {0.80, 0.85, 0.95, 1.00} 上，用新生成的 16 个 pilot state（`sample_instance(seed, 'development', cfg)`，seed = 202610350000 + 半径偏移 {0, 1000, 2000, 3000} + 0..15），14 个 Ring Flow 用 F10 与 S500 各跑 seed 0；
- 各 controller 的 max(F10, S500) 成功率中位数分别为 0–3%、0–6%、0%、0–3%，全部失败；
- 没有在 r = 0.90 上用同一个 pilot 生成器跑过。
- 门里用的两个臂都已经加了修正。所以"不过门"既可能是 controller 本身不会走，也可能是死锁出现了但修正没修好。后者属于方法失败。

**A1 生成器阳性对照：** r = 0.90，生成器、split、config 构造方式与门规则都和 round35 完全相同，种子取 202610350000 + 4000 + 0..15（与 DB 和 round35 全部 pilot 去重，冲突就按预注册规则顺延）。14 个 controller × 16 个 state × {F10, S500} × seed 0 = 448 条。

**A2 主评估 state 换半径：** round35 主评估的 32 个 state 中按 state_uid 排序的前 16 个，在 r ∈ {0.80, 0.85, 0.95, 1.00} 下重新 reset。逐个检查 reset 有效性，无效的记为缺口，不替换。14 个 controller × {safety baseline, F10, S500} × seed 0，最多 14 × 16 × 4 × 3 = 2,688 条。r = 0.90 下的对应标签已存在，直接配对复用。

**执行：** FF 位置，H700，x64 = False，`key_for` 与 round10 一致。复用 round35 的 runtime 时按复制规则处理，并先过复现门（复现 round35 的 16 条几何 pilot 记录与 16 条 round10 记录）。

**半径传入方式的单元测试：** 必须与 round35 的 `geometry.py` 完全相同。测试 r = 0.90 时构造出的 config 与原环境默认 config 逐字段相同，并且同一 state 与 seed 下的轨迹与 round10 逐位一致。如果不一致，A 部分的结论全部作废，并在报告第一段写明：round35 的半径传入方式本身改变了 r = 0.90 的行为。

**A3 失败分类（描述性，不得称为 deadlock）：** 对 round35 能力门与本轮 A1、A2 中每一条未成功的 rollout，记录：
- 终止类型：碰撞、timeout、unknown；
- 终止时的平均进度 p(T)（定义与 round35 的 `progress.py` 相同）；
- 最后 100 步内的最大位移，以及所有机器人速度低于 0.05 m/s 的连续最长步数；
- 同一 state 在 nominal only 下的终止类型（若已有标签就复用；没有就在 A2 中补 nominal only 一臂，计入规模）；
- 据此把每条失败归入预注册的三类之一：**停滞候选**（timeout，且末段位移与速度都低于阈值）、**仍在移动的超时**、**碰撞或 unknown**。阈值写入预注册。

**判定：**

| 编号 | 内容 | 判定 |
|---|---|---|
| A-判定1 | A1 生成器阳性对照 | 各 controller max(F10, S500) 成功率中位数的已知下界 ≥ 20%：生成器有效，round35 门的结论成立；上界 < 20%：生成器无效，round35 门的结论作废；其余为 INCONCLUSIVE |
| A-判定2 | A2 同 state 跨半径（描述性） | 每个 r：中位数，以及相对同一批 state 在 r = 0.90 下的配对差与 CI；另报每个 controller 的逐 r 曲线 |
| A-判定3 | 条件性主评估 | 只有当 A-判定1 为"生成器无效"时：对 A2 中位数下界 ≥ 20% 的 r，按 round35 的几何主评估设计跑（32 个 state × seed 0–1 × 5 个臂 × 14 = 4,480 条/r），A3 的选择在评估标签之前冻结。门与评估用了同一批 state，因此一律标为 DEVELOPMENT |
| A-判定4 | 失败分类（描述性） | 每个 r：三类失败的比例；停滞候选占多数时，写明"修正未能解除的停滞"，作为方法的局限，不得写成 controller 能力局限 |

# 三、B：按进度切换时间表（用户已决定：候选 1）

**用户决定（2026-10-16 已填写）：** `p* 定义 = 候选 1：只用确实跑到第 500 步的 699 条，p* = 0.9681620884830543`

用户原话的意思：第 500 步之前就提前成功的 episode 不管；没有第 500 步读数的，不读就不读。因此 195 条提前成功的和 2 条在第 500 步之前数值失败的，都不进入 p* 的计算。

- 执行前，从 round35 的 `PSTAR_CALIBRATION_CANDIDATES.json` 独立重算：取 `coverage == "observed500"` 的 699 条 `p500_candidate`，其中位数必须与 0.9681620884830543 逐位相同；不同就停下 B 部分并报告。
- 把本决定的原文、重算结果、源文件 sha256 与 UTC 写入 `PSTAR_DECISION.receipt.json`。这份收据必须早于任何按进度切换时间表的 rollout。
- 在 REPORT 中写明：本定义排除了最快完成的 episode，所以 p* 比"把提前成功的进度保持在结束时的值"的定义（0.97281）略低，切换会略早。

**背景：** round35 预注册的 p* 是 round10 的 H700 下 S500 完整轨迹（14 个 controller × 32 个 state × seed 0–1，共 896 条）在第 500 步的平均进度的中位数。其中 699 条跑到了第 500 步；195 条在第 500 步之前已经成功结束；2 条在第 500 步之前数值失败。后 197 条没有第 500 步的进度，字面定义算不出来。

| 候选 | 定义 | p* |
|---|---|---|
| 1 | 只用跑到第 500 步的 699 条 | 0.96816 |
| 2 | 已成功结束的 195 条，进度保持在结束时的值；2 条数值失败的按 [0, 1] 给界 | 0.97281（给界区间 [0.97274, 0.97284]） |
| 3 | 用户另行指定 | — |

**选定后的执行：** 按 round35 的 `progress.py` 执行；到第 round(6/7·H) 步仍未达到就在该步切换。复用 round35 已通过的单元测试，并对选定的 p* 再做一次在线与离线切换时刻一致性的测试。H500、H700、H1000 × 14 × 32 × seed 0–1 = 2,688 条；切换前的前缀与 F13 逐位相同，按 round35 `EXACT_PREFIX_REUSE_AUDIT` 的规则复用，只跑切换后的部分。终点沿用 round35 预注册的第 3、4 个时间预算终点（按进度切换是否更好；方法在新预算下是否成立），判定标准不变。证据等级为"CONFIRMATORY（定义修订）"。

# 四、C：round36 Gap 场景 N=2 Flow 的只读审计

不跑 rollout，只读已有文件，回答：
1. 16 个 pilot state 上三臂的成功率（safety baseline 6.25%、F10 12.50%、前 round(5/7·H) 步 η13 之后 η10 的时间表 18.75%）分别来自哪些 state；失败的终止类型分布；按 A3 的失败分类规则归类。
2. 该 checkpoint 的历史能力证据，是在相向交换任务上，还是非相向任务上？与本轮 pilot 任务的差别是什么？
3. `CONSISTENCY_GATE_BEFORE_ACTION_ENCODING_FIX` 与修正后的 `CONSISTENCY_GATE`：确认修正的是复现检查器（float32 与 float64 的比较方式），而不是执行代码；确认实际执行的动作路径与历史执行逐位一致。
4. 结论只能二选一，不得重判门：
   - "门的执行正确，N=2 Flow 在相向任务上能力不足"；
   - "发现执行或判定错误"：列出证据，建议一个新的预注册轮次，本轮不重跑。

# 五、D：持续右手规则作为黑盒上的对照方法

**为什么做：** 持续右手规则同样不需要知道 controller 的式子，是与我们的修正同类的竞争方法。round30d 只在学生 controller 上比较过（时间表 94.20%，持续右手规则 11.27%）。本部分把比较扩展到各泛化轴的黑盒 controller 上。

**实现：**
- round30d 的 `fair_rhr.py` 是写给分散式局部安全层的，不能直接用。在本轮目录写一个新函数，逻辑与 `fair_rhr.py` 的触发器（`Trigger`：always、slow、progress 三种模式，规则逐字照搬）和切向方向（世界坐标系顺时针 (p_y, −p_x)/|p|）完全相同。执行顺序：
  1. 取该 controller 在原执行位置、η = 0 下的动作，也就是 safety baseline 的动作；
  2. 对触发的机器人叠加切向速度；
  3. 用同一个安全投影再投影一次。
  
  不加任何 η 修正，碰撞与终止仍由原 simulator 判定。
- **单元测试：**
  - 触发器从不激活时，轨迹与 safety baseline 逐位一致；
  - 触发器的激活序列与 `fair_rhr.py` 在同一条位置、速度历史上的结果逐位一致。

**公平调参（与 round30d 规则相同）：**
- 参数网格为 round30d 的 `RHR_GRID.json`，共 20 组。
- 每组 controller（14 个 Ring Flow 为一组，6 个高斯行为克隆策略为一组，6 个分散式模型预测控制 controller 为一组）分别在 16 个新 tuning state 上调参：seed 0，种子区间写入预注册，与 DB 和所有评估 state 去重。
- 选择规则：成功率下界最大；平局依次取速度小者、持续步数小者，最后按 always、slow、progress 的顺序。
- 选择在任何评估标签之前冻结并留收据。
- 规模：20 × 16 × (14 + 6 + 6) = 8,320 条。

**评估（全部与已有标签配对；已有的臂直接复用，只新跑持续右手规则一臂）：**

| 子项 | controller | state | 新跑规模 |
|---|---|---|---|
| D1 同分布新 state | 14 个 Ring Flow | round32 的 1a 的 64 个 state，H700，seed 0–1 | 1,792 |
| D2 分布偏移 state | 14 个 Ring Flow | round32 的 1b 的 32 个 state | 896 |
| D3 时间预算 | 14 个 Ring Flow | round35 的 32 个 state，H500 与 H1000 | 1,792 |
| D4 几何 | 14 个 Ring Flow | A2 的 16 个 state × 4 个半径，seed 0 | 896 |
| D5 新类型黑盒 | 6 个高斯行为克隆策略 + 6 个分散式模型预测控制 controller（TT 位置，与 round33 相同） | round33 2d 的 32 个评估 state，seed 0–1 | 768 |

**终点：**

| 编号 | 内容 | 判定 |
|---|---|---|
| D-判定1（主） | D1：S500 − 持续右手规则 | 配对 CI 下界 > 0：时间表更好；上界 < 0：持续右手规则更好；其余：未检出差别 |
| D-判定2 | D2、D3、D4、D5 各自同上 | 同上；D3 用按比例切换时间表，D4 只在 A-判定中有效的半径上判定，其余描述性 |
| D-描述 | 各子项中持续右手规则 − safety baseline；以及"在安全层下失败、持续右手规则救回、时间表没救回"和反过来的逐条交叉表 | 描述性 |

D1–D5 的标签全新，选择先于评估冻结，标为 CONFIRMATORY。

# 六、E：用分散式模型预测控制 controller 补做机器人数量轴

**为什么做：** round36 中，与机器人数量无关的两个 controller（通用势场、互惠避碰基线）都是闭式规则，不属于黑盒。黑盒中只有各 N 专用的 Flow，每个 N 用不同的 checkpoint，无法检验"同一个 controller 跨 N"。分散式模型预测控制 controller 天然与 N 无关，是这一检验唯一明确的黑盒候选。

**移植：**
- 把 round33 `controllers.py` 中的 `distributed_shooting` 复制到本轮目录，只做两处推广：
  1. 机器人数量从写死的 4 改为从环境读取；
  2. 碰撞代价中的障碍改为从 Gap 场景环境读取真实墙段与门的几何。
  
  代价项（到目标距离、碰撞惩罚、平滑项）、采样方式与随机流的派生方式逐字不变。
- 单元测试：在 Ring、N=4 下，推广版与原版逐位一致。
- 执行链与 round36 中 Flow 的执行链完全相同（nominal → 速度界与原 goal-stop → 原 `CertifiedHardSafetyFilter` → 修正 → 同一投影 → native），只替换 nominal 的来源。复现门：复现 round36 的 16 条记录。
- 环境、地图、门宽、机器人半径、动力学、H_N 与 round36 完全相同（H_N = 2000 / 8000 / 16000 对应 N = 2 / 10 / 20）。AGENTS.md 中关于可扩展母版的全部硬性要求照常适用。

**实例：** 从 round33 的 6 组参数中，按预注册写定 2 组：计算最省的 (规划步长 10, 样本数 64, 避碰权重 100)，以及 round33 中在 Ring 上成功率最高的一组（在预注册之前从 `DM_RESULTS.json` 读取，写入预注册）。

**运行时间预算：**
- 每个 N，先在 2 个 pilot state 上测量每个物理步的耗时，推算该 N 主评估的总 CPU 时间；
- 若超过预注册的上限（写入预注册，建议为每个 N 不超过 18 CPU × 24 小时），该 N 记为"计算预算缺口"，不降低样本数、不改参数；
- 测时用的 rollout 不计入任何结果。

**门、评估与终点：**
- 门、动作菜单（9 个臂）、state 生成器、种子规则、专家参照、迁移检验的规则，都与 round36 的预注册完全相同，只把 controller 换成上面 2 个实例。N ∈ {2, 10, 20}。
- 终点沿用 round36 的四个：修正在更大 N 上是否仍有用；选择能否跨 N 迁移（判定标准为平均遗憾 ≤ 5pp）；最佳动作族是否随 N 改变；五层证据。判定标准不变。
- 规模：能力门最多 3 × 2 × 16 × 3 = 288 条；主评估最多 3 × 2 × 32 × 2 × 9 = 3,456 条；另加测时。
- 证据等级：CONFIRMATORY。

# 七、F：round33 新类型结论在黑盒子集上的离线重算

**不跑 rollout。** 只用 round33 的 `LABELS.npz`、`FROZEN_CHOICES.json` 与 `statistics_core.py`（只读 import，或按复制规则复制）。

- 子集：6 个高斯行为克隆策略 + 6 个分散式模型预测控制 controller，共 12 个实例。
- 重算 round33 的三个主要终点，判定标准不变：
  1. 方法在新类型 controller 上是否可用（A3 相对完整菜单 oracle 的平均遗憾 ≤ 3pp；A3 − safety baseline 的 CI 下界 > 0；伤害数为 0）；
  2. critic 学到的关系能否跨类型（先检查前提：固定偏好是否有双向分歧）；
  3. 训练中加入更多类型是否有帮助（混合类型 critic − A3）。
- 另报一个描述性对照：两个黑盒类型各自的结果，与 round33 原来四类合并的结果并列。
- **证据等级：** 子集是在看过 round33 结果之后才划定的，所以一律标为"DEVELOPMENT（事后子集）"，不得写成 CONFIRMATORY。在 REPORT 中写明划定的依据：用户于 2026-10-16 给出的黑盒定义。

# 八、交付

按共同协议第五节交付，放在本轮目录。另外：
- A：A1、A2 的能力表；失败分类表；半径传入方式的单元测试；
- B：`PSTAR_DECISION.receipt.json`、按进度切换时间表的实现、单元测试与结果；
- C：`N2_GATE_AUDIT.md`；
- D：持续右手规则的新函数、单元测试、调参表与冻结收据、D1–D5 结果与交叉表；
- E：推广版分散式模型预测控制 controller 与单元测试、测时报告、能力门、五层证据表、迁移检验表；
- F：`BLACKBOX_SUBSET_RESULTS.json` 与对照表。

AXIS_TABLE 用共同协议的统一格式。每一行是一个子项，列在标准方法之外，增加"持续右手规则"一列。未做或被挡住的子项保留一行，标"缺口"并写明原因。表头注明本轮只含明确黑盒的 controller。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过，差异各来自哪里；
- A：round35 的几何能力门是否成立；失败中停滞候选的比例；
- B：是否执行；若已执行，给出结论；
- C 的结论；
- D：时间表与持续右手规则谁更好（D1 为主，D2–D5 逐项）；
- E：机器人数量轴在五层证据的哪一层、哪个 N 上第一次失败；修正与迁移选择的结论；
- F：黑盒子集上三个终点的结果（DEVELOPMENT）。

REPORT 与 HANDOFF 中不使用简写：第一次出现的代号必须给出全称，此后正文用全称。HANDOFF.md 要做到不依赖本对话也能读懂。

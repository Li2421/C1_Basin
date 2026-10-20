你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

# 零、源码保护（最高优先级，先于本文件其他所有要求）

用户明确要求：**不得改动现有源码。** 本轮的全部新代码与输出，只能写在新目录 `diagnostics/teacher_student_macflow_v1/` 中。

**1. 只读，不得修改、删除、移动、重命名或重新格式化：**
- Basin_C1 中所有已有文件，包括 `new_benchmark_common/`、`ring_exchange/`、`shared_control/`、`shared_rollout_db/`（代码只读；DB 的写入方式见第七节）、`bottleneck_family/`、`toy_giveway/`、`double_bottleneck/`、`four_way_intersection/`、`docs/`、`AGENTS.md`，以及 `diagnostics/` 下其他所有子目录；
- PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`（round28 runtime 所在，不在 git 中）；
- MAC-Flow 官方仓库 `/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/`。它本身带有用户尚未提交的改动，必须保持原样；
- Python 环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/`：不安装、不升级任何包。如果缺少依赖，在新目录内另建环境，或停下报告。

**2. 复用已有代码的方式：**
- 只读 import；
- 或复制到新目录后再修改，并记录源文件路径与 sha256；
- 需要不同行为时，在新目录中写子类或新函数。不得在运行时替换已有模块的函数或全局状态来改变它的行为。唯一的例外是按协议显式恢复 JAX 精度。

**3. git：** 两个仓库都只允许只读命令（status、diff、log、rev-parse、ls-files）。严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。两个仓库都有用户尚未提交的改动。

**4. 完整性审计（交付必需）：** 在开始时与结束时各记录一次：
- Basin_C1 的 `git status --porcelain`，以及 `git diff` 输出的 sha256；
- MACFlow_Official 的 HEAD、`git status --porcelain`，以及 `git diff` 输出的 sha256；
- PoC 目录不在 git 中：记录它的 `poc/` 下全部 .py 文件的 sha256 清单；
- 本轮 import 或复制的每个已有源文件的 sha256。

除新目录外，结束时的记录必须与开始时完全一致。不一致就在 REPORT.md 第一段写明，并列出改动。

**5. 被挡住时停下报告。** 如果已有代码缺少需要的接口，或者只有改源码才能继续，就停下报告，不要绕过。

**续跑说明：**
- 第一次运行按本节第 5 条在接口处停下，见本目录的 REPORT.md。原因是本文件旧版对教师采样的要求自相矛盾，现已改为第二节所列的已有接口。
- 续跑前，把第一次运行的全部产物原样移入本目录下的 `attempt1_blocked/` 子目录，不改内容。
- 然后重新做开始时的完整性审计，再按本文件执行。
- 第二次运行停在噪声一致性门的第一项：`nominal(traced=True)` 的内置断言。原因是该断言按 float64 设定容差，与协议的 float32 精度不符；门的定义已在第二节修正。
- 第三次运行前：
  - 把第二次运行的产物原样移入 `attempt2_gate_tolerance/`，`attempt1_blocked/` 保持不动；
  - 原预注册保留不动，另写一份补充预注册，记录门的修改与原因，附 UTC 与 sha256 收据。到目前为止还没有读取任何学生结果，这一修改与结果无关；
  - 重新做开始时的完整性审计，再从噪声一致性门开始执行。
- 第三次运行中，噪声一致性门已 1,008/1,008 通过；之后停在 DB 身份登记，因为旧版第七节没有授权登记新身份。现已在第七节补上，并补充了第三节的 round28 复现门。
- 第四次运行前：
  - 把第三次的 REPORT.md、HANDOFF.md、results.json 复制一份到 `attempt3_db_registration/` 保存；
  - 第三次已核验的产物（噪声一致性门结果、已完成的训练数据重建），在代码与输入 hash 不变时直接复用，不必重跑；中途取消、未完成的部分重做；
  - 另写一份补充预注册，记录本次的两处修改（DB 登记授权、复制版 pipeline 与 round28 复现门）及原因，附 UTC 与 sha256 收据。到目前为止仍未读取任何学生结果；
  - 重新做开始时的完整性审计，然后从 design 与 DB 身份登记继续。

# 一、要回答的问题：专家无法部署时，结论是否仍然成立？

开始前先读：
- `AGENTS.md`、`shared_rollout_db/README.md`；
- 总结报告 `docs/controller_generalization_ring_summary_20261014.md`，重点是第 14–17 节；
- round28 的报告与代码：`diagnostics/decentralized_pipeline_v1/`；
- round27 的探测与冻结选择代码：`diagnostics/other_controller_classes_v1/probes.py`、`freeze_choices.py`；
- MAC-Flow 官方实现：`/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/agents/macflow.py`。论文为 Lee、Lee、Zhang，"Multi-agent Coordination via Flow Matching"，ICLR 2026，arXiv 2511.05005。

**背景：**
- 到目前为止，所有 controller 都是集中式的。联合 MACFlow 看到全部机器人的状态，并隐含地知道全部机器人的目标；它是模仿集中式专家训练出来的。
- 集中式专家在 round28 的 32 个 state 上 32/32 成功，每局规划只要 0.075 s。审稿人可以直接说"用专家就行"。
- 学习型 controller 的存在理由，是专家在部署时用不了。本轮的设定是：每个机器人只知道自己的目标，靠自身感知看到其他机器人的位置与速度，不与其他机器人通信。专家需要全部机器人的目标，在这个设定下无法部署。

**做法：效仿 MAC-Flow。**
- MAC-Flow 分两个阶段：第一阶段学联合 flow 策略；第二阶段把它蒸馏成每个机器人各自的一步策略，用于分散式执行。
- 仓库中的 MACFlow 来自 MAC-Flow 官方代码，但只实现了第一阶段，而且是展平的联合版本（4×23 维观测、8 维动作、联合噪声）。
- 本轮补上第二阶段：把已有的联合 MACFlow 作为教师，蒸馏出分散式的学生。

**本轮问题：**
1. 可部署的学生加上安全层后，是否仍然"安全但卡住"？
2. 时间表相对固定 η 的优势，在学生上是否仍然存在？
3. 只在联合 Flow 上训练的冻结 critic A3，能否零样本地为学生选出接近最优的修正？分两层回答：
   - 方法层面的性能；
   - critic 是否真的利用了 controller 信息。完整菜单里一律 S500 往往已接近最优（round27、round28），A3 与不看 controller 的 A0 都选 S500 时无法区分。所以"学到的 controller 与 Q 的关系"要在答案有分歧的固定菜单里检验。
4. 学生是否继承教师的 η 偏好？
5. 分散化的代价有多大，学生在推理速度上有多大优势？

# 二、学生的构造（MAC-Flow 第二阶段）

**教师：** round28 所用的 14 个原 Ring Flow（best.pkl），以 round28 的 `main_protocol.json` 为准。教师全部冻结，先核对 checkpoint 的 hash。

**学生网络：** 采用官方 `actor_onestep_flow` 的形式（`ActorVectorField`）。实现写在新目录中，可以 import 官方模块，或复制后再修改。
- 输入：机器人编号的 one-hot、该机器人自己的观测 o_i、它自己的噪声 z_i ∈ R²。
- 输出：该机器人的速度命令 u_i ∈ R²，一步得到，不做 ODE 积分。
- 4 个机器人共享参数。
- 网络宽度、学习率等超参数采用官方 `get_config()` 的默认值（例如 actor_hidden_dims = (512, 512, 512, 512)，lr = 3e-4）。

**观测 o_i：** `ring_exchange/local_frame.py` 中 `local_observation` 的第 i 行，只读调用。
- 包含：自身状态、自身目标、其他机器人的相对位置与速度（自身感知）、障碍几何。
- 不包含其他机器人的目标。
- 所有学生的输入格式统一，与教师各自的观测版本无关。
- 只看 1.5 m 以内机器人的局部感知版本，不在本轮做，视本轮结果再定。

**蒸馏目标：** 采用官方的蒸馏损失。
- 在数据集 state x 上，用随机 key k 调用教师，得到教师动作 u，以及它所用的联合噪声 ξ = (z_1, …, z_4)。
- 损失为对 i 求平均的 ‖μ(o_i, id_i, z_i) − u_i‖²。学生只看到自己的噪声分量 z_i。

**教师动作与噪声 ξ 的取法：** 用已有接口，不需要改源码，也不需要复制采样器。
- 只读 import round28 runtime 的 `Runtime`（`/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py`），用它的公开属性：
  - `runtime.observation(env)` 给出教师观测 `obs`；
  - **教师动作：** `runtime.native_fn(obs, k)` 的输出 `raw`。`nominal` 内部调用的就是它，路径为 `native_fn → sample_bounded_actions → JointMACFlowAgent.sample_actions`，与 round28 执行时完全相同。`raw` 为局部坐标、已限速。学生输出与 `raw` 处在同一空间，执行时走与教师相同的 `to_world` 和再限速。
  - **噪声 ξ：** `runtime.trace_fn(obs, k)` 的第 0 步，reshape 成 (steps + 1, 4, 2) 后取 [0]，为归一化坐标，每个机器人 2 维。`nominal(env, k, traced=True)` 内部用的就是这个函数，它的 z 与原生采样内部的 `jax.random.normal(k, (1, joint_act_dim))` 是同一次抽样；`nominal` 已断言 trace 路径的终点与原生动作一致。FF 管线的 `integrate` 也是这样从 key 取噪声的。
- 数据集 state 不一定是 t0，而 `make_env` 只接受 t0。教师采样只用到位置、速度与目标，所以在新目录中直接构造 `RingExchangeEnv` 并 reset 即可，不要用 `make_env`。

**噪声一致性门（训练前必须通过）：** 在至少 1,000 个（数据集 state, k）上确认三点：
- **噪声逐位一致：** `trace_fn` 的第 0 步，与在新目录中独立调用 `jax.random.normal(k, (1, joint_act_dim))` 的结果逐位一致；
- **教师动作逐位一致：** `native_fn(obs, k)` 的输出经 `to_world` 与再限速后，与 `nominal(env, k, traced=False)` 返回的动作逐位一致；
- **trace 终点与原生动作吻合：** 在新目录中，按 `nominal` 的同样方式解码 trace 终点：`clip(zpath[-1]·scale + mean, −1, 1)`，限速，再 `to_world`。它与原生动作的最大绝对差，在每个样本上都 ≤ 1e-5 m/s。报告差值的分布（最大值与 99.9 分位）。

任何一项不通过就停下报告。

**为什么第三点不用 `nominal(traced=True)` 的内置断言：**
- 该断言的容差是 rtol = 1e-10、atol = 1e-11，是按 float64 写的；PoC 在 import 时默认开启 x64。
- 协议精度是 x64 = False，也就是 round28 部署时的精度。这时 JAX 按 float32 计算，原生路径与 trace 加 float64 解码这两条计算路径，会有约 1e-8 的舍入差。第二次运行的首个样本就是这样：最大差 9.89e-9，而另外两项逐位一致。
- 如果噪声取错，差值会达到 1e-2 以上，比 1e-5 高出三个数量级以上，所以第三点仍能发现噪声对不上的问题。
- 不要为了通过内置断言而改用 x64 = True 运行教师，部署时的教师是 float32；也不要修改原断言。

**与官方实现的差异（写入预注册）：**
- 教师冻结，事后蒸馏。官方实现是 BC flow 与一步策略一起训练。冻结教师是为了让每个学生只继承一个固定的教师。
- 不加 critic 与 Q 项，相当于官方 Q 损失权重为 0。原因：我们的第一阶段是纯 BC，数据没有 reward；本轮研究的是模仿得到的学生。
- 我们的教师是展平的联合 flow；官方第一阶段是按机器人编号逐个处理的。

**训练数据：**
- 用每个教师自己训练集的 state，标签由教师生成，不用专家标签。dev 划分只用于按 dev 蒸馏损失选 checkpoint，不用任何闭环结果选择。
- 数据集的每个 rollout 文件存有 `states`（位置）、`observations`、`actions`、`initial_state_json` 等。在新目录中用它们重建 env 的位置、速度与目标。教师观测一律用 `runtime.observation(env)` 计算，与部署一致；同时与文件中存的 `observations` 比对，报告差异。有差异时，以 runtime 计算的为准。
- 不做 DAgger，与 MAC-Flow 的离线设定一致。
- 步数、batch、验证批数与 seed 全部写入 config（round24 的教训）。每个学生一个训练 seed，列在预注册中。
- 动作的归一化与裁剪与教师一致，保证学生与教师的输出在同一物理单位下比较。

**离线检查：**
- 每个学生的 dev 蒸馏误差，以及与教师动作的一致程度。
- 单元测试：
  - 改变其他机器人的目标，学生输出不变；
  - 学生只用自己的噪声分量；
  - 给定观测与噪声时，输出是确定的。

# 三、执行管线与臂

**安全层与修正：** 使用 round28 冻结的 D-local 实现（`diagnostics/decentralized_pipeline_v1/pipeline.py`，hash 见 `IMPLEMENTATION_FROZEN.receipt.json`），只替换 nominal 动作的来源。
- **不要直接 import round28 的 `pipeline.py`。** 它通过 `from core import *` 加载 round28 的 `core.py`，而后者在 import 时会把 TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME 指向 round28 的目录，并把 `write()` 限定在 round28 的目录内。直接 import 会让本轮的临时文件和缓存写进 round28 的目录。
- **做法：** 把以下代码复制到新目录：
  - `pipeline.py`；
  - `run.py` 中的单局执行函数 `episode`，round28 的 rollout 循环在这里；
  - 它们用到的 `core.py` 部分。

  只允许修改路径常量与上述环境变量，使它们指向新目录；其余逻辑逐字保持不变。提交与原文件的 diff，以及原文件的 sha256。
- 每步 key 的推导函数 `key_for`（`diagnostics/orthoflow3_tt_ff_matched_learnability_v1/execution.py`）在 import 时没有副作用，可以只读 import。

**学生与新增臂的接入方式（新代码写在单独的文件中，复制版的逻辑不改）：**
- **学生：** 包装成一个 runtime 对象，实现与教师 runtime 相同签名和返回格式的 `nominal(env, key)`。其中第 i 行由机器人 i 的学生计算：用它自己的观测，噪声取自 `jax.random.fold_in(key, i)`。这样复制版的 `decentralized_action` 一行不改，只是传入的 runtime 换成学生。
- **safety baseline：** 复制版 `decentralized_action` 的 `local` 模式，η = (0, 0, 0)。
- **nominal only：** round28 没有这一模式。新写一个函数，直接执行 nominal 动作，不投影、不修正；碰撞仍由原 simulator 判定。
- **B-RHR：** round28 的右手规则只在 D-shared（感知半径 ∞）下实现。新写一个函数：沿用复制版的 `LocalSafety` 与右手规则的冻结常数（低于 0.01 m/s 连续 40 步、未到达目标时，叠加 0.15 m/s 的顺时针切向速度），但感知半径取 1.5 m。
- **新增函数的单元测试：**
  - B-RHR 在右手规则未触发时，执行动作与 safety baseline 逐位一致；
  - 构造一个低速场景，确认右手规则的触发条件与扰动大小、方向符合上述定义；
  - nominal only 的执行动作与学生的 nominal 动作逐位一致。
- **round28 复现门（在任何学生 rollout 之前必须通过）：** 用复制版复现 round28 的 16 条 D-local 记录：4 个教师 × {F10, S500} × 2 个 state，seed 0，教师中至少含一个 local 观测、一个 global 观测。首动作字节、终止类型、步数与完整位置轨迹必须逐位一致（round28 的轨迹存于 `main/traces/`）。不一致就停下报告。
- 每个机器人只解自己的 QP，只计 1.5 m 以内的邻居。成对约束各承担一半，邻居按零速度假设；障碍与外环由自己完全承担。
- 修正采用 TT 位置。B_goal_i 与 B_perp_i 只用自身信息，B_rel_i 只用感知到的邻居。时间表按各自的时钟切换。

**通信假设（写入预注册）：**
- 学生不通信，只知道自己的目标。
- 机器人 i 的噪声取自 `jax.random.fold_in(key_for(state, seed, step), i)`：每个机器人用事先约定的根种子、自己的编号和自己的时钟推出，不需要通信；各机器人的噪声互不相同，不是同一份联合噪声。
- 时间表只用自己的时钟。

**学生的臂（每个学生都跑）：**

| 代号 | 内容 |
|---|---|
| nominal only | 只执行学生动作，不加安全层，不加修正 |
| safety baseline | D-local 安全投影，不加修正 |
| F10、F13 | 全程固定 η |
| S300、S500 | 时间表 η13 → η10 |
| B-RHR | D-local 安全层加右手规则扰动；参数用 round28 冻结的值，不使用 OrthoFlow3 修正 |

**教师补充臂：** 14 个教师在同一批 state 上补跑 nominal only 与 D-local safety baseline，只跑 DB 中缺失的。教师的 F10、F13、S300、S500（集中式、D-shared、D-local 三条管线）以及专家结果，直接复用 round28 的记录。

**专家：** 本轮不运行。只引用 round28 的特权参照（32/32，每局 0.075 s），并写明它需要全部目标，在本设定下无法部署。

# 四、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**state：** 直接使用 round28 的 32 个评估 state（`diagnostics/decentralized_pipeline_v1/EVAL_STATES.json`），seed 0–1，以便与教师逐条配对。不要另外生成评估 state。

**能力门（只用新的 pilot state）：**
- 16 个新 pilot state。种子区间事先算好，写入预注册，并与 DB 以及 round28 的保留区间去重（round27 的教训）。
- 每个学生跑 safety baseline 与 S500 两个臂，seed 0。
- 两个臂的成功率都低于 15%，或都高于 98%，就排除该学生并报告原因。不得替换、重训或调参。
- 能力门后剩下的学生少于 8 个，本轮记为 INCONCLUSIVE。

**无标签探测与冻结选择（在任何评估标签之前完成并冻结，留收据）：**
- 对每个通过能力门的学生运行 R5 探测：16 个 probe state × {η10, η13} × 200 步，关闭终止，管线为本轮的 D-local。预处理与 PCA 用 round16 冻结的版本。
- round27 的 `probes.py` 与 `freeze_choices.py` 含全局变量替换和向其他目录的写入，不能直接运行。在新目录中复制后改写，其余依赖只读 import。
- 学生的执行：参照 round27 `controllers.py` 中 `class Nominal(Runtime)` 的写法，在新目录中写子类，只替换 nominal。每个机器人用自己的观测与噪声，计算自己的动作。
- 用 round16 冻结的 A3 与 A0 做选择。完整菜单取 A3 冻结菜单与 {F10, F13, S300, S500} 的交集，交集写入预注册；固定菜单为 {F10, F13}。
- 同时记录 G_E 的选择、一律 S500、一律 F10。

**规模：**
- 学生评估：14 × 32 × 2 × 7 = 6,272 条；
- 教师补充：最多 14 × 32 × 2 × 2 = 1,792 条；
- 另有能力门 448 条与探测。

**流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。学生 checkpoint、新管线与各臂都登记为新的执行身份。preflight 必须在提交作业之前封存（round28 的教训）。

**精度：** 全部导入完成之后，显式恢复协议规定的精度（round25 的教训），并记录 x64 状态。

**推理耗时（描述性）：** 在 1 个 CPU 核、JIT 预热后测量每步耗时：学生每个机器人一步推理；教师联合 10 步采样；D-local QP。专家规划耗时引用 round28。

# 五、预注册的终点与判定

所有 CI 先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次，使用配对比较。unknown 按下界计，另报上界。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| TS1 | 学生设定下，问题是否仍然存在 | S500 − safety baseline 的平均值 CI 下界 > 0。另报从 nominal only 到 safety baseline 的逐条转移表：nominal 下碰撞的 episode，加安全层后变成成功、timeout 与 unknown 的比例 |
| TS2 | 时间表优势是否仍在 | 最佳时间表 − 最佳固定 η 的平均值 CI 下界 > 0，且 ≥ 70% 的学生满足 S(最佳) ≥ max(F10, F13) − 1pp |
| TS3a | 方法层面的零样本性能（完整菜单） | A3 相对四臂 oracle 的平均遗憾 ≤ 3pp，且 A3 − safety baseline 的 CI 下界 > 0。另报 A3 与一律 S500、A0、G_E、F10 的配对差（描述性） |
| TS3b | critic 是否真的利用了 controller 信息（固定菜单 {F10, F13}） | 前提：学生的固定偏好确实有分歧，即至少 3 个学生 F10 比 F13 高 ≥ 10pp，且至少 3 个学生 F13 比 F10 高 ≥ 10pp；不满足就记为 INCONCLUSIVE（没有可检验的差异）。判定：A3 的平均遗憾低于 A0，配对 CI 上界 < 0。另报 A3 的选择与学生实际最佳固定臂的一致率 |
| TS4（描述性） | 学生是否继承教师的偏好 | 教师与学生最佳臂的一致率（教师取 round28 的 D-local 标签）；四个臂成功率的 Spearman 相关；教师的 η 类型能否预测学生的 |
| TS5（描述性） | 分散化代价与速度 | 每个臂上学生与教师（D-local、集中式 TT）的成功率差；碰撞、数值失败与停滞诊断；推理耗时表 |

**证据等级：**
- 学生的标签全新，且已预注册，TS1–TS3 为 CONFIRMATORY。
- 教师的标签此前已打开过，只作为 DEVELOPMENT 参照。

# 六、结论映射

| 观察结果 | 结论 |
|---|---|
| TS1、TS2、TS3a 都通过 | 在专家无法部署的分散式学生上，"安全层导致停滞"仍然存在，时间表优势与零样本选择都成立。可以作为论文的主实验 |
| 另外 TS3b 也通过 | critic 学到的 controller 与 Q 的关系，迁移到了一个由不同过程产生、信息也不同的新 controller 家族 |
| TS3a 通过，TS3b 不通过或 INCONCLUSIVE | 方法层面零样本可用，但这主要来自时间表本身的稳健，学到的关系是否迁移未得到确认 |
| TS1 通过，TS2 不通过 | 学生的失败模式与教师不同（例如各自选了相反的环流方向而对向冲突），时间表解决不了。报告失败类型与发生时间 |
| 多数学生因"两个臂都高于 98%"被能力门排除 | 学生在只加安全层时已近满分，这个问题在学生设定中不存在，如实报告 |
| 多数学生因"两个臂都低于 15%"被排除 | 蒸馏学生能力不足，记为 INCONCLUSIVE |
| 加了安全层的臂出现任何碰撞 | 先报告安全问题，所有成功率结论都附上这一前提 |

# 七、其他硬性规则

- **源码保护：** 见第零节，优先于一切。
- **DB 写入：** 新目录之外唯一允许的写入，是共享 DB 的官方流程，共三处：
  - **design 阶段登记新身份。** 官方 merger 只接受已登记的身份。按 round28 `design.py` 的做法，用官方的 `shared_rollout_db.src.rollout_db.connect()` 与 `transaction()`，对 experiment、state、controller_config、eta 表（需要时还有 scenario 表）执行 `INSERT OR IGNORE`，只追加本轮的新身份。要求：
    - 不得 UPDATE 或 DELETE 任何已有行，也不得直接插入 rollout 行；rollout 只能经 journal 与官方 merger 入库；
    - 插入前逐条检查：同一 uid 若已存在且内容不同，停下报告；
    - 记录本轮插入的全部行（JSON）及其 sha256；
    - 插入前后，各登记表中已有行的内容哈希必须相同，rollout 表中已有 experiment 的行数不变。官方 merger 自身对 experiment 行的更新（例如结束时间）除外。
  - **journal：** worker 用 `shared_rollout_db.src.cache_writer.append_journal` 把 journal 写到默认位置 `shared_rollout_db/journals/<experiment_uid>/`。
  - **合并：** 用 `python -m shared_rollout_db.src.merger --experiment <experiment_uid>` 单进程合并。

  `rollout.sqlite` 与 journal 目录都已被 git 忽略，不影响完整性审计。不要通过修改 `cache_writer.ROOT` 等模块全局变量来重定向 journal。
- **不得改动核心组件：** 不修改物理、事件判定、成功标准，以及 round28 的安全层与修正实现。教师冻结。
- **参数写死：** 学生的训练设置、观测定义、臂、能力门与阈值全部写入预注册。不得根据闭环结果调整、重训或替换学生。
- **冻结时序：** 学生训练与 checkpoint 选择、探测、冻结选择，都必须在评估标签之前完成并留收据。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock；停滞诊断只作描述。
- **视频：** 本轮不要求视频。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 八、交付

全部交付物放在 `diagnostics/teacher_student_macflow_v1/`：
- 源码完整性审计：开始与结束两次记录，以及二者的比对结果；
- PREREGISTRATION.md 及收据；
- 历史审计：仓库中是否已有分散式或一步式的学生可以复用，例如 `toy_giveway/flowbc.py`、`double_bottleneck/flowbc_4a_agent.py`；
- 学生训练代码、config、checkpoint、训练曲线、蒸馏报告与单元测试；
- 能力门报告；
- 探测特征、冻结选择及收据；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 论文用的总表：学生与教师，列出 nominal only、safety baseline、F10、F13、S300、S500、B-RHR、A3 零样本、一律 S500、oracle 的成功率与 CI，另附碰撞率、超时率；
- 转移表、继承表与推理耗时表；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：
- 源码完整性审计是否通过；
- 可部署的分散式学生是否仍然"安全但卡住"（TS1）；
- 时间表优势是否仍在（TS2）；
- 冻结 critic 能否零样本迁移（TS3a），以及它是否真的利用了 controller 信息（TS3b）。

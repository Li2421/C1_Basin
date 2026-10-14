# UTC 2026-10-13T06:34:45.522655+00:00

# Round9 η切换时刻预注册

## 范围与历史证据
先读取AGENTS.md、shared_rollout_db/README.md、round9全文、round4执行/design/protocol、Flow adapter、原FF/runtime/safety、DB planner/journal/merger/schema、视频规范。此前未做本轮新结局统计、未生成新rollout。所有证据DEVELOPMENT。
历史round4 raw/CASES/LABELS/protocol包含7Flow、64loco_test、K16、seed0..7，可复用纯η10/13、terminal/first_action/first-event信息。round7 R5/raw与round8fewshot报告是背景，不追加critic训练或表示选择。检索diagnostics文件名与DB代码没有找到已执行的同η时刻切换实验；设计后另用DB SELECT核对schedule profile并官方preflight，不将新semantic混入旧fixedη缓存。新证据缺口只有两段η时刻干预及其轨迹；旧纯η基线全部复用。
先做stage0零rollout描述，先提交一页摘要，再执行stage1一致性门；主设计在此冻结，不依stage0修改。

## 固定设计、开关时刻与一致性门
四controller=base_u_macflow、base_u_v10_local_macflow、base_u_v9_local_macflow、base_u_plus_w_v4_wide_macflow；原round4 test64state，seed0..3、原key_for原future_root2026100403和state rng_namespace。方向10→13与13→10，t100/200/300/400/500，H700，dt.05，FF，原安全/成功/碰撞/终止，JAXx64false。step编号0..699；前t个动作用a，step=t的动作开始用b。成功/碰撞在切换前终止则不继续，列入意向干预总体；不强制延长episode。
10,240主请求，不删臂。t0/700曲线端点只复用旧raw标签（2048个主基线seed格），不重跑；全7Flow stage0共7168个旧η10/13/testseed0..7记录。
一致性门32例预先固定（不按结局选例）：每controller在teststate0/1×seed0/1×有效pureη10/13各一次共8，4controller合32。η10：seed0用13→10,t0；seed1用10→13,t700。η13：seed0用13→10,t700；seed1用10→13,t0。这32个modified-runtime验证请求各有独立schedule执行UID，官方preflight后只跑其missing；不是主纯η基线补采。runtime重算的outcome、episode_length和first_action必须与同初态/seed/有效η旧raw逐位一致；first_action按float64 bytes比（包含signed zero），None按None；并记录summary首success/各collision/错误信息作为辅助。任一不一致立即停止主实验，报告FAIL与原因，不改容差或用旧cache替代门。总新完整请求上限10,272（32验证+10,240主）；unknown保留不重跑。
保持native v10_local走原FourRingRuntime，另外三个走原RingFlowAdapter；原scene_precision_guard/x64false导入与精度顺序保持。原flow_adapter/runtime/CBF/plant不改。SafetyTracker仅读取投影调用返回结果，记录any norm(projected-target)>1e-6，返回原对象；不改变solver参数/调用次序。

## 新identity、缓存与写入
DB eta表严格3个float64、canonical_hex唯一，不能把schedule假装一组三标量。采用标准eta_identity(eta_a)作请求eta_uid，将eta_a/eta_b原float64 hex及t、switch时间语义、旧controller完整UID/payload、runner codeSHA写入新的controller_config identity。复合(state,eta_a_uid,new_schedule_controller_uid,seed_key)是新执行semantic；两direction和每t均不同controllerUID，包括验证t0/700。这是新的controller/η复合身份，不改DB schema，也不把新semantic冒充旧fixedη。主两端纯η以旧controller/eta identity SELECT/preflight精确复用。
执行design/register → 官方preflight → 保存MISSING（raw与DB均核对、unknown不重跑）→ worker只写raw/trace+append_journal → 单进程merger → postflight。workers不连SQLite写入；新controller/experiment注册与merger单进程。DB读连接mode=ro，遵守conflict quarantine/ambiguous规则；AMBIGUOUS不自动reuse。用户此轮明确要求的DB注册/共享journals/merger是“只在新目录写实验文件”的必要显式例外；代码、raw、trace、报告仅在此新目录，其余旧资产不改不删、不commit。新DB行严格新experiment/controller身份，不修改旧rollout。

## 轨迹与机制（仅描述）
每rollout每10步、初态、真实终态及精确切换t保存positions/velocities/project-active；另保留完整每步positions/velocity/projection_active与simulator swept_clearance，避免未来为了完整视频重复rollout。记录switch前state（step=t动作前）及是否alive。solver error保存partial、标unknown，不拼造全程。
机制量预定：未到goal且最近2s内机器人最大位移<.02m作为local stall候选数量（非deadlock标签）；全程真实最小pair surface clearance及原margin、projection比例；每10步goal-distance/speed/角进展，近距离opposing-goal机器人最接近位置及时间。主相互作用位置取两机器人距离最小位置均值，解释其不等价真实“会车因果事件”。timeout保持timeout，Ring原评价器没有runtime deadlock分支，不追加deadlock判定。基线无轨迹，因此机制比较限已有gate完整pure轨迹和各switch轨迹，不为大批基线补跑。

## 统计、决定时刻与判定
每controller/direction的S(t)报告success下界=known-success/所有请求、上界=known-success+unknown；unknown不作为失败计数。seed先在同state内平均；bootstrap5000，state64成簇，种子202610130901，同draw跨所有controller/direction/t。95%percentile success区间，纯η的配对效应上下界CI；另给所有主切换×两基线的双侧max-T同时CI，结果探索性DEVELOPMENT。两个semantic的unknown效应lo=Ya−Yb−Ub，hi=Ya+Ua−Yb，已知same pureη端点完全抵消；不因相同seed就取消不同ηschedule的unknown。前t已终止的cases列出比例，不能删掉改变overall S(t)。
D=全程η13−全程η10。主要半效应定义：Δ(t)=S10→13(t)−S10，D>0时t_half为预定网格{0,100,200,300,400,500,700}中Δ(t)/D>=.5的最晚t；D<0用伤害H(t)=S10−S10→13(t)、Htot=S10−S13，H/Htot>=.5的同一定义。比例点估计用known下界success，同时给unknown可行区间及bootstrap t_half分布；D unknown范围跨0则INCONCLUSIVE，bootstrap非正分母记NA并报告，不clip比例或强制单调/插值。曲线非单调就报告所有通过时刻，不只呈最晚值。反向13→10另外报告移除η13前缀的rescue/loss，提供是否早期已经不可逆的证据。
H_late主操作门：base_u与v10两者t_half>=300；η10型两者在10→13,t300仍保留>=50%总η13伤害，且13→10,t200恢复>=50%该伤害（即前200η13暴露未把结果不可逆注定）。这具体化“伤害主要200后”；点判定SUPPORTED DEVELOPMENT，附bootstrap与unknown稳健性；未知边界足以翻转则INCONCLUSIVE。H_early：两必要Flow t_half<=100则SUPPORTED DEVELOPMENT；其他混合/中间曲线INCONCLUSIVE并分controller报告，不强行二分。点门不追加终点后挑选的阈值，但CI不能支持时明确不称前瞻因果定理。

## 分析4的固定模型及状态介导门
只用10→13五t、切换实际发生（alive at t）的cases，valid success/failure训练，unknown排除且报数；已提前终止不能假造t状态。物理特征为world positions8、velocities8、goals8，加真实时钟t/700，共25；不得加入controller UID、checkpoint、source标签、未来统计，onehot对照另加4维。source-fold-only按连续25维zscore（sd<1e-6尺度1）、onehot不标准化；固定二元logistic C1、sum binary NLL+.5L2非截距参数、LBFGS max2000/gtol1e-8/ftol1e-12，无调参。按64个原physicalstate LOO：同一state所有controller/seed/t均留出，不能按轨迹行随机划分。输出OOF概率、AUROC、logloss、Brier、accuracy@.5、balanced accuracy，常量baseline使用source-fold成功率。
“充分/几乎不增加”预定为：无ID OOF AUROC>=.75，Brier比fold-source常量改善>=.02；且ID带来AUROC改善95%CI上界<=.02、logloss改善CI上界<=.01（按state5000bootstrap，seed202610130902）。全部满足H_state_mediated SUPPORTED DEVELOPMENT；ID增益下界超过这些界则REJECTED对应小增益命题，其余INCONCLUSIVE。单纯ID不显著不是等价性证据。预测性状态介导描述不等同正式causal mediation，survivor选择偏差与仅4已知controller边界写明。未来state-only中途critic仅下一轮候选，本轮不训练或部署。

## 视频、资源与交付
本轮固定Flow的成功概率Q估计与单Ring时刻机制诊断，不是四场景大型训练/跨场景controller性能batch；按AGENTS纯Q例外及用户单场景机制范围，不触发四场景强制视频验收，不追加其他场景rollout。完整每步物理证据仍保留；若做Ring机制示例视频用统一render_video只读trace，标签保持actual timeout/collision/success，不充当deadlock_show/deadlock_resolution成功验收。
共同CPU<=18、GPU shard<=3，提交前实时占用扣除。当前外部作业4CPU/1GPU，本轮CPU worker初始13（留1本地编排核），每job1核、BLAS/OMP1；CPU版JAX跑原物理FF、无需GPU占满。动态作业不取消他人；资源变化只调并发不删臂。先gate真实耗时估预算；预估3–5h，若明显更长报告而不删臂。Python -B/cache/tmp在本目录，允许DB及本experiment journals必要写例外，旧代码/数据全部只读。
交付预注册收据、历史index、stage0一页摘要、gate报告及raw/trace、新identity/protocol/cases/preflight/postflight/journalsmerge收据、S(t)/t_half/机制/classifier图表、results、REPORT（首段直接回答阶段/泛化原因）与独立HANDOFF。任一一致性门不通过不进入主实验，也不能标main COMPLETE。

# 用户round9全文（冻结副本）
你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md` 和 `shared_rollout_db/README.md`，并严格遵守。

# 一、要回答的问题（只回答这一个）

在 Ring 场景下，η13 能帮助 base_u、v10_local（+26～30pp），却伤害 v9_local（−16pp）。**这个 controller 依赖的 η 效应，是在 episode 的哪个阶段产生的？**

round7–8 的结论（均为 DEVELOPMENT）：
- 静态描述 R3 和 200 步行为探测 R5 都识别不出 η 的 regime。
- 200 步内，η13 对所有 Flow 都推进得更快，但这种推进与最终成功呈负相关。
- 少样本适应只有在预设候选 η 菜单的条件下，才在 128 条标签时成立。源先验和 critic 都没有节省标签。

因此，当前最可疑的根因是：**η 的效应由后期的闭环交互决定（例如会车或停滞），短时或静态的 controller 描述看不到它。**

本轮用最直接的因果干预检验这一点：在 episode 中途切换 η，测量成功率随切换时刻的变化。

# 二、阶段 0：零 rollout 的结局分析（先做）

只读已有的 raw 记录：`diagnostics/controller_loco_real_flow_v1/raw/`，约 86k 个 JSON。

- 每条记录包含：outcome（success / timeout / collision / deadlock）、episode_length、first_success_step、各类碰撞的首次步数。
- 记录中没有轨迹。

**对象：** 7 个面板 Flow × {η10, η13}，使用 loco_test 的 64 个 state、seed 0–7。

**需要报告：**
- 各结局类型的比例。
- 成功步数的分布。
- 失败的类型与发生步数。
- 重点比较以下三组，看 η13 在每组里改变了哪一种结局、发生在第几步：
  - base_u / v10_local 在 η10 与 η13 下；
  - v9_local 在 η10 与 η13 下；
  - v9_local 与 v10_local 之间（两者静态描述相近）。

**本阶段只做描述，不作结论。** 写一页摘要后，再进入阶段 1。阶段 1 的设计不因阶段 0 的结果而改变。

# 三、阶段 1：η 切换时刻的干预实验（主实验）

## 设计（在执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**controller（4 个）：**
- η13 型：base_u、v10_local；
- η10 型：v9_local（受 η13 伤害、且与 v10_local 静态相近）、plus_w_v4_wide。

**state 与 seed：**
- state：round4 的 64 个 loco_test state。
- seed：0–3，seed key 与已有记录完全相同。这样切换前的轨迹与纯 η 基线逐步一致，可以配对比较。

**干预：**
- 前 t 步用 η_a，从第 t 步起改用 η_b，直到第 700 步。
- OrthoFlow3 修正仍然每步重算；唯一的改变是从第 t 步起，所用的 η 参数换掉。
- 两个方向：η10→η13，η13→η10。
- t ∈ {100, 200, 300, 400, 500}。
- t = 0 与 t = 700 就是纯 η 基线，直接复用已有标签，不重跑。

**规模：** 4 个 controller × 64 个 state × 2 个方向 × 5 个 t × 4 个 seed = **10,240 条完整 rollout**。

## 执行要求

- **一致性门（必须先通过）：** 修改后的 runtime 在 t = 0 和 t = 700 时，必须与已有记录逐位一致：outcome、episode_length、first_action。在 32 个 case 上检查。不通过就停下报告，不得进入主实验。
- **共享 DB：** 切换 η 是一种新的执行语义，必须登记为新的 controller / η 身份（在 identity 中写入 η_a、η_b、t），并按 shared_rollout_db 的流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。
- **轨迹记录：** 每条 rollout 每 10 步保存一次位置、速度和安全投影激活情况，供机制分析使用。只写到本实验目录。
- **其他设置：** Ring、FF、H700、K16 中的这两个 η、安全投影、成功判定、JAX x64 = false，全部不变。
- **资源：** 本 workstation 上用户所有任务合计不超过 18 个 CPU shard 和 3 个 GPU shard。
- **视频：** 本实验只涉及单个场景，属于机制诊断。按 AGENTS.md 判断是否适用视频要求；若适用，只对 ring_exchange 运行统一入口，并在报告中说明。

## 分析

**1. 成功率曲线 S(t)。** 对每个 controller × 方向，画成功率随 t 变化的曲线：
- 按 state 成组做 bootstrap 95% CI；
- 与两个纯 η 基线配对比较。

**2. 定位“决定时刻”。** 对 η13 型 controller，定义 η13 的总收益 D = S(全程 η13) − S(全程 η10)。
- t_half：η10→η13 方向上，保留至少 50% 收益的最晚切换时刻；
- 对 η10 型 controller，用同样的方法定位 η13 的伤害在何时产生。

**3. 机制。** 用轨迹描述切换前后的状态，例如停滞的机器人数、最小间距、会车所在位置，并说明 η 改变的是哪一种失败（timeout / collision / deadlock，以及发生步数）。

**4. 当前状态能否预测切换的效果（描述性）。** 以切换时刻 t 的物理状态为特征（不使用 controller 身份），用预注册的固定 logistic 回归（C = 1，按 state 留一），预测“切换到 η13 是否会成功”。比较两种情况：
- 加入 controller 身份 one-hot；
- 不加 controller 身份。

如果当前状态已经足以预测，而 controller 身份几乎不增加信息，就说明 controller 的作用主要通过它诱导出的状态分布体现。

## 预注册的判定

| 观察结果 | 结论 |
|---|---|
| η13 型 controller 的 t_half ≥ 300，且 η10 型的伤害也主要在 200 步之后产生 | **H_late SUPPORTED**：η 效应由后期交互决定。它解释了 R5（200 步）为什么失败；下一步的表示应基于后期状态，或改为 episode 中途重新选择 η |
| t_half ≤ 100 | **H_early**：效应在早期就已注定。R5 失败不是 horizon 的问题，而是统计量没有选对 |
| 介于两者之间 | 报告曲线形状，不强行二分 |
| 分析 4 中，状态预测充分，而 controller 身份增益很小 | **H_state_mediated SUPPORTED（DEVELOPMENT）**：可以设计一个与 controller 无关、在中途重新选择 η 的 critic，作为下一轮的零样本候选 |

# 四、硬性规则

- 只在新目录 `diagnostics/controller_eta_switch_time_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- 不修改控制器、物理、安全投影、事件判定或成功标准。唯一允许的改变，是在第 t 步替换 η。
- unknown 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- 所有结论标为 DEVELOPMENT。阶段 0 只做描述，不得用它来修改阶段 1 的设计。

# 五、交付

全部交付物放在 `diagnostics/controller_eta_switch_time_v1/`：
- PREREGISTRATION.md 及收据；
- 代码；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- 阶段 0 摘要；
- S(t) 曲线与 t_half；
- 机制图；
- 分析 4 的结果；
- results.json；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：η 效应在哪个阶段产生，以及这对“零样本 controller 泛化”的根本原因意味着什么。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE；事实与解释分开写，不夸大。

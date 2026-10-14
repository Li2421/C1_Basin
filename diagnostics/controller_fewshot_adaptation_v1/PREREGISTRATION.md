# Round8 少样本适应预注册

UTC 2026-10-13T05:55:06.921183+00:00

本文件在任何新采集模拟、后验、选择、统计或新模型推理之前冻结。此前仅阅读AGENTS.md、任务/早期草稿、历史报告/代码、协议及标签数组结构，核对UID交集和请求/unknown数量。本轮优先采用round8，早期草稿的256预算、S策略、C-FT不执行。

## 历史证据与独立结构复核
历史round7 REPORT/HANDOFF/results：真实四格无SPT，ρ=-.297 CI[-.854,.347]，P_rule全η13有5/7灾难；这些不证明统计意义的无信息。round6 REPORT/fit、真实REAL6 F2-R3-PCA2和round4 R0冻结模型可直接复用。round4 HANDOFF/REPORT、TRAIN/终点标签、静态特征和controller_prior、round3 P1、round5 Stage0/融合标签和粗略校准表已检索；其fixed前kstate无200次重复，不能回答本轮预算问题。execution完全不变，无需rollout、探测、Flow训练、共享DB连接。唯一缺口：对全部可采集pool的冻结critic预测，以及预定重复采集/校准/数值评价；旧终点预测直接复用，不重训critic。
StageA labels(7,192,16,8) int8；TRAIN前128state、seed0..3，每state8head；每目标4096请求槽位，unknown按controller顺序54/48/44/60/17/41/6。StageB labels(7,128,16,4)；P1 Y/U逐seed数组(14,24,16,4)，并非仅汇总成功数。P1 controller顺序不同，必须按名称对齐；unknown U=1转换-1，Y=0且U=0才失败，不造标签。主采集池4096+1536=5632请求槽位/目标；StageB扩展只4096TRAIN格、P1仅评价。P1 24state、TRAIN128、test64 UID两两无交集，继续核对physical内容与execution身份。主真实7折6源、必要base_u/v10，主終点seed4..7；hindsight固定η只testseed0..3选。
14Flow经验regime9/2/2/1来自round5；R集合{2,4,10,13}按任务给定全局预声明候选库，不能声称每折6源都实测有η4（StageA没有η4型）。主H严格只有6源TRAIN；目标P1从不进入源prior/H，目标标签只按采集序列暴露。StageB7个新增Flow用其他13Flow作源，仅E/H及safe/P0，因无对应冻结R0/F2的13源留一checkpoint，C明确INAPPLICABLE，不造模型；只描述，不进主判定。

## 完全指定的适应公式
预算B={0,4,8,16,32,64,128}，200重复；每条预算为一个现存(state,eta,seed)请求格，无放回。unknown占一次预算，但不更新成功/失败计数或critic残差。所有无法采集请求记明，绝不新生成。B0无标签E fallback源prior；H用先验；C保留冻结预测；P0固定源prior。E平局优先prior否则最小eta，H也如此。
E：a_eta=1+s_eta，b_eta=1+f_eta；未采集valid标签的eta不参与非safe选择（若只有unknown或B0则fallback）。safe仍将eta10的Beta(1,1)纳入比较。未观察头不会因unknown虚构证据。
H：预注册为共轭mixture-Beta的精确实现，不另设未经指定的prior strength。每源j、head k，a_jk=1+源TRAIN有效成功seed数，b_jk=1+有效失败seed数，即Beta(1,1)平滑；controller先验权重1/6。目标计数合并为Beta(a_jk+s_k,b_jk+f_k)。后验权重w_j∝(1/6)乘所有head的B(a_jk+s_k,b_jk+f_k)/B(a_jk,b_jk)。这积分掉head成功率不确定性，等价逐观察用各分量当前posterior predictive更新权重；unknown权重不变。H选择sum_j w_j * posterior_mean_jk最高的head，包括未目标观测头。源标签量决定Beta浓度，这是H方法本身的source-strength假设，报告其边界，不根据终点更改。H-safe安全概率使用同一controller分量j的head联合后验：sum_j w_j P(Beta_j,candidate>Beta_j,prior)；不能把两head各自独立重抽不同controller混合分量。E/H比较概率采用整数Beta参数精确log-Beta有限和，阈值>=.9；同eta概率门不需比较。
C有两个冻结骨干：C-R0与C-F2(R3-PCA2 REAL6)，各对应C-ctl和safe。把“残差”明确为概率空间逐样本y-sigmoid(logit(state,eta))，每fit独立累计；n_eta为有效样本数，offset_eta=n/(n+8)*residual_mean=sum_residual/(n+8)。部署q'_fit(s,eta)=clip(q_fit+offset_eta,0,1)。这避免另增logit残差的label clipping参数；不是round5的未收缩logit-offset。unknown不进n、不进residual。C非safe每fit各state argmax并平均实际成功率；C-ctl每fit在无标签评估state的q'均值上选固定eta。允许按任务要求用评估初态无标签平均，但任何终点标签不参与选择。safe：3fit top1一致且ensemble mean候选较eta10优势>=.10才偏离；C逐state执行，C-ctl在state均值后执行；否则eta10。3fit不作为3独立实验重复。

## 采集、配对与随机性
U在所有请求格（含未知标签结果）均匀；R筛固定四head后均匀；A在K16头上用H posterior Thompson：先抽一个共用controller分量j，再抽16个Beta后验成功率，取最大且仍有可用格的eta，在该head剩余(state,seed)格均匀。每标签后更新H，仅用已采标签；没有可用head记无法采集。A所有方法共享由H生成的路径，故E-A也获得源知识的采集帮助，E-U是完全无源采集基线；策略内H-vsE才识别适应先验额外价值。这一点在报告中明确，不将E-A称纯目标知识。
每目标/策略/重复先生成长度128的路径，全部预算为同一路径的前缀；同一路径跨全部适应方法共享。预算嵌套用于配对预算节省，不扩大样本；不按已打开标签挑采样state。RNG种子202610130800 + target_index*10000 + strategy_index*1000 + repeat_index（target0..13；strategy U/R/A=0/1/2，repeat0..199）。StageB source13与主source6分开构造。保存每条格身份、结果、posterior、偏置、选择；模拟采集代码不打开终点标签容器。

## 评价、CI与主样本效率
主终点固定test64state seed4..7；hindsight eta仅同64state seed0..3的observed-valid mean curve argmax，ties最小；不同eta/state种子组成的unknown差值按round6：lo=Ya-Yb-Ub*I(a!=b)，hi=Ya+Ua*I(a!=b)-Yb，同etaunknown完全抵消。报告success/效应上下界，unknown绝不默认失败。
E1全部7目标等权、200采集重复等权、fit先平均；E2逐必要目标及合并，相对prior；E3对每target/rep平均state/fit的paired lo<=-.10为possible，hi<=-.10为certain，同时报告平均target灾难比例和“任一次采样”target比例。SPT主口径possible E3平均=0（因此不能隐藏任何被抽到的灾难），E1同时CI下界>-.02且合并E2同时CI下界>0；CT额外两个必要target对prior保守点估计>0。
每B统一95%双侧max-T族覆盖全部12方法×3策略（P0仅一份）以及E1、合并/逐target E2、预定H-E、两个C骨干-E、C-Cctl及对应safe配对对照；这些对照在所有策略下计算。重复的P0/等价B0列只报告引用，不重复增加独立样本。CI联合两类变异：5000次physical-state成簇bootstrap（64state各controller同draw），每target独立对200次采集重复有放回抽200次，所有方法/策略/预算共享该target重复draw；fit不重抽。随机种子202610130883，StageB描述202610130884。这种CI估计200重复平均性能，另报200重复deployment效果分位数与any-catastrophe，不能称单次部署的95%保证。bootstrap计算允许对state-resampling项与repeat项用固定seed双向加性bootstrap分解以降低复杂度，但主优先精确交叉重抽；若执行精确算法不可行，先在无终点条件下记录实现与数值对照，不悄悄换算法。
B_SPT/B_CT为预定网格首次通过，没通过记>128（非无穷/129实测）。同时记录其bootstrap分布/右删失率，不按终点新增中间预算。B_2pp每target以相对hindsight固定eta的保守差lo>=-.02的最小B为点指标，另给同时CI下界>=-.02的认证版本和bootstrap分布；超过oracle的逐state策略不clip成0，gap-closed允许>1或<0，分母oracle-prior保守收益<=0时比例NA并报告。
“明显节省”预先操作化为：同策略同安全状态下H或C的B_SPT/B_CT严格更小，并在其首次过门B上的相对E配对差的同时CI下界>0；两者都满足才SUPPORTED。若比较方法>128仅给预算节省的严格下界，不把右删失当128精确值。H≈E/C≈E仅表示未建立优势，不作为等价性结论。逐state价值按所有B配对C-Cctl同时CI报告：下界>0有正证据，上界<=0有负证据，跨零INCONCLUSIVE；不按最好终点挑一个胜出预算。结论严格沿用户第五节，无法唯一分类时明示INCONCLUSIVE。

## 资源、写入与交付
只有新目录可写；不commit，不改删历史，不连共享DB，无rollout/probe。Python -B，cache/tmp/matplotlib目录在此目录；audit hook阻止外部文件写和sqlite连接。共同18CPU/3GPU shard，提交前扣其他任务；当前Slurm队列空/GPU无计算，GPU只Slurm，推理一shard/1CPU，其余CPU采集模拟按余量并发。不申请资源来重训已有模型。全部结果DEVELOPMENT，纯Q无需视频。
交付含预注册/receipt、池身份与unknown审计、历史hash索引、冻结checkpoint引用与新pool预测、所有采集序列/适应预测、JSON/图表/REPORT/HANDOFF。科学图用matplotlib standalone PNG/PDF。R5步数等价描述：6400/H700上限=9.14完整rollout，实际根据已有真实episode长度再报告；无标签probe和有标签episode不可当证据等价。当前只7主Flow/2必要、固定Ring/FF/H700/K16geometry；StageB P1与source验证同分布只描述。

# 用户round8完整任务（冻结副本）
你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不生成任何新 rollout 或探测。** 所有“目标标签”都从已有标签中抽取，以模拟采集过程。这是纯 Q 实验，不需要视频。

# 一、目标与本轮问题

研究总目标：让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在**新 controller** 上选好 η。

round4–7 的零样本路线已经基本走到头（全部为 DEVELOPMENT 证据）：

- **round6：** 约束 controller 通路后，“给 controller 信息反而有害”的问题被消除。
- **round7**（`diagnostics/controller_behavior_repr_v1/`）：真实 Flow LOCO 的“表示 × 覆盖”四格都没有达到安全正迁移。
  - **零样本无法识别 regime：** 静态 R3、200 步行为探测 R5，以及加入锚点邻域的融合覆盖，都不能在零样本下识别真实 Flow 的 η regime。
  - **ρ(Δprog, ΔS) 为负：** 14 个真实 Flow 上为 −0.30，CI [−0.85, 0.35]。7 个面板 Flow 的 Δprog 全部 > 0，因此按 Δprog 选 η13 的零参数规则 P_rule 全选 η13，E3 = 5/7。短时进展预测不了长时成功。
  - **必要 controller 上没有收益：** 安全规则全部回退到 η10，在 base_u 和 v10_local 上收益为 0。
  - **决策本身是 controller 级的二选一：** η13 对 base_u 是 +30pp，对 v10_local 是 +26pp，对 v9_local 是 −16pp，对其余面板 Flow 也有害。
  - **逐 state 选择在新 controller 上有害：** R0 的逐 state argmax 比固定源先验 η10 低约 5pp（54.9% vs 60.2%）。
  - **Q 拟合变好不等于选择变好：** R5-PCA2 的 Q 拟合 NLL 比 R0 更好（0.370 vs 0.426），但选择没有变化。

由于只有 2/14 个真实 Flow 属于 η13 型，零样本在现有 controller 总体上无法辨识 regime。“增加独立的 η13 型 controller”仍为 INCONCLUSIVE，而且需要训练新的 Flow，成本高。

本轮回答一个可部署的互补问题：**面对一个新的真实 Flow controller，至少需要多少条有标签的目标 rollout，才能安全地选对 η？源 controller 上学到的知识（regime 先验、critic）能把这个数量降低多少？**

最后一个问题是本轮对研究目标最直接的度量：critic 或源先验学到的 controller–Q 结构，体现为它们相对“只用目标标签”所节省的样本数。

# 二、必读材料（亲自核对关键数字）

1. `diagnostics/controller_behavior_repr_v1/`：REPORT.md、HANDOFF.md、results.json。
2. `diagnostics/controller_pathway_fix_v1/`：REPORT.md，以及 F2 与 R0 的冻结 checkpoint。
3. `diagnostics/controller_loco_real_flow_v1/`：HANDOFF.md、REPORT.md、LABELS.npz、protocol.json、`fits/`、`controller_prior/`。
4. `diagnostics/controller_q_transfer_round3_v1/`：P1 数据，即 14 个 Flow × 24 个 state × K16 × Q4。
5. `diagnostics/controller_flow_blend_v1/`：
   - STAGE_0_RESULTS.json，含 14 个 Flow 的 TRAIN 曲线；
   - stage_2 中的融合标签；
   - REPORT 中“目标少样本校准参照”表（这是 round5 用固定前 k 个 state 做的粗略版本，本轮取代它）。
6. `docs/prompt_fewshot_controller_adaptation_20261013.md`：本任务的早期草稿，未执行。以本文件为准。

# 三、数据划分（先审计身份与 unknown，写进预注册）

真实 Flow 面板为 round4 Stage A 的 7 个 Flow，用 7 折 LOCO 划分；必要折为 base_u 和 v10_local。

**目标标签池（模拟采集只能从这里抽取）：**
- 目标 controller 在 128 个 loco_train state 上的已有标签：8 个掩码 η 头 × Q4；
- 目标 controller 在 P1 的 24 个 state 上的已有标签：K16 × Q4。

**评估：**
- 只用目标的 64 个 loco_test state × seed 4–7。
- 目标自身的 hindsight 最佳固定 η 用 loco_test 的 seed 0–3 选出，只作参照。

**源知识：** 只能用 6 个源 Flow 的标签与冻结模型。源模型是该折 round6/round7 已冻结的 REAL6 checkpoint。

**描述性扩展：** 对另外 7 个 Stage B Flow 做留一法：目标池为 loco_train 标签，评估用 P1 的 24 个 state。因为评估与源知识共用 P1，结果只作 DEVELOPMENT 描述，不进入主判定。

# 四、设计（在任何新分析之前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 预算与重复

- 预算 B（目标 rollout 条数）∈ {0, 4, 8, 16, 32, 64, 128}。
- 每个“预算 × 策略 × 目标”重复 200 次随机抽样，随机种子预先固定。
- 每次抽样不放回；某格已有标签用尽时，记为“无法采集”并报告，不得伪造。
- unknown 标签按配对保守规则处理：既不算成功，也不算失败。

## 采集策略

| 代号 | 策略 |
|---|---|
| U | 在可用的 (state, η, seed) 格上均匀随机抽取 |
| R | 只在源 regime 集合 {η2, η4, η10, η13} 上均匀抽取（这个集合来自源知识） |
| A | 自适应：用方法 H 的后验做 Thompson 抽样，选择下一条标签的 η；state 在可用格中随机 |

## 适应方法

| 代号 | 方法 | 说明 |
|---|---|---|
| P0 | 源先验 | 固定选 η10（7 折的源先验都是 η10），对应 B = 0 |
| E | 只用目标标签 | 对每个 η 用 Beta(1,1) 后验，取后验均值最大的 η；没有标签的 η 不参与；平局时依次取源先验、编号最小者 |
| H | regime 混合先验 | 以 6 个源 Flow 的 K16 成功率曲线（源 TRAIN 标签）为混合分量，等权先验。用目标标签计算各分量的后验权重，再与目标自身的 Beta 计数合并，取后验期望最大的 η。分量曲线用 Beta(1,1) 平滑 |
| C | 冻结 critic + 目标偏置 | 用该折冻结的 R0（以及 F2 R3-PCA2 REAL6）的逐 state 预测，加上每个 η 的目标偏置：offset_η = n_η/(n_η+8) × (目标残差均值)。按 state 取 argmax |
| C-ctl | C 的 controller 级版本 | 把 C 校准后的预测在评估 state 上取平均，选一个固定 η。用于区分逐 state 选择与 controller 级选择 |

**安全版本：** E、H、C、C-ctl 都另有一个安全版本（加后缀 -safe）。只有当所选 η 优于源先验 η10 的后验概率 ≥ 0.9 时，才偏离 η10。
- E 与 H 用各自的后验计算这个概率；
- C 与 C-ctl 用 3 个 fit 一致且平均预测优势 ≥ 0.10 作为判据（同 round6 的保守规则）。

所有超参数（Beta(1,1)、n0 = 8、0.9、0.10、200 次重复、预算网格）都在预注册中写定，不得根据终点结果调整。

## 终点（只用目标 loco_test 的 seed 4–7）

对每个“方法 × 策略 × B”计算以下指标。CI 要把 state 成组 bootstrap 和 200 次抽样的变异都纳入；同一预算下所有方法共用一个 max-T 同时 CI 族。

| 编号 | 指标 |
|---|---|
| E1 | 7 个目标的平均选择成功率相对源先验：非劣界 −2pp |
| E2 | 必要 controller（base_u、v10_local）相对源先验的收益：逐个报告，并给出合并值 |
| E3 | 灾难率：比源先验低 ≥ 10pp 的目标所占比例。在 200 次抽样上取平均，同时报告“任一次抽样发生灾难”的比例 |
| G | 距目标 hindsight 最佳固定 η 的差距，以及已闭合的比例 |

**判定定义：**
- **安全正迁移（SPT）：** E1 过门、E3 = 0，且 E2 合并 CI 下界 > 0。
- **完整迁移（CT）：** SPT 成立，且 base_u 与 v10_local 的收益点估计都 > 0。

## 样本效率指标（主分析）

- **B_SPT / B_CT：** 每种“方法 × 策略”首次达到 SPT / CT 的最小预算。若在 128 以内都达不到，记为 > 128。
- **B_2pp：** 每个目标达到“距 hindsight 最佳固定 η ≤ 2pp”所需的最小预算。必要目标与非必要目标分开报告。
- **源知识的价值（本轮的核心度量）：**
  - H vs E、C vs E 的预算节省，即 B_SPT 与 B_CT 之差；
  - 每个 B 下成功率的配对差，附同时 CI。
- **逐 state 选择的价值：** C vs C-ctl。逐 state 选择在少样本下是否有额外收益，还是像零样本时一样有害。

# 五、结论映射（预注册，必须按此下结论）

| 观察结果 | 结论 |
|---|---|
| H 或 C 的 B_SPT 明显小于 E（同时 CI 支持） | **源知识可迁移为样本效率**：critic 或源先验学到了可用的 controller–Q 结构。零样本失败只是 regime 不可辨识，而不是没有学到结构。可部署路线是“源先验 + 少量目标标签” |
| H ≈ E，C ≈ E，但都能在小预算下达到 SPT | 少样本可行，但源知识没有带来额外价值；controller–Q 结构的学习在本设定下尚未体现 |
| 所有方法在 128 以内都达不到 SPT | 即使有目标标签，安全选择也很昂贵。报告瓶颈是噪声（Q 估计方差）还是 regime 判别本身 |
| C > C-ctl | 逐 state 选择在少样本下有价值 |
| C ≤ C-ctl | 可迁移的部分是 controller 级 regime，逐 state 选择没有价值（与 round4 的结论一致） |

此外报告：如果用零样本的 R5 探测来分配采集预算（R5 每个 controller 6,400 个模拟步，不带标签），等价于多少条目标 rollout。这只作描述。

# 六、硬性规则

- **不得新增 rollout 或探测，不得写入共享 DB。** 读取 DB 一律用只读或 immutable 模式。
- **文件写入范围：** 只在新目录 `diagnostics/controller_fewshot_adaptation_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **seed 隔离：** 任何选择、调参或校准都不得使用 loco_test 的 seed 4–7；hindsight 参照只能用 seed 0–3。
- **证据等级：** 所有 state 都已被打开过，结论一律标为 DEVELOPMENT。
- **JAX x64：** 记录并匹配原设置（false）。
- **资源：** 本 workstation 上用户所有任务合计不得超过 18 个 CPU shard 和 3 个 GPU shard。提交前先查看实时占用。

# 七、交付

全部交付物放在 `diagnostics/controller_fewshot_adaptation_v1/`：

- PREREGISTRATION.md 及收据；
- 代码；
- 标签池审计：每个目标可采集的格数、unknown 数，以及每个 η 的覆盖情况；
- results.json；
- 学习曲线图：成功率与 E3 随 B 的变化，按方法 × 策略画，必要目标与非必要目标分开；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 必须做到以下几点：

1. 回答：新 controller 需要多少条目标 rollout 才能安全选对 η，必要目标与非必要目标分开报告。
2. 回答：源知识（H、C）节省了多少目标标签，并按第五节的结论映射给出结论。
3. 回答：少样本下，逐 state 选择是否有价值。
4. 对研究的含义：
   - controller 泛化是否应改为“源先验 + 少样本快速适应”的路线；
   - 零样本还剩什么值得做，例如训练新的独立 η13 型 controller 的成本是否值得。
5. 每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。事实、解释和猜想分开写，不夸大。写明结论的边界：Ring、FF、K16、固定几何，以及只有 2 个必要的真实 controller。

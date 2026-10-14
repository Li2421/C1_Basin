# Round7 行为表示 × 覆盖预注册

UTC: 2026-10-13T05:09:44.888555+00:00

# 执行细节与独立事实复核（优先于所附提示中的历史事实描述）

此文件在本轮新统计、模型重放、拟合前冻结。此前仅阅读历史报告、代码、协议及数组结构/NaN位置、核对state UID集合；未计算本轮I1–I4或新终点结果。用户提示全文附后并作为完整设计。所有结果DEVELOPMENT。

## 历史证据、兼容性及缺口
round6 controller_pathway_fix_v1 的 HANDOFF/REPORT/PREREGISTRATION、util.py、fit.py、fit_real.py、logistic.py、evaluate.py、冻结 fits/shared_sparse_fits/logistic、results/real_results/blend_results/eligibility/checkpoint index可直接复用。round5 controller_flow_blend_v1 的TRAIN标签、静态特征、831个完整raw短探测、失败partial、R5_FEATURES与协议可复用。round4 controller_loco_real_flow_v1 的Stage A标签和R0、round3 P1的24state验证标签可复用。execution语义未变；无需训练Flow、无新rollout、无新探测、无需连接DB。唯一新证据缺口是预定R5表示/覆盖组合的小critic和分类器预测；R3 REAL6与R3融合LOPO复用冻结模型。全量hash索引在stage0制作。

## 已独立核对的结构事实
R5_FEATURES为(26,160) float32，前14anchor_order、后12blend；probe.py.merge按controller、state顺序、eta(10,13)、5统计量展开。每格依次为总goal距离相对进展、投影激活比例、整个含初始positions轨迹的最小两两距离、actions范数平均、每步机器人平均速度<.01的步数；动作均值不是位移除dt的估计。POSTFLIGHT为832请求、831有效；唯一NaN为第9行base_u_v3_macflow第105..109列；raw error step132 CBFSolverError，statistics=null，physical_steps132。不填零、不使用partial构造特征。
probe16state的UID与R4全部192、R5全部128均无交集；融合TRAIN96全部属于R4 TRAIN128。TRAIN数组(12,96,16,4)、终点数组(12,32,16,2)；掩码每state8头，其余为未请求。真实LABELS数组(7,192,16,8)，仅TRAIN前128/seed0..3用于fit；终点最后64/seed4..7仅stage0复现与最终评价读取。阶段1代码单独使用预先导出的TRAIN文件，不导入旧Data（旧Data会读取同时含终点的NPZ）。

重要历史差异：round6 PREREGISTRATION与util.Data明确真实LOCO R3均值用128 TRAIN state，融合LOPO用96共同state，并非全部96。以文件为准：stage0与R3×REAL6精确复用128state模型；R3×REAL6+BLEND保留真实Flow的128state均值，融合取已有96state均值，不新增静态查询。这样同一真实Flow输入保持不变，但融合均值的state组成不同，是覆盖比较的局限。阶段1全部26controller的R3/R4_action按共同96state均值；I4用14真实Flow拟合描述性zscore/PCA。不得悄悄称R3全为96。

F2源码确认16/16/16（两个16隐层和16输出）SiLU，末层零初始化，tanh界1；AdamW lr.0008/wd.01/clip5，12000step、batch128head、controller均匀抽样，fit17/23/41，源验证3fit平均NLL选共同step；schedule50,100,200..12000(200间隔)。x64=false。F3 std为population std；保守规则三fit一致且均值优势>=.10。F4源only multinomial logistic，sum cross-entropy+.5L2非截距权重，L2=1，source坐标再次标准化，缺失类概率0，confidence>=.6选regime否则prior。真实prior只6原Flow定义；增广不改R0/P1验证/prior。

## 预先固定的实现与统计细节
缺失主方案全26controller删除state10 eta13五维，R5为155；所有差值均删除state10，以15state均值。Δprog选择进展比例，因为它与任务成功最直接相关，其余四差值仅探索。敏感性仅stage1/3：每折源且不含base_u_v3在缺失格均值插补；PCA160维、delta使用16state，完整controller保留原值。敏感性不进入主判定，额外stage3 30个F2 fit（主设计135个，加敏感性30=165独立seed fits），不删任何臂。SPARSE每表示只优化一套，跨4fold哈希复用；每fold仍保存预测/引用。
PCA以source controller等权zscore（sd<1e-6尺度1），SVD分量最大绝对载荷定正，目标仅投影；F2输入physical23+pad513并用源训练rows归一化，沿用round6。R5-Δprog的P_rule使用未zscore原始差值的0阈值。
I1 bootstrap5000次，seed20261013071：real controller簇、blend pair簇、combined14单Flow+4pair簇；percentile95%CI，常量/退化draw报告并剔除，绝不将NaN改0；真实14Flow的主ρ CI是结论映射中I1信息门，其余独立报告。I2 ridge alpha1带不惩罚截距，LOO真实13源；blend LOPO源14anchor+其他9blend（不读任何test标签）。每折源拟合预处理，combined表示拼接两个source-only PCA2。Spearman与sign(pred)==sign(truth)，恰好0作为第三符号。I3 LOO 1NN欧氏、并列按源顺序；logistic C1与round6 sum-NLL/L2定义相同，无调参。报告fixed majority9/14与每折source-majority参照。
I4描述性预测：H_static_recipe为d(v10,v9)<d(v10,base)分别在R3全维/PCA2；H_behavior_regime为d(base,v10)<d(base,v9)且d(base,v10)<d(v10,v9)，分别在R5PCA2与raw delta（source14标准化）。两表示均满足才SUPPORTED，否则REJECTED IN THIS DESCRIPTIVE TEST；不影响主结论。
stage0从冻结checkpoint重新推理所有15task的R0/F2_PCA2与F4分类器；主门为每个选择完全一致、每个controller成功上下界/相对收益与round6结果1e-10内相同；logit浮点误差只记录，不加未要求的误差门。若选择不一致，先定位并报告，绝不进入stage1/新fit。不能借round6曾允许原cache绕过本轮主门。
E1使用same physical state跨全部controller/fit联合resample，5000次；real seed20261013072，blend20261013073。unknown按round6配对上下界、同eta精确相消。E1 max-T family为本测试床18个主要方法（3表示×2覆盖×3规则）加唯一P_rule；B0/prior与argmax/k.5仅描述。blend P_rule按coverage若实际不同保留两个，等同则报告一个重复引用。E2同时校正：每real单元格R3三规则或R5六规则，联合vs prior与vs R0两个contrast上下界做95%双侧max-T；SPT要求两个比较的下界均>0；P_rule单独族但联合两个比较。CT另要求两个必要controller相对prior与R0的保守点估计均>0。blend按coverage用同样R3/R5族，必要controller沿用旧gate，不重新选。所有指标及point收益明确保守区间。敏感性使用单独family且不改变主判定。
按附录完整结论映射，优先同时报告两个补救格均通过；四格失败时才用真实I1/P_rule分流。覆盖失败仅否定锚点邻域增广，独立eta13真实controller覆盖假设INCONCLUSIVE。所有单项假设用SUPPORTED/REJECTED/INCONCLUSIVE，结果为DEVELOPMENT，边界Ring/FF/H700/K16固定几何、2必要真实Flow。

## 写入、资源与失败门
仅此新目录可写，不commit、不改/删历史。不导入或调用probe worker/merge、rollout/db connect。Python -B/PYTHONDONTWRITEBYTECODE=1；MPLCONFIGDIR、TMPDIR、cache均定向此目录。GPU只Slurm，最多3并发GPU shard各1CPU，给本地审计/编排留1核，总CPU<=4（比用户4shard上限更保守）；BLAS/OMP1。已读round6 RESOURCE_POLICY(旧18CPU/3GPU)与round5旧资源授权，用户本轮4CPU/4GPU覆盖旧上限。实时squeue空、nvidia-smi0进程；不取消其他任务。用正式首个F2耗时估全部预算，若>8h先报告再执行；旧实测三fit37.43s提示预算远低8h，队列不保证。任何stage0失败立即暂停后续阶段，先诊断与报告，不把门失败写COMPLETE。

## 核对后的原Flow regime（历史Stage0记录）
[
  {
    "controller": "base_u_plus_w_v4_wide_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_macflow",
    "best_eta": 13
  },
  {
    "controller": "base_u_plus_w_v3_macflow",
    "best_eta": 2
  },
  {
    "controller": "base_u_plus_w_v4_train_targeted_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_v10_local_macflow",
    "best_eta": 13
  },
  {
    "controller": "base_u_plus_w_v4_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_v9_local_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_v4_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_plus_w_v6_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_v3_macflow",
    "best_eta": 2
  },
  {
    "controller": "base_u_v2_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_plus_w_v5_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_v7_local_macflow",
    "best_eta": 10
  },
  {
    "controller": "base_u_v8_local_macflow",
    "best_eta": 4
  }
]
## 核对后的pair（历史stage1记录）
[
  {
    "controller": "blend_pair0_lambda25",
    "pair": 0,
    "A": "base_u_macflow",
    "B": "base_u_plus_w_v4_wide_macflow",
    "lambda_value": 0.25
  },
  {
    "controller": "blend_pair0_lambda50",
    "pair": 0,
    "A": "base_u_macflow",
    "B": "base_u_plus_w_v4_wide_macflow",
    "lambda_value": 0.5
  },
  {
    "controller": "blend_pair0_lambda75",
    "pair": 0,
    "A": "base_u_macflow",
    "B": "base_u_plus_w_v4_wide_macflow",
    "lambda_value": 0.75
  },
  {
    "controller": "blend_pair1_lambda25",
    "pair": 1,
    "A": "base_u_macflow",
    "B": "base_u_v9_local_macflow",
    "lambda_value": 0.25
  },
  {
    "controller": "blend_pair1_lambda50",
    "pair": 1,
    "A": "base_u_macflow",
    "B": "base_u_v9_local_macflow",
    "lambda_value": 0.5
  },
  {
    "controller": "blend_pair1_lambda75",
    "pair": 1,
    "A": "base_u_macflow",
    "B": "base_u_v9_local_macflow",
    "lambda_value": 0.75
  },
  {
    "controller": "blend_pair2_lambda25",
    "pair": 2,
    "A": "base_u_v10_local_macflow",
    "B": "base_u_plus_w_v4_wide_macflow",
    "lambda_value": 0.25
  },
  {
    "controller": "blend_pair2_lambda50",
    "pair": 2,
    "A": "base_u_v10_local_macflow",
    "B": "base_u_plus_w_v4_wide_macflow",
    "lambda_value": 0.5
  },
  {
    "controller": "blend_pair2_lambda75",
    "pair": 2,
    "A": "base_u_v10_local_macflow",
    "B": "base_u_plus_w_v4_wide_macflow",
    "lambda_value": 0.75
  },
  {
    "controller": "blend_pair3_lambda25",
    "pair": 3,
    "A": "base_u_v10_local_macflow",
    "B": "base_u_v9_local_macflow",
    "lambda_value": 0.25
  },
  {
    "controller": "blend_pair3_lambda50",
    "pair": 3,
    "A": "base_u_v10_local_macflow",
    "B": "base_u_v9_local_macflow",
    "lambda_value": 0.5
  },
  {
    "controller": "blend_pair3_lambda75",
    "pair": 3,
    "A": "base_u_v10_local_macflow",
    "B": "base_u_v9_local_macflow",
    "lambda_value": 0.75
  }
]

# 用户设计全文（冻结副本）

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不生成任何新 rollout 或探测。** 只用已有标签、round5 已采集的无标签闭环行为探测（R5）和冻结模型，训练小型 critic 与分类器。这是纯 Q 实验，按 AGENTS.md 不需要视频。

# 一、目标与本轮问题

研究总目标：让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在**未见过的 controller** 上选出更好的 η。

到 round6 为止的结论（全部为 DEVELOPMENT 证据）：

**1. “给 controller 信息反而有害”已定位并修复。**
- 原因：controller 通路缺乏归纳偏置，即输入高维、没有约束，argmax 又会选中被高估的 η。
- 修复：低维 PCA、在冻结 R0 上的有界残差（F2）、悲观/保守选择（F3）、两段式 regime 模型（F4）。这些修复消除了灾难。

**2. 真实 Flow 上仍没有“既安全又有效”的正迁移。**

round6 真实 LOCO（7 折）中，F2_PCA2（输入为 R3 静态描述的 PCA-2）的逐 controller 选择如下。收益相对源先验，取 unknown 保守下界：

| controller | regime 类型 | F2_PCA2 的选择 | 收益 |
|---|---|---|---|
| v10_local | η13 型 | 70–91% 选 η13 | +17.6 至 +21.5pp |
| base_u | η13 型 | 约 95–100% 选 η10 | −1.2 至 +1.4pp |
| v9_local | η10 型 | 53–88% 选 η13 | −10.2 至 −13.3pp（E3 = 1/7） |

由此有一个可检验的**猜想**（本轮要检验）：R3 静态描述反映的是训练配方的相似性（local 系列 vs plus_w 系列），而不是 η regime。round5 Stage 0 中，R3 与 R4_action 的 14-Flow LOO regime 分类准确率都没有超过多数类基线 64.29%。

**本轮问题：真实 Flow 上剩下的瓶颈是“表示”还是“覆盖”？**
- **表示：** 静态描述捕捉不到决定 regime 的闭环行为；换成闭环行为探测（R5）就能迁移。
- **覆盖：** 每个必要折的源集合里只有 1 个 η13 型 exemplar；增加 regime 覆盖就能迁移。这里的增加方式是加入不泄漏的融合 controller。

**检验方法：** 在真实 Flow LOCO 上做“表示 × 覆盖”析因，全部零 rollout。

# 二、必读材料与需要亲自核对的事实

## 材料

1. `diagnostics/controller_pathway_fix_v1/`
   - HANDOFF.md、REPORT.md、PREREGISTRATION*.md；
   - results.json、blend_results.json、real_results.json、STAGE2_ELIGIBILITY.json；
   - 代码：util.py、fit.py、fit_real.py、logistic.py、evaluate.py；
   - `fits/`、`shared_sparse_fits/`、CHECKPOINT_INDEX.json。
2. `diagnostics/controller_flow_blend_v1/`
   - HANDOFF.md、REPORT.md；
   - STAGE_0_RESULTS.json：14 个真实 Flow 的 regime 与固定 η 曲线；
   - stage_1/results.json：融合 pair 的定义；
   - stage_2/：融合标签、protocol.json、TRAIN_ETA_MASK.json；
   - probe.py；
   - **`stage_2/probes/`**：R5_FEATURES.npz、POSTFLIGHT.json、RAW_TRACE_INDEX.json、protocol.json、raw/。
3. `diagnostics/controller_loco_real_flow_v1/`
   - Stage A 7 折真实 LOCO 的标签、protocol.json、R0 checkpoint、P1 源验证集。

## 需要核对的事实

以下事实我已只读核对过。你要独立复核，并写进 PREREGISTRATION。若与文件不符，以文件为准，记录差异，并评估是否影响设计。

**R5 特征**
- R5_FEATURES.npz 中 `features` 为 (26, 160) float32。controller 顺序为 14 个真实 Flow（anchor_order）在前，12 个融合 controller 在后。
- 160 = 16 个 probe state × {η10, η13} × 5 个统计量。维度排列以 probe.py 的 merge() 为准。
- 5 个统计量的定义在 probe.py 第 62–92 行：
  1. 200 步内总目标距离的进展比例；
  2. 安全投影激活比例；
  3. 最小两两距离；
  4. 平均速度；
  5. 平均速度 < 0.01 的步数。
- 唯一失败的探测：base_u_v3_macflow，probe state_index 10，η13，第 132 步 CBFSolverError。因此该行有 5 个 NaN。
  - base_u_v3 不在真实 LOCO 的 7 个 Flow 中，只影响阶段 1，以及作为源出现在融合族测试床中。
- 16 个 probe state 与 round4 的 192 个 state、round5 stage2 的 128 个 state 都不重叠。

**regime 与融合 pair**
- Stage 0 regime（14 个真实 Flow，依据稀疏 TRAIN 标签）：

  | regime | controller |
  |---|---|
  | η10 | v4_wide、v4_train_targeted、plus_w_v4、v9_local、v4、plus_w_v6、v2、plus_w_v5、v7_local |
  | η13 | base_u、v10_local |
  | η2 | plus_w_v3、v3 |
  | η4 | v8_local |

- 融合 pair（λ 为 B 的权重，λ ∈ {.25, .5, .75}）：

  | pair | A（η13 型） | B（η10 型） |
  |---|---|---|
  | pair0 | base_u | v4_wide |
  | pair1 | base_u | v9_local |
  | pair2 | v10_local | v4_wide |
  | pair3 | v10_local | v9_local |

**融合 controller 的标签**
- TRAIN 标签：96 个 state × 每 state 8 个掩码 η 头（TRAIN_ETA_MASK）× Q4。
  - 这 96 个 state 全部属于 round4 的 128 个 loco_train state，与 64 个 loco_test 终点 state 不重叠。
- test 标签：32 个新 state × K16 × seed 0–3。必要性门用 seed 0–1，终点用 seed 2–3。

**round6 的设置**
- R3：在共同的 96 个初态上取 controller 级均值，只用源 controller 做 z-score + SVD。
- F2：
  - 结构：冻结同折 R0，加 16/16/16 SiLU 残差网络（末层零初始化，输出 1.0·tanh，weight decay 0.01）；
  - 优化：AdamW，lr 0.0008，clip 5，12,000 步；
  - fit seed 17/23/41，按 3 个 fit 平均的源验证 NLL 选 step。
- F3a：悲观选择 κ = 1；κ = 0.5 为敏感性分析。
- F3b：保守偏离。只有 3 个 fit 的 top-1 一致，且平均预测优势 ≥ 0.10 时才偏离源先验。
- F4：只用源 controller 训练的 logistic（L2 = 1）。最大概率 ≥ 0.6 时选该 regime 的 η，否则回退源先验。

**两个测试床**
- 真实 LOCO：
  - 每折 6 个源 Flow × 128 个 TRAIN state（Q4）；
  - 源验证集 = P1 的 24 个 state；
  - 终点 = 64 个 loco_test state × seed 4–7；
  - 必要折 = base_u、v10_local；
  - 7 折的源先验都是 η10。
- 融合 LOPO：
  - SPARSE：14 个真实 Flow 作源；
  - DENSE：在 SPARSE 基础上，再加 9 个非留出 pair 的融合 controller；
  - 终点 = 32 个 state × seed 2–3。

# 三、设计（在任何新分析之前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 0：复现与 R5 审计（任何一项不通过，先定位原因并报告，不得继续）

**1. 复现 round6。** 用冻结 checkpoint，在真实 LOCO 与融合 LOPO 上重放 round6 的以下臂，选择与成功率须与 round6 报告逐项一致：
- R0；
- F2_PCA2（悲观 κ = 1、保守）；
- F4。

**2. R5 审计。**
- 从 831 条 raw 记录重新计算统计量，与 R5_FEATURES.npz 比对，报告最大绝对误差，并确认 160 维的排列。
- 确认 NaN 只出现在上述位置。
- 报告每个 controller 的有效探测数，以及各统计量的分布。
- 确认没有以 0 填充。

**3. 缺失格的处理（预先写定，对所有 controller 对称）。**
- 主方案：
  - 所有 controller 都删去（probe state 10, η13）这一格的 5 维，得到 155 维；
  - 需要 η13 − η10 差值的特征，所有 controller 都去掉 probe state 10，只用 15 个 state。
- 敏感性方案（只在阶段 1 与阶段 3 报告）：在每折内，用该折源 controller（不含 base_u_v3）在这一格上的均值，插补 base_u_v3 的这 5 维。
- 失败探测的 132 步部分轨迹不用于构造特征，只做描述。

## 表示

所有表示的 z-score 与 PCA 只用该臂的训练 controller 拟合；目标 controller 只做投影。

- **R3-PCA2**：同 round6。
- **R5-PCA2（主）**：155 维 → 按维 z-score（sd < 1e-6 时尺度取 1）→ SVD → 取前 2 个主成分。
- **R5-Δprog（1 维，预注册的领域对比量）**
  - 定义：Δprog(C) = 15 个 probe state 上 [进展比例(C, s, η13) − 进展比例(C, s, η10)] 的均值，再做 z-score。
  - 在预注册中写明选“进展比例”的理由：它与任务成功最直接相关。
  - 其余 4 个统计量的差值只在阶段 1 做描述性报告。
- **仅在阶段 1 额外使用**：R3（全维）、R4_action、R5（155 维全维）、R3-PCA2 ⊕ R5-PCA2。

## 阶段 1：信息量与可辨识性审计（不训练 Q，不读任何终点 seed）

### 真值定义

- **真实 Flow：** regime 用 Stage 0 的 best_eta。ΔS(C) = S(C, η13) − S(C, η10)，取自 STAGE_0_RESULTS.json 的 observed_valid_curve，与 Stage 0 使用相同的 TRAIN 数据和 unknown 规则。
- **融合 controller：** regime 与 ΔS 用其 96 个 TRAIN state 的标签计算，regime 定义沿用 round6 util.Data。若最佳 η 不在 {2, 4, 10, 13} 中，如实报告。不得使用 test state 的任何 seed。
- **局限（须在报告中说明）：** 在稀疏掩码设计下，η13 与 η10 是在不同的 state 子集上观测的，因此 ΔS 含有 state 组成差异。

### 分析

**I1 零参数信息检验（主）**
- 计算 Spearman ρ(Δprog, ΔS)，分别在 14 个真实 Flow、12 个融合 controller、全部 26 个 controller 上计算。
- bootstrap 95% CI 的重抽单元：
  - 真实 Flow：按 controller 重抽；
  - 融合：按 pair 成簇重抽；
  - 合并：每个真实 Flow 各为一簇，每个融合 pair 为一簇。
- 其余 4 个统计量的差值同样报告，但标为探索性。

**I2 连续可辨识性**
- 对每种表示，用固定 α = 1 的 ridge 回归预测 ΔS。
- 划分：真实 Flow 用 LOO；融合用留 pair 法，锚点 Flow 保留在源中（同 round5）。
- 报告留出预测的 Spearman 与符号准确率。

**I3 regime 分类（与 round5 可比）**
- 在 14 个 Flow 上做 LOO，分类器为 1-NN 与 logistic（C = 1）。
- 报告准确率、相对多数类 64.29% 的差，以及混淆矩阵。

**I4 “配方 vs regime”诊断（预注册的描述性预测）**
- **H_static_recipe**：在 R3 与 R3-PCA2 中，v10_local 离 v9_local 比离 base_u 更近。
- **H_behavior_regime**：在 R5-PCA2 与 Δprog 中，base_u 与 v10_local 彼此更近，且二者都离 v9_local 更远。
- 报告内容：
  - 7 个面板 Flow 的两两距离（用 14 个真实 Flow 做 z-score，仅作描述）；
  - 最近邻排序；
  - 7 个面板 Flow 的 Δprog 值及其符号。

## 阶段 2：真实 Flow LOCO 的“表示 × 覆盖”析因（主检验）

### 因素 A：表示

R3-PCA2、R5-PCA2、R5-Δprog。

### 因素 B：覆盖

**REAL6：** 每折 6 个源真实 Flow，同 round6。

**REAL6+BLEND：** 在 REAL6 基础上，加入所有“两个锚点都在本折源集中”的融合 controller 的 TRAIN 标签（96 state × 8 掩码头 × Q4）。
- 各折加入的 pair：

  | 留出的 Flow | 加入的 pair |
  |---|---|
  | base_u | pair2、pair3 |
  | v10_local | pair0、pair1 |
  | v4_wide | pair1、pair3 |
  | v9_local | pair0、pair2 |
  | 其余 3 折 | 全部 4 个 pair |

- 逐折报告新增 controller 的数量及其 regime。
- 只用融合的 TRAIN 标签，不用融合 test state 的任何标签。
- 注意：这种增广只增加已有锚点附近的 exemplar，不是独立的新 η13 型 controller。

**两种覆盖之间保持不变的项**
- 冻结的同折 R0（round4 Stage A）；
- 源先验 η：只由 6 个源真实 Flow 定义，因此回退时的选择不变；
- 源验证集：只用真实 Flow 的 P1 24 个 state；
- 训练预算、fit seed 17/23/41、JAX x64 = false。

z-score、PCA 和 F4 分类器则在各臂的全部训练 controller 上拟合。

### 选择规则

- **主要规则（3 条）：**
  - F2 + 悲观选择 κ = 1；
  - F2 + 保守偏离；
  - F4：logistic，阈值 0.6，输入为该表示，标签为训练 controller 的 regime。
- **只作描述：** F2 argmax、κ = 0.5。

### 零参数参照 P_rule（不学习，只用 R5）

- 若源先验 ∈ {η10, η13}：Δprog(目标) > 0 时，所有 state 都选 η13；否则选 η10。
- 若源先验不在其中：保持源先验。
- 阈值固定为 0，不得调整。它对应的可学习版本是“F4 + R5-Δprog”。

### 基线

- B0（R0）和源先验。
- R3-PCA2 × REAL6 直接复用 round6 的 fit，但纳入本轮的同时推断族。

### 拟合量

| 内容 | 数量 |
|---|---|
| R5-PCA2、R5-Δprog × {REAL6, REAL6+BLEND} × 7 折 × 3 seed | 84 |
| R3-PCA2 × REAL6+BLEND × 7 折 × 3 seed | 21 |
| 合计 F2 拟合 | 105 |

另有若干小型 logistic。

## 阶段 3：融合族 LOPO（次检验）

- 设定同 round6：4 折 × SPARSE / DENSE。
- 表示：R5-PCA2、R5-Δprog；R3-PCA2 复用 round6 的结果。
- 规则：F2 + 悲观、F2 + 保守、F4、P_rule。
- 拟合量：2 种表示 × (SPARSE 1 套 + DENSE 4 套) × 3 seed = 30 个。

## 终点

只用原终点 seed：真实 LOCO 用 seed 4–7，融合用 seed 2–3。

| 编号 | 指标 | 判定方式 |
|---|---|---|
| E1 | 全部留出 controller 上的平均选择成功率，与 R0 比较 | 非劣界 −2pp。按 state 成组做 bootstrap，在本测试床的全部主要臂（含 P_rule）上做 max-T 同时 CI，要求下界 > −2pp |
| E3 | 灾难率：比源先验低 ≥ 10pp 的留出 controller 所占比例 | possible 与 certain 两种口径都报告 |
| E2 | 必要 controller 相对源先验与 R0 的收益 | 真实 Flow 上逐个报告 base_u、v10_local，并给出合并 CI |

另外逐目标报告所选 η 的直方图，重点关注 base_u、v10_local、v9_local。

### 判定定义

- **安全正迁移（SPT）：** 同时满足：
  - E1 过门；
  - E3 = 0（possible 口径）；
  - E2 合并 CI 的下界 > 0。
- **完整迁移（CT）：** SPT 成立，且 base_u 与 v10_local 的收益点估计都 > 0，即不对称被消除。
- **单元格通过：** 某个“表示 × 覆盖”单元格的判定族中，至少一个臂达到 SPT。
  - R3 单元格的判定族 = 3 条主要规则；
  - R5 单元格的判定族 = {R5-PCA2, R5-Δprog} × 3 条主要规则，共 6 个臂；
  - 判定所用的 E2 合并 CI，用该判定族上的 max-T 同时 CI。

# 四、结论映射（预注册，必须按此下结论；以真实 LOCO 为准）

R3-PCA2 × REAL6 是 round6 已知失败的参照格。表中的“R5”指 R5 判定族，报告时要写明是哪个表示、哪条规则达到了 SPT。

| 观察结果 | 结论 |
|---|---|
| R5 × REAL6 通过 | **表示是瓶颈**：在现有覆盖下，闭环行为表示即可迁移。若同时达到 CT，说明 base_u / v10_local 的不对称也已消除 |
| R5 × REAL6 不通过，R3-PCA2 × REAL6+BLEND 通过 | **覆盖是瓶颈**：静态表示在 regime exemplar 足够时即可迁移，换表示不是必需的 |
| 上两行的两个单元格都不通过，R5 × REAL6+BLEND 通过 | **两者都需要**：行为表示加上 regime 覆盖 |
| R5 × REAL6 与 R3-PCA2 × REAL6+BLEND 都通过 | 两种补救各自充分，分别报告 |
| 四格全不通过，但 I1 显示 R5 含信息（ρ 的 CI 下界 > 0），或 P_rule 达到 SPT | **信息存在，但学习通路没有用上**：在 6–18 个训练 controller 的条件下，“从表示到选择”的学习环节是瓶颈 |
| 四格全不通过，I1 无信息，P_rule 也失败 | **表示（探测设计）仍不足**：200 步、2 个 η 的探测没有编码真实 Flow 的 regime。下一步是重新设计探测（更长 horizon、与死锁相关的统计量），或接受使用目标标签（少样本） |
| 融合族与真实 Flow 结论不一致 | 分别报告并解释，例如融合族的 regime 随 λ 连续变化，更容易辨识 |

**补充规则：**
- 本轮的“覆盖”只检验了“锚点邻域增广”。若覆盖臂失败，只能否定这种增广，不能否定“增加独立的 η13 型 controller”这一假设；后者应标为 INCONCLUSIVE。
- I4 的两个描述性预测（H_static_recipe、H_behavior_regime）单独标注结论，不参与上表的判定。

# 五、硬性规则

- **不得新增 rollout 或探测，不得写入共享 DB。** 读取 DB 一律使用只读或 immutable 模式。
- **文件写入范围：** 只在新目录 `diagnostics/controller_behavior_repr_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **不得用终点 seed 做任何选择、调参或校准。** 所有超参数沿用 round6 的预注册值。本轮新增的只有：表示、覆盖因素、P_rule 和缺失格处理，这些都必须在预注册中写定。
- **证据等级：** 所有数据都已被打开过，结论一律标为 DEVELOPMENT。
- **unknown 处理：** 沿用 round6 的配对保守规则：保留 unknown，不重跑，也不算作失败。
- **资源：**
  - GPU 通过 Slurm 使用，最多 4 个 shard；CPU 最多 4 核。
  - 服务器上另有 Drone3D 任务在运行。先查看 RESOURCE_POLICY 与实时占用，不要影响其他任务。
  - 若预计墙钟时间超过 8 小时，先报告再执行，不得自行删减臂。

# 六、交付

全部交付物放在 `diagnostics/controller_behavior_repr_v1/`：

- PREREGISTRATION.md 及收据；
- 代码；
- 阶段 0 的复现结论与 R5 审计；
- 阶段 1 的 I1–I4 结果，包括 Δprog 对 ΔS 的散点图、混淆矩阵、距离表；
- 每个“臂 × 折 × seed”的 checkpoint 与预测；
- results.json；
- 图表：析因的 E1 / E2 / E3 图，以及逐目标的 η 直方图；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 必须做到以下几点：

1. 按第四节明确回答：真实 Flow 上剩下的瓶颈是表示、覆盖、两者都是，还是学习通路。
2. 把以下三件事分开报告：Q 预测、给定 controller 时的 η 选择、controller 级 regime 识别。
3. 给出下一步的最小设计，并列出所需 rollout 数：
   - 若结论是表示：写前瞻确认，需要新的 controller（例如新锚点组合的融合，或用不同 seed 重训的 Ring Flow）和新的 state；每个新 controller 另需 32 次短探测。
   - 若结论是覆盖：写构造更多 η13 型真实 controller 的方案与成本。
   - 若结论是学习通路：写结构化模型的方案，例如以探测为输入的 regime 先验。
4. 成本对照：
   - R5：每个 controller 需要 32 × 200 = 6,400 个模拟步，不需要标签。
   - 少样本校准：round5 的 k = 1 方案需要 32 个完整 H700 episode，并且需要成功标签。
   - 直接引用 round5 REPORT 中“目标少样本校准参照”表，并写明该表的比较基线。
5. 每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。事实、解释和猜想分开写，不夸大。写明结论的边界：Ring、FF、K16、固定几何，以及真实 Flow 上只有 2 个必要 controller。

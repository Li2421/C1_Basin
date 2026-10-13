# State-conditioned η Selection：数据库优先调查报告

## 1. 问题、背景与结论范围

问题是在当前 OrthoFlow3 修正参数化和真实闭环中，**初始物理 state 是否能使 η 选择稳定优于场景级固定 η**。此前 `controller_state_geometry_transfer_v1` 的 Toy 宽度及独立因素面板多数被固定 η 饱和，不能据此推广为所有场景都无 state×η 决策价值。上游 `docs/next_experiments_state_eta_headroom_20261011.md` 曾列出13个待查数据库面板并估计57,223个 Q16 格子。本次先重做全库结构普查，再按冻结的 E0→E1/E2/E3→E4→E5 规则走完适用分支。

**核心结果：** Ring 的固定 FF、H700、16 个预先冻结 η 候选上，state-conditioned 选择确有可部署收益。一个 312 state×13 η 的数据库 E0 面板，交叉拟合 state-specific 参考比独立 seed 选出的固定 η 高 **5.79 个百分点**，state bootstrap 95% CI **[+3.35,+8.81]**；面板包含此前确认的最强通用 η。更强的历史前瞻性确认在256个独立新 Ring 初态、完整K16、Q16上，冻结的 Q(s,η) 选择器达到 **93.54%**，最佳固定 η 和 Q(η) 为 **88.33%**，配对差 **+5.21pp [2.65,7.88]**。这足以否定“当前参数化下选 η 总是纯场景级问题”的普遍说法。它不证明该选择规律能跨几何或 controller 迁移。

## 2. Implementation Summary 与审计链

先读 `AGENTS.md`、`shared_rollout_db/README.md` 和 schema，以及上游几何泛化报告、数据库索引、Ring 冻结协议、已有模型与预测缓存。预注册文件 `PREREGISTRATION.md` 在读取本任务任何新计算的 outcome 前写入；收据 UTC `2026-10-11T07:16:11.920134+00:00`、SHA256 `7c717740cc8b7ed99b145b9649328bf1984d77267aa227ae6e7af4ac92452788`。E4 的固定岭回归/校准规则在读取本次 factor outcome 前另存 `ADDENDUM_E4.md`，收据 SHA256 `c8253d325302c82d57de032a0e1e85c31d4ff60da215006428d0a18331abacf7`。代码基线 Git HEAD `a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e`；工作区原本含大量未跟踪研究资产，本任务仅在新目录写派生文件，不修改其原件。

`inventory.py` 只按 `rollout` 的有效记录数、controller/state/η UID 和 metadata 建面板，不用 outcome 选面板；`extract.py` 只读 SQLite，按同一 state、所有 η 的完整 literal seed_key 交集选16个语义一致 seed，把二值真实闭环结果与 rollout UID 存为压缩 NPZ。`e0_analyze.py` 使用偶/奇 8/8 双向交叉拟合、四个固定随机 8/8 划分和2000次物理 state 成组 bootstrap；`e0_decision.py` 额外执行预注册的“不得漏掉强固定 η”资格规则。对每个面板同时输出固定 η、state-specific 交叉拟合参考、鲁棒 7/8、完整16 seed 的 B15 覆盖及 η set-cover、all-η-fail 比例。后者参考选择器需要每个待选 state 的 discovery rollout 标签，**不是部署策略，也不是连续 η 的理论 oracle**。

`e4_regime.py` 复用上游12个物理配置、两 horizon、16 state、两个解析 controller、K4、Q2的 3,072 条轨迹；同时留出同一物理几何的两 horizon，目标 state 0–7 可作无标签几何/机制特征或1/2/4-shot校准，state 8–15只供最终评估。源配置可用所有源标签；目标测试 success 不进入预测。另做留整个 topology 的更难划分。特征来自原 `factor_data.py`：raw 几何参数，或通道宽度/机器人直径、直线时间余量、绕行比、开口数、clearance 等可计算代理；这些不是经实测的吞吐能力。`controller_e5_audit.py` 比较真正共享 state/η 的 controller 对，并普查 Q16/Q64 连续采样。原始数据库与旧报告、checkpoint 均未覆盖；新增 rollout、训练均为 **0**。纯 Q 审计按 `AGENTS.md` 不触发视频验收。

数据库 SHA256 为 `7fe981632aea449762233f21431d3ed1a9fea6866edb7fc65d69094b46d6e509`，文件 5.19 GB；当前共有2,330,826条 rollout，2,298,054条无冲突且无 numerical failure 可作二值标签。6,064条 rollout 被冲突隔离、26,713条标有 numerical failure，二者有交叠，均未默认为失败。Q16 格子重算为 **57,223**。`PANEL_INDEX.json` 保存 seed 对齐策略和每个 NPZ 哈希；`SOURCE_FILES.json` 保存17,733个所用原始来源路径/存在性/哈希。119个历史来源路径当前缺失，但三个主要代表面板引用的原始来源文件均存在；SQLite 的 UID、seed 与原始记录 hash 仍可追溯。E0 实际复用180,928条不同数据库 rollout。连同历史 E2 的65,536条和 E4 的3,072条，去重后本任务引用 **187,888条旧物理 rollout**；E0/E2交叠61,648条，故不能把三项简单相加。

## 3. E0：全库普查和选择收益

57,223格横跨92个 controller。按≥12 state、≥2共享η、controller EXACT_PROFILE、true-t0、场景一致等 metadata 条件，有106个候选结构面板；其中97个确实能逐 state 对齐16个共同 seed，9个因某些 state 不能同时对齐而单列。97个面板覆盖 Ring、Toy、DB、Four-Way、Gap1；`E0_PANEL_RESULTS.json` 逐个给出完整结果。之前的13个前缀均在 `LEGACY_13_AUDIT.json` 重新核实：7个达到本次≥12 state且 seed/metadata 可评分，几个 Ring 前缀是中途状态语义或不足12 state，DB一个先前声称的32×16面板在逐 UID Q16 交集下没有形成合格共同面板。遗漏的最大证据簇是 `field_ring_t0_source_v1_ctl_8fb9…`，之前13项没有列它。

| 同一 controller 下的代表完整面板 | 格子 | 固定 held-out | state-specific held-out | 增益及95% CI | B15可行 state 的最少 η |
|---|---:|---:|---:|---:|---:|
| Ring FF, H700, `panel_7eebcd2d390d705f391b` | 312×13×16 | 88.50% | 94.29% | +5.79pp [3.35,8.81] | 4（精确整数规划） |
| Ring FF, H700, `panel_564ea45fb87a525999d0` | 141×16×16 | 90.07% | 95.39% | +5.32pp [1.73,9.49] | 4 |
| Toy TT, H850, `panel_7d79df2f3f3622b69c7b` | 24×16×16 | 84.11% | 98.44% | +14.32pp [1.82,26.31] | 2 |
| Toy FF, H850, `panel_300b92169df72e037494` | 23×16×16 | 100% | 100% | 0 | 1 |
| DB_4A, `panel_7a167c3a1d1c9a6e7344` | 48×13×16 | 100% | 100% | 0 | 1 |

Ring 312×13 面板的固定失败 `U=0.1150`、全部η失败 `I=0`，因此尚有真实可观测选择空间；完整16 seed 上有74个 state 的唯一经验最优 η、238个并列。四个预定随机 split 的收益为+5.75至+6.53pp；鲁棒7/8代理增益+5.61pp，CI [+1.76,+11.70]。这些97个面板是大量嵌套的 state/η 子集，不能按97次独立实验解释。原始四条件数值门槛使37个面板暂通过；资格复核发现其中4个漏掉已知最强固定η（Ring3、Toy FF1），于是**主判定33个，Ring32、Toy TT1**。所有被否决数值保留在结果文件，作为探索性结果。Ring 最大共同面板含历史已知最佳固定 η13（数据库 UID `eta_9992dfff…`），因此关键正结果不是通过删去通用强 η 制造。Toy TT正结果样本仅24 state、CI宽，是次要支持；Toy FF全K16的零收益反映该控制器的候选饱和。

## 4. E1–E3：决策门、独立证据及迁移边界

**E1：不触发。** E0 已有合格 Ring面板，按预注册不得再为 Toy w0.41 K8 做精度补充；新增 rollout 0。

**E2：复用既有前瞻性确认。** `state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/` 的协议在其256个新 Ring true-t0 初态 outcome 前冻结了K16、16 future seed、模型、固定 η 和分析规则；这些 state 与其此前608个开发 state 不重合。65,536次真实 FF rollout 中有296次 numerical unknown，原实验上下界保存而非改成失败，`verification.json` 证实全部标签与共享数据库一致。8个 discovery seed 选择固定η与确认偏好，另8个 seed 评估冻结模型。Q(s,η) TRAIN512 三随机拟合的 all-state成功 **93.54%**，Q(η)/最强固定η13均 **88.33%**，配对 +5.21pp [2.65,7.88]；针对每个 bootstrap 样本重新取最强固定η的同时比较仍为+5.21pp [2.64,7.94]。三拟合各自+5.47、+5.66、+4.49pp。TRAIN32诊断模型仅85.07%，比TRAIN512低8.46pp [6.14,11.08]，说明数据覆盖量对可部署选择器重要。完整数值、冻结预测和逐seed CSV在历史 `audit_results.json`、`frozen_predictions.npz`、`seed_outcomes.csv`。**独立性界限：** 这项确认相对当年的模型开发是前瞻、独立的；其61,648条标签又被本次 E0 结构普查引用，因此不能谎称它是本次 E0 之后新采集的独立复现。

**E3：仅复用历史迁移对照，正式 MANY/FEW/EXTRAP 未执行。** E0 合格 Ring FF profile 的 Q16 标签只在同一 Ring 地图/场景 UID 上；没有与之执行语义相同的多几何 Q16面板。上游 Toy width 的 MANY−FEW Q MSE差约−0.032、CI不含0，但给定 controller 的η选择成功差为0，因固定η近乎饱和。既有四场景 LOSO 共用候选 bank 上，各 held-out scene 的 critic选择基本固定一个η，matched−mismatched无可靠收益；它混杂场景/geometry/controller，不能替代 Ring 的受控几何迁移。Ring 旧 Q512 vs Q32 是同一几何内数据量对照，不是 MANY/FEW 几何对照。若新建一个最小可比较的 Ring 几何族，4个训练+2个测试几何、每几何16新state×16η×16seed，仅一个 controller 就需至少**24,576**次新物理 rollout，两个 controller 至少49,152次，尚未计几何有效性 pilot、训练和复验；参照历史原始文件粗估单 controller 原始存储约99 MB，可靠 CPU时长需单独 benchmark。现有缓存无法替代这些新执行语义，且这不是本任务允许的“针对性最小补充验证”，故本次不启动。这是明确的资源/证据边界，不是阴性 E3 结果；详见 `E3_COVERAGE_GAP.json`。

## 5. E4：场景级 prior、机制代理和 few-shot

上游 factor物理面板的解析 controller 与 Ring固定 Flow checkpoint 不同；以下是**独立因素场景级 regime** 的次要、探索性证据，不替代 Ring 强 state 结论。12个物理配置×2 horizon，共24目标配置；每个目标配置有16 state、2 controller、4η、2seed；两 horizon 一起留出同一物理几何。12个物理几何成组配对 bootstrap 2000次。特征只看目标 state 0–7 的无标签初始状态，测试 state 8–15 未参与零标签 prior；1/2/4-shot只用前池 outcome，分别对应每目标 horizon 配置16/32/64个已有 rollout，**新增成本0**。目标最终测试没有 unknown；factor源标签有1个 numerical unknown，拟合时按缺失跳过。

| 方法 | 留物理几何成功率 | 相对 source prior | 留拓扑成功率 | 相对 source prior |
|---|---:|---:|---:|---:|
| source-only固定η | 73.96% | 基准 | 69.66% | 基准 |
| raw几何参数零目标标签 | 77.86% | +3.91pp [−1.04,9.51] | 72.79% | +3.12pp [+0.13,6.64] |
| 机制代理零目标标签 | 77.99% | +4.04pp [−0.91,9.51] | 74.22% | +4.56pp [+0.26,9.38] |
| prior-only，1目标state | 75.61% | +1.65pp [−0.28,4.26] | 72.65% | +2.99pp [1.08,5.33] |
| prior-only，2目标state | 77.03% | +3.07pp [0.12,6.86] | 74.74% | +5.08pp [2.45,8.47] |
| prior-only，4目标state | 78.39% | +4.43pp [0.41,9.41] | 77.15% | +7.49pp [4.26,11.41] |
| 机制代理+4-state offset | 78.51% | +4.55pp [0.13,9.75] | 78.29% | +8.63pp [5.27,12.50] |
| hindsight测试集最佳固定η（仅参考） | 80.21% | — | 80.21% | — |

模型固定为源 fold 标准化岭回归 L2=1，没有基于目标测试调参。零目标标签的机制代理在留拓扑上胜过 source prior，但与 raw 几何参数之差仅+1.43pp，12几何bootstrap CI [−2.73,+6.64]，且拓扑只有两类，所以**不能证明机制不变性或物理吞吐规律**。原四场景旧 `calibration_results.json` 也显示目标少样本可修正 η regime：例如 DB目标1-state prior-only闭环成功约98.8%（zero-shot critic 12.5%），Ring目标4-state prior-only约76.0%（zero-shot critic 30.0%）；该旧四场景 controller/checkpoint 与 scene 混杂，应只作校准有效性、非零标签迁移的既有证据。Toy within-scene强 state选择与这些目标场景 prior结论并不矛盾。

## 6. E5：连续 success basin 证据审计

全库有6,820个 state×controller 组至少一个Q16 η；其中45组含≥64个不同Q16 η，最多336。提高到每格≥64有效seed后，有1,635组、8,524格；**12组**含≥64个不同Q64 η，最多334：Toy 8 state、DB 4 state。Toy这8个密集 state 的 η全集交集仅15，DB四个仅58；至少12个state的完整共享Q16面板最大仅K40。可见局部连续采样和较高精度 **确实存在**，但不是支撑跨state/跨几何的稠密交叉采样。上游 `orthoflow3_continuous_basin_critic_v1/failure_diagnosis.md` 已指出 Toy历史高精度 η 多为 state-specific 自适应探针、近边界极少，连续 critic 的 simultaneous unseen-state/unseen-η 排序迁移失败。这里不从K4/K8/K16推断盆地形状、体积或连通性；连续 basin泛化状态为 **INCONCLUSIVE**，未补密网格 rollout。

## 7. 假设判定、机制解释与限制

| 假设 | 判定 | 事实边界 |
|---|---|---|
| 当前参数化中 state-conditioned η 永远不优于场景固定η | **REJECTED** | Ring FF K16 历史前瞻确认 +5.21pp，CI下界>0；Toy TT亦有较小面板支持。 |
| Ring固定FF初态分布可由已训练 Q(s,η) 实际利用 | **SUPPORTED** | 256独立新初态、三拟合、冻结checkpoint/候选、独立seed测试。 |
| 同一场景所有controller共享同一最佳η | **INCONCLUSIVE（描述性反例）** | Toy同23物理state、16η、同H/Flow下 TT/FF最佳固定η不同，交换会使TT均值降9.78pp、FF降4.08pp，但state bootstrap区间含0，不能作强总体效应。其他三配对没有最佳η切换。 |
| Ring的 state×η 排序可跨几何及controller迁移 | **INCONCLUSIVE** | 缺同语义多几何强headroom Q16标签；旧四场景LOSO失败但混杂。 |
| 机制代理比原始几何参数更能预测零标签η regime | **INCONCLUSIVE** | 两类拓扑因子面板仅小幅、未有稳定的直接优势。 |
| 连续success basin形状或泛化已被离散候选证明 | **REJECTED as claim** | 局部Q64点存在，但交叉/边界覆盖不足，连续泛化问题本身未决。 |

最有证据支持的诊断是：**state选择收益取决于场景、controller、候选池和训练state覆盖**。Ring有可学习的真实交互，TRAIN512模型成功而TRAIN32模型失败；Toy TT/FF同物理state的固定η排序也可能不同；DB、Gap1及部分Toy面板在当前候选上被固定η成功饱和。机制解释“反馈场已内置足够state适应性”不再能作为全局理论，应只在特定饱和面板作猜想。现有证据没有排除跨场景方法层面的表示、训练目标和分布外选择缺陷。Ring正结果局限于固定Flow/FF、H700、预冻结K16和其原生 global-development 初态；E4因子解析controller不是该 checkpoint 的 geometry transfer。

## 8. 复现、成本与文件

运行 Python 3.11/NumPy 2.4.6/SciPy 1.17.1 环境，仓库根目录：

```bash
C1PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.inventory
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.extract
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.e0_analyze
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.e0_decision
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.legacy13_audit
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.e3_coverage
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.e4_regime
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.controller_e5_audit
$C1PY -m diagnostics.controller_eta_selection_db_audit_v1.assemble_results
```

先核对 `preregistration_receipt.json`、`addendum_e4_receipt.json` 和数据库 hash。上述程序复算均只读共享DB及历史输入；E0 95% CI 2000次 bootstrap，E4 20个固定校准排列和2000次几何 bootstrap，RNG 20261011；历史 E2 5000次 state bootstrap、冻结模型 seed17/23/41，详见其原 `protocol.json`。`results.json` 为机器主结果，`E0_PANEL_RESULTS.json`、`E4_REGIME_RESULTS.json`、`E5_DENSE_ETA_AUDIT.json` 和 `panel_data/` 为细证据，`SOURCE_FILES.json`、`PANEL_INDEX.json` 可逐UID回溯。当前仅新建派生数据约120 MB；新 rollout及训练均0，DB未写入。正式 E3 未完成，成本和恢复入口明确保存在 `E3_COVERAGE_GAP.json`。

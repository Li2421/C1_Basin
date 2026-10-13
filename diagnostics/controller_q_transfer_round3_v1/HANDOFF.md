# Controller–Q Round 3：独立科研任务交接

本文件可脱离原对话阅读。项目根目录是 `/home/zhihan/research/Basin_C1`，本轮证据均在 `diagnostics/controller_q_transfer_round3_v1/`；机器可读主指标在 `results.json`，完整设计和统计解释在 `REPORT.md`。执行日期 2026-10-11（UTC；本地 Asia/Singapore 跨至 10-12）。**最终结论：现有数据确认 Controller 会改变闭环 Q 和离散 η 偏好，但本轮没有确认 Critic 在未见 Controller 上取得可靠的 state-conditioned η 选择收益。真实 Flow checkpoint 的正式 held-out 泛化实验因预注册门槛失败未执行，仍未下结论。**

## 核心问题与先前证据

Controller 是 Flow checkpoint、FF/OrthoFlow3 执行算子和硬安全投影组成的完整程序；η 是三维 episode 级修正参数。目标是让 Critic 学习 (Q(s,C,η)) 中 Controller 的可迁移作用，在没见过的 Controller 上选择比固定 η 更好的 η。主要终点是**独立 closed-loop rollout 成功率、配对 selection gain 和 oracle regret**；Q MSE/相关只作诊断。此前原生 Ring FF 的 Q512 在已打开的 256 state 上为 93.54%，K16 最佳固定 η13 为 88.33%（+5.21pp），但只能说明同一 Controller/geometry 下可学 state selection；相关细节见 `../ring_state_selection_transfer_v1/REPORT.md`。旧 256 state 已经使用过，本轮在其上的 P0/P1 一律标 DEVELOPMENT。

## 实际实施与 provenance

先审计 `AGENTS.md`、`docs/next_experiments_controller_q_round3_20261011.md`、共享 DB、历史 checkpoint 与 execution UID/JAX x64；兼容标签复用而不因输入表示更换重跑。P0 从历史原生 Ring v10 读取 **106,496** 条标签，比较 128 state×16η、256×8、512×4 的等预算布局和 NLL/selected-success checkpoint 规则，27 次拟合，**0 新 rollout**。P1 用 14 个真实 Ring Flow checkpoint、24 个旧开发 state、K16/Q4/H700/FF，精确复用 **1,536** 个 cache slot（1,527 valid、9 unknown），新增 **19,968** 次 rollout；其中 P1 cache slot 是 P0 所读原生开发数据的一部分，不能简单相加成独立历史标签。P1 全部请求 21,504 的 postflight 为 valid 21,232、unknown 272、missing/quarantined 0。真实 Flow 的 local/global observation chart 分别适配，所有物理、seed、评价器一致；12 个 selected manifest 与训练 manifest 字节相同，另 2 个 fallback 的实际读取字段与原 manifest 相同，保守 12-controller 敏感性分析结论不变。

P1 未通过资格门，因此原定留出真实 Flow 的 P2 **NOT EXECUTED**。按冻结的条件分支 P2′，固定原生 v10 Flow/Ring geometry/H700/K16/safety evaluator，训练 FF、TT、κ−、κ0、κ+ 五个执行算子，留出 TF、FT。每个源 Controller 有 256 个全新训练 state×随机 8η×Q4 和 32 个全新验证 state×K16×Q4；每个 held Controller 在相同的 64 个全新测试 state 上 K16×Q8，seed0–3 供固定η参照选择，seed4–7 独立评价。新请求 **67,584**，preflight 全部 genuinely missing；最终 raw/DB valid **66,747**、数值 unknown **837**、missing/quarantined **0**。22 个主 CPU worker 加 5 个按同一代码分段的辅助 worker，唯一新 rollout 恰为67,584；一致重复 journal 1,792 条去重。P2′ 仿真累计 **49.27 CPU 小时**，P1 为36.19小时；本轮总新 rollout **87,552**、累计约 **85.46 CPU 小时**。静态特征和模型作业申请全部 8 个 GPU shards；P2′ 共30个模型拟合（5表示×3/5源×3 fit seed），P3 共12个，均按 source validation NLL 选 checkpoint。最终 `P2PRIME_LABELS.npz` 保留成功/明确失败/unknown 三类，不将 unknown 默认为失败。

P2′ 首轮 merger 因设计脚本漏登 DB `experiment` 行发生外键错误；当时该实验已入库 source/rollout 均为0。登记冻结身份后只重跑单进程 merger，**未重跑 rollout**；失败日志和修复收据在 `P2PRIME_MERGE_REGISTRATION_FIX.md`。分析中还将相同 state/η 的 selector/baseline unknown 作正确配对抵消；旧的过宽界 JSON 保留为 `P2PRIME_RESULTS_UNCOUPLED_UNKNOWN.json`，修正不改变主判定，见 `P2PRIME_PAIRWISE_UNKNOWN_FIX.md`。

## 关键定量结果与判断

**P0（开发校准）**：同为2,048个训练 state–η 对，K16 最强固定 η13 的开发成功率为 **88.33%**。128×16、256×8、512×4 的16-head成功率分别为 **90.82%、91.04%、90.34%**；256×8 对固定的 +2.71pp 同时95% CI **[+1.05,+4.55]pp**。selected-success checkpoint 未可靠超过验证 NLL，故 P2′ 用 256×8/Q4/NLL。见 `P0_RESULTS.json`、`P0_DECISION_SUPPLEMENT.md`、`p0_fits/`。

**P1（真实 Flow 的 Controller–Q 关系，开发 pilot）**：14 个 Flow 中仅 **7** 个最佳固定 η 成功率在预注册 [40%,95%] 区间，低于所需8个，故 P2 gate **FAIL**。在合格7个中，用 A 的最佳固定 η 到 B 的 global 遗憾中位数 **17.19pp**，unknown 保守、两终点同时95% CI **[3.13,28.39]pp**；state 层后验 oracle cross-controller 遗憾中位数 **27.60pp**、CI **[17.71,33.33]pp**。Controller 平均 Q 的 SD **0.119**，η 排序两两 Spearman 中位数 **0.704**，60.4% 的 Controller 对有不同 hindsight 最佳离散 η。这些说明开发面板上的 Controller 作用不只是统一 Q level 平移，但 oracle 差异**不是可部署迁移收益**。`P1_RESULTS.json`、`P1_Q_DECOMPOSITION.json` 提供每 Controller 和有向配对。原 P2 另外至少需73,728个新标签（超70,000上限），且完整 R4 空间无法满足两个留出点位于训练凸包内，证据见两个 `P2_*AUDIT.json`。

**P2′（独立新 state + 未见执行算子，固定 Flow）**：源合并最佳固定、R4 最近源最佳固定、目标 seed4–7 hindsight K16 最佳固定，TF/FT **均为 η13**。其闭环成功下界分别是 **91.02% / 78.52%**；FT 固定η13 unknown 0.78pp。五源模型的 TF/FT 成功率：R0 **95.96/77.34%**、R1 one-hot（held全零）**95.70/66.28%**、R3 compact64 **92.97/80.99%**、R4_action **91.41/78.52%**、R3+R4_action **93.36/80.86%**。R3+R4 在 FT 相对源固定η13 为 **+2.34pp**，paired unknown 最坏、六机会同时CI **[+0.13,+6.01]pp**；但相对 R0 的 CI **[−2.34,+7.94]pp**，matched−mismatched CI **[−0.13,+17.21]pp**，相对全部K16固定η的同时CI **[−13.41,+4.69]pp**，故未过联合门槛。R4_action 的 FT matched−mismatched 为 **+9.90pp**、同时CI **[+4.69,+14.97]pp**，表明描述改变选择，却只与固定η13打平。没有任何表示在 TF/FT 达到预注册的可靠 state-conditioned selection transfer。完整逐fit选择、unknown上下界、配对CI、Q误差在 `P2PRIME_RESULTS.json` 与 `p2prime_fits/`。

**表示与训练 Controller 数**：R3 在 FF/TT 与 TF/FT 的同 state 上逐点完全相同，但 split-half 去噪的 Q 差平方约 **0.1276/0.1265**；同输入的 pair-mean MSE 不可约下界约 **0.0319/0.0316**，是具体的表示别名。R4_action 的投影 Jacobian 可区分这些算子。R3+R4 的 TF/FT Q MSE **0.0442/0.0627**，低于 R0 的 **0.0723/0.1401**，Controller×η 平均模式相关 **0.883**，但仍未提供选择收益。3→5源 Controller 的某些单项有正数值，10项同时CI没有一项稳健为正，不能宣称增加两个κ源已经修复泛化。见 `P2PRIME_R3_ALIASING_DIAGNOSTIC.json`、`P2PRIME_STATIC_SEPARABILITY.json`、`P2PRIME_SOURCE_COUNT_ANALYSIS.json`。

**P3（失败后探索性诊断；0 rollout）**：只用部署时可计算的 η0 投影动作进行一步解析 open-loop probe，封锁 `env.step`；新建8组静态特征、12次拟合。五源 R4_openloop 的 Controller×η 平均模式 MSE 从 R0 的 **0.0246** 降至 **0.0091**，FT 成功率却为 **76.69%**，低于固定η13的 **78.52%**；TF 为 **93.88%**。短时静态 probe 不能被称为已修复闭环选择。见 `P3_RESULTS.json`、`P3_FEATURE_RECEIPT.json`、`p3_features/`。

## 已支持、被否定、未解决

- **SUPPORTED，开发证据**：真实 Flow Controller 会改变 Q level 及 K16 η 偏好；P1 合格子集有正的跨 Controller 后验遗憾。
- **SUPPORTED，独立/静态证据**：R3 对 FF/TT 与 TF/FT 存在表示别名；R4_action 能静态区分且 FT 的 matched 描述优于错配描述。
- **REJECTED，本轮预注册门槛**：P1 有至少8个适合正式真实 Flow 选择 benchmark 的 Controller（只有7个）；P2′ 任一表示在未见 TF/FT 上取得同时超过所有规定对照的可靠 state-conditioned η selection gain。
- **INCONCLUSIVE**：真正未见 Flow checkpoint 的 Critic 选择收益、跨 geometry/跨安全投影迁移、连续 η success basin 的形状；3→5源数量的一般效应。P2′ 负结果不等于所有 Controller 表示无效，因为它只改变 placement/κ，且目标最佳固定 η13 与源 prior 相同。

事实与机制解释须分开：R3 输入别名是确定的表示缺口；R4 能区分 controller 却未带来选择收益，只说明这套局部特征+训练目标在此数据下不足以证明选择迁移。它**尚不能**区分完整闭环机制缺失、训练 Controller 覆盖、优化/目标函数和 Q4/Q8 标签精度各自的贡献。P1 的真实 Flow 排序差异使跨 Controller 机制研究仍有价值；下一阶段宜先构造不少于预设数量、具有中等成功率且完整 R4 几何可分辨的真实 Flow 面板，并精确预算 K16/Q8 标签，再用新 state 验证 selector，而不是将 P1 开发 oracle 当成成功率。

## 后续研究可直接使用的文件

- `REPORT.md`、`results.json`：逐阶段状态、配置、统计、局限与主指标；`RUN_COMMANDS.md`、`ENVIRONMENT.json`：重算入口和环境。
- `PREREGISTRATION_P0.md`、`PREREGISTRATION_P1.md`、`PREREGISTRATION_P2PRIME.md`、`PREREGISTRATION_P3.md`、补充文件与 UTC/SHA256 收据：先于结果冻结的门槛与后续透明修正。
- `P1_PROTOCOL.json`、`P1_CASES.json`、`P1_CACHE_PREFLIGHT.json`、`p1_raw/`、`P1_LABELS.npz`、`P1_RESULTS.json`：真实 Flow pilot 的身份、raw 与统计。
- `P2PRIME_PROTOCOL.json`、`P2PRIME_CASES.json`、`P2PRIME_CACHE_PREFLIGHT.json`、`p2prime/raw/`、`P2PRIME_LABELS.npz`、`P2PRIME_RESULTS.json`：held placement 的全部请求、raw、配对标签与结果；`P2PRIME_POSTFLIGHT.json` 记录0缺失/0冲突。
- `p2prime_features/`、`p3_features/`、`p2prime_fits/`：静态 R3/R4/probe、逐fit checkpoint/归一化/预测；`P2PRIME_R3_ALIASING_DIAGNOSTIC.json`、`P2PRIME_SOURCE_COUNT_ANALYSIS.json`、`P3_RESULTS.json`：机制与失败定位。
- `P2_MINIMUM_BUDGET_AUDIT.json`、`P2_HULL_FEASIBILITY_AUDIT.json`：为什么原真实 Flow P2 没有执行；`P2PRIME_MERGE_REGISTRATION_FIX.md`、`P2PRIME_PAIRWISE_UNKNOWN_FIX.md`：运行异常及不改结果的修复。
- `ARTIFACT_INDEX.json`、`FINAL_CONSISTENCY_AUDIT.json`：主要文件 SHA256/文件数及四阶段预注册哈希、raw/DB/unknown/门槛一致性审计（PASS）。

所有路径相对本目录。旧 256 state 的分析为 DEVELOPMENT；P2′ 独立测试属于固定 Flow 的 held-placement 评估；P3 是主失败后的探索性分析。切勿把三种证据级别合并，也不要由离散 K16 推断连续 success basin。

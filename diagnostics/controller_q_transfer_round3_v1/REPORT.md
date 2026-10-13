# Round 3: Controller–Q 关系与真实 Controller 迁移

> 本报告覆盖 P0、P1、条件性 P2′ 与 P3 的最终执行状态；机器可读指标、逐模型结果和精确配置见 `results.json` 及所引证据文件。

## 研究问题和证据边界

目标是学习真实执行程序 (C=(\text{Flow checkpoint},\text{FF/OrthoFlow3},\text{硬安全投影})) 对 (Q(s,C,\eta)) 及最优三维 episode 参数 \(\eta\) 的作用，并检验未见 Controller 上的闭环选择成功率。Q 回归、静态动作响应和最优 \(\eta\) 排序分别报告；Q 回归不能替代选择结果。所有新增文件位于本目录，已有诊断与原始标签保持原状；共享 rollout DB 仅按其 journal/单进程 merge 规则增加新记录。

历史证据审计见 `P2PRIME_HISTORICAL_AUDIT.json`、`P1_CHECKPOINT_COMPATIBILITY.json`、`P1_MANIFEST_CROSSCHECK.json` 和 `RUN_COMMANDS.md`。以前的 Ring Q512 及 256 个确认 state 已用于分析，因而本轮在这些 state 上的 P0/P1 结果均归为 **DEVELOPMENT**。原生 Ring v10 的既有标签按 exact execution UID 与 seed 精确复用，数值失败保留为 unknown。

## 实施摘要

先对 Ring v10 进行零新增 rollout 的 P0 标签分配和 checkpoint 规则校准；随后对 14 个历史 Ring Flow checkpoint 做执行兼容性与静态查询审计，再实施 24 state × K16 × Q4 的 P1 成对 Controller 普查。P1 的门槛决定执行留出真实 Flow 的 P2，或固定 Flow、留出 placement 的 P2′。P3 仅作为失败诊断。每阶段的预注册与 UTC/SHA256 收据均在本目录。所有静态 Controller 特征只用部署时可见的 state/geometry/controller 和不调用 `env.step` 的动作查询构造。

P1 的 14 个候选包含 10 个 global-frame 和 4 个 local-frame checkpoint；checkpoint 的观测 chart 按其保存的环境指纹适配，物理 dynamics、Ring geometry、H700、FF、K16、硬安全投影与评价器固定。12 个 checkpoint 的所选 manifest 与保存的训练 manifest 字节哈希直接一致；另 2 个在冻结时选了相同物理配置的 fallback manifest。其后只读核查找到这 2 个的原训练 manifest：与所选文件在 adapter 实际读取的 `scenario_config` 和 `environment_fingerprint` 两字段完全一致，仅 dataset `files/counts/extra_report` 不同；执行 profile 保持冻结，另做保守的 12-controller provenance 敏感性分析。证据为 `P1_MANIFEST_CROSSCHECK.json` 与 `P1_MANIFEST_EQUIVALENCE_SUPPLEMENT.json`。原生 v10 的静态首步在 \(\eta_0\) 和 \(\eta_{13}\) 与历史执行逐位一致。P1 从先前打开的开发 state 预定选 12 个 adaptive-demand 与 12 个按旧 fixed-\(\eta_{13}\) 成功率四分位分层的 state；选择使用旧数据，未按新 Flow outcome 调整。

## P0：等标签预算设计校准（已完成；DEVELOPMENT）

复用 106,496 条历史 source/validation/development rollout 标签；新增 rollout 为 0。三个布局均为 2,048 个训练 state–\(\eta\) 对，使用同一 16-head MLP，未观察的 head 不参与 loss。每布局 3 个子集 × 3 次拟合；旧 256 个开发 state 上按 seed 0–7 选所有 K16 中最佳固定 \(\eta\)，seed 8–15 评估。固定基准为 \(\eta_{13}\)，成功率 88.33%。下表为选定 checkpoint 规则 NLL 的 9 个拟合的均值及相对固定 K16 基准的同时 95% CI；这些是已打开 state 上的开发证据。

| 训练分配 | 闭环成功率 | 选择增益 | 同时 95% CI |
|---|---:|---:|---:|
| 128 state × 16 η | 90.82% | +2.49pp | [+0.62,+4.49]pp |
| 256 state × 8 η | 91.04% | +2.71pp | [+1.05,+4.55]pp |
| 512 state × 4 η | 90.34% | +2.01pp | [0,+4.16]pp |

验证集 selected-success checkpoint 相比 NLL 在 256×8 布局的差为 −0.14pp，配对 95% CI [−1.02,+0.78]pp；未达到预注册的切换门槛。故后续预定 **256 state × 8 随机 η、Q4、NLL checkpoint**。原始逐拟合结果、预测、模型与统计见 `P0_RESULTS.json`、`P0_TRAINING.json`、`p0_fits/`（其中含逐拟合 `predictions.npz`）、`P0_DECISION_SUPPLEMENT.md` 及其哈希收据。该决策只说明本开发协议下的成本折中，不能作为未见 Controller 迁移收益。

## P1：真实 Flow Controller 普查

状态：**完成，预注册资格门未通过（DEVELOPMENT）**。预注册见 `PREREGISTRATION_P1.md` 及收据。21,504 个请求中，数据库精确复用 1,527 个有效标签及 9 个 numerical-unknown；genuinely missing 的 19,968 次新 rollout 按官方 DB preflight 确认，低于 25,000 上限。24 个单 CPU worker 产出 raw 和 journal；部分 checkpoint 的单次均值达 9–10 秒，原 Slurm 2h window 的 6 个尾部 worker 到时后只补未生成 raw 的 UID，详见 `P1_RESOURCE_ADJUSTMENT.md`。单进程 merge 导入 19,968 条新标签；postflight 对冻结的 21,504 个请求确认有效 21,232、numerical-unknown 272、缺失 0、冲突隔离 0，raw 恰为 19,968。新完整 episode 的累计仿真计时约 36.19 CPU 小时、均值 6.52 秒/次；续跑时读已有 raw 的开销和调度等待不计于此。合并日志见 `P1_MERGE.log`，原始数据 `p1_raw/`，配对张量 `P1_LABELS.npz`。

按 seed0–1/2–3 双向半交叉拟合，只有 **7/14** 个 Controller 的最佳固定 η 成功率处在预注册 [40%,95%] 区间，未达到至少 8 个的门 A。门 B 在这 7 个的有向配对上确实满足：跨 Controller 使用源最佳固定 η 的 global 遗憾中位数 **17.19pp**，unknown 保守的双检验同时 95% CI **[3.13,28.39]pp**；源 state oracle 在目标上执行的 state 层遗憾中位数 **27.60pp**，相应区间 **[17.71,33.33]pp**。但两个门必须同时通过，故 **P1 gate 失败，原 P2 不启动**。这些 oracle/遗憾只说明开发样本中的 Controller–η 差异，不代表可部署 critic 已学到迁移。`P1_RESULTS.json` 保存所有 controller 行、91 个有向配对及 unknown 下界；以下表列出关键 per-controller 交叉拟合结果（state oracle 为双半选择/评价差，可能因 Q2 选择噪声为负）：

| Flow checkpoint（缩写） | 固定 η 成功率 | state oracle − 固定 | 门 A 合格 |
|---|---:|---:|:---:|
| v4 wide | 64.58% | +11.46pp | 是 |
| base u | 41.67% | +9.38pp | 是 |
| v4 | 34.38% | +8.33pp | 否 |
| v6 | 31.25% | −1.04pp | 否 |
| v3 plus w | 91.67% | +2.08pp | 是 |
| v4 targeted | 46.88% | 0 | 是 |
| v3 | 95.83% | +3.12pp | 否 |
| v2 | 37.50% | +16.67pp | 否 |
| v10 local | 82.29% | +14.58pp | 是 |
| v5 | 38.54% | +4.17pp | 否 |
| v7 local | 98.96% | 0 | 否 |
| v4 plus w | 41.67% | −1.04pp | 是 |
| v8 local | 100.00% | 0 | 否 |
| v9 local | 90.62% | +2.08pp | 是 |

只读的探索性分解显示 Controller 平均 Q 的标准差 0.119（成功率单位）、controller 两两 η 均值排序的 Spearman 中位数 0.704，60.4% 的 Controller 对有不同的 hindsight 最佳离散 η；配对 state CI 和 unknown 上下界在 `P1_Q_DECOMPOSITION.json`。只保留所选 manifest 字节哈希与训练 manifest 一致的 12 个 Controller，门 A 合格数降为 5；其 global/state 遗憾中位数均为 34.38pp、保守同时 CI 下界分别 21.88/26.04pp，故主要“真实排序差异但资格数不足”的判断在这个保守 provenance 敏感性分析中不变。独立 seed 半交叉积估得 controller×η 能量 0.0281、state×controller×η 能量 0.0183；这两个辅助估计在全 unknown 半格使用零下界，不能代替 censor-aware 的 gate。整体 Q 差异与 η 排序差异均存在，但大小与因果机制仅能限定在这个开发 state 面板。

静态 R3/R4_action 查询另存，不参与 outcome 标签生成；8 个 GPU shard 上 14×24 组均成功，`P1_FEATURE_RECEIPT.json` 显示 0 次 `env.step`、0 个查询错误。无 outcome 的静态几何审计显示两两 Controller 的平均投影 \(\eta=0\) 执行动作距离为 0.257、投影 Jacobian 距离为 0.225；t0 到固定 probe 的两者平均改变为 0.0268/0.0424（原动作单位），详见 `P1_STATIC_GEOMETRY_AUDIT.json`。这些执行响应差异不能独自推出闭环 η 排序或迁移收益。唯一执行后分析修复是将 NumPy 布尔值转为 JSON 原生布尔值；初次分析已写出 `P1_LABELS.npz` 但在 JSON 序列化处报错，重跑只读统计后成功，未修改标签、公式或门槛。

## P2 或 P2′：留出 Controller 泛化

状态：**原 P2 未执行（P1 gate failed，且独立存在预算/凸包不可行性）；P2′ 已完成，但未确认可靠 state-conditioned selection transfer。** P2 原方案在硬预算与严格 R4 凸包条件下已有执行可行性缺口：最少 7 个 source Flow 且精确复用原生 v10 的条件下，新训练/测试标签至少 73,728 次，超过 70,000 上限且未计 validation；14 个 Flow 的单 state R4 投影均无法由其他 13 个的凸包表示，故“至少 2 个留出点在完整 R4 凸包内”的原条件不能满足。详见 `P2_MINIMUM_BUDGET_AUDIT.json`、`P2_HULL_FEASIBILITY_AUDIT.json`。不得将降维后的凸包位置冒充原定条件，也不得超预算提交 P2。P1 gate 失败后按已预注册的 placement/κ P2′ 执行：源 FF/TT/κ、留出 TF/FT，67,584 个新请求且完整 K16 测试，详见 `PREREGISTRATION_P2PRIME.md` 与补充收据。352 个全新 state、controller profile、execution identity 和 η mask 已冻结于 `P2PRIME_PROTOCOL.json`、`P2PRIME_TRAIN_ETA_MASK.json`；官方 preflight 确认 genuinely missing 67,584、ambiguous 0、compatible reuse 0，`P2PRIME_STATIC_AUDIT.json` 在 8 GPU shard 上无 `env.step` 且通过。闭环任务 19320 使用 22 个单 CPU worker；同时的静态特征任务 19343 使用剩余 2 CPU 与 8/8 GPU shard，8 组（7 个新 Controller 加历史 FF512）全部成功，零 `env.step`、零 outcome 读取与零查询错误，约 256 秒，见 `P2PRIME_FEATURE_RECEIPT.json`。静态查询结束后，新增辅助数组 19345 使用空出的 2 CPU 核按同一冻结执行代码计算原慢分片的远端尾部 UID；其调度冻结、哈希和 merger 双数组成功依赖见 `P2PRIME_TAIL_ASSIST_SUPPLEMENT.md`、收据和 `RUN_COMMANDS.md`，不改变 67,584 个唯一请求。五种表示的预定输入形状均已做无标签检查：source `(1952,343)`，held/mismatched `(2,64,343)`。无 outcome 的同 state 表示核查显示：FF 对 TT、TF 对 FT 的 R3 均**逐点完全相同**，所以这个静态 descriptor 对这两个 placement 对不可辨；η=0 投影动作距离仅约 0.000087/0.000063，但投影后 Jacobian 距离约 0.958/0.955，R4 能静态区分其修正方向。见 `P2PRIME_STATIC_SEPARABILITY.json`。这只说明表示内容，不等于闭环选择有效。新标签已由单进程 merger 精确入库并通过 postflight；P2′ 主结果如下。

P2′ 的 **67,584 个唯一新 rollout** 均已完成并按 frozen UID 入库：valid **66,747**、numerical unknown **837**（1.24%）、missing **0**、quarantined **0**；1,792 条由辅助 worker 与原 worker 提交的**一致重复 journal**经单进程 merger 去重，未增加唯一 rollout。27 个 worker 收据的真实仿真累计 **49.27 CPU 小时**，主 22 worker 生成 65,792 条、5 个辅助 worker 生成 1,792 条。两个补充调度的精确 UID 区间、SHA 和原因见 `P2PRIME_TAIL_ASSIST_SUPPLEMENT.md`、`P2PRIME_SECOND_ASSIST_SUPPLEMENT.md` 及收据。首次 merger 因设计脚本漏登 P2′ 独立 `experiment` 行发生外键错误，失败前 DB 中该实验的 source/rollout 数均为 0；补登冻结身份后同一 raw/journal 成功合并，**无任何重跑**。失败日志、修复脚本、哈希及新 job 见 `P2PRIME_MERGE_REGISTRATION_FIX.md` 和 `P2PRIME_MERGE_REGISTRATION_RECEIPT.json`。随后在 8 个 GPU shard 配额上完成 5 种表示 × {3,5} 个源 Controller × 3 fit seed 的 **30 次拟合**，拟合耗时约 73 秒；source-only 验证 NLL 选 checkpoint，held 标签未进入训练、归一化或选步。

两个 held placement 的源合并最佳固定 η、R4 最近源最佳固定 η，以及独立评价 seed4–7 的 hindsight K16 最佳固定 η **均为 η13**。源固定 η13 的保守成功下界在 TF 为 **91.02%**、FT 为 **78.52%**（FT 的 unknown 为 0.78pp）。目标 seed0–3 选择的固定 η 分别是 TF 的 η4 和 FT 的 η13；TF 的 η4 在独立评价上为 89.45%，说明 Q4 目标校准也有选择噪声。目标 hindsight η13 仅用于评估，绝未反馈给模型。下表为 5 源模型在独立 64 state × K16 × Q8 测试上的下界成功率；最后一列是相对**全部 K16 固定 η**的配对 unknown 最坏界、max-over-K16 同时 95% CI，单位 pp。它比只与源固定 η13 比较更严格。

| 表示 | TF 成功 | TF−η13 | TF K16 同时 CI | FT 成功 | FT−η13 | FT K16 同时 CI |
|---|---:|---:|---:|---:|---:|---:|
| R0 无 Controller | 95.96% | +4.95pp | [−0.65,+11.74] | 77.34% | −1.17pp | [−17.45,+2.73] |
| R1 one-hot，held 全零 | 95.70% | +4.69pp | [−1.06,+11.48] | 66.28% | −12.24pp | [−30.60,−4.56] |
| R3 compact64 | 92.97% | +1.95pp | [−1.04,+5.47] | 80.99% | +2.47pp | [−13.67,+6.66] |
| R4_action | 91.41% | +0.39pp | [−2.73,+1.17] | 78.52% | 0 | [−16.47,0] |
| R3+R4_action | 93.36% | +2.34pp | [−3.12,+7.70] | 80.86% | +2.34pp | [−13.41,+4.69] |

**预注册判定：无可靠 state-conditioned selection transfer。** R3+R4_action 在 FT 相对**源合并固定 η13**的 +2.34pp 具有 unknown 最坏界、六机会 Bonferroni 同时 CI **[+0.13,+6.01]pp**；但相对 R0 的区间为 [−2.34,+7.94]pp，matched−mismatched 的区间为 [−0.13,+17.21]pp，且相对全部 K16 固定 η 的区间下界为 −13.41pp，因而没有通过联合门槛。R4_action 在 FT 的 matched−mismatched 为 **+9.90pp**、同时 CI **[+4.69,+14.97]pp**，证明正确 Controller 描述会改变这套选择器的行为；但它在 FT 的成功率恰为源固定 η13 的 78.52%，不能把描述敏感性称为选择收益。R0 在 TF 的普通 95% CI 相对 η13 为 [+0.26,+10.55]pp，但 K16 同时 CI 跨 0，也不能声称可靠超过最强固定候选。全部逐表示/held 的 success、regret、matched/mismatched、unknown 上下界、5,000 次 state 成组 bootstrap 与 Q 指标见 `P2PRIME_RESULTS.json`，逐 fit 预测和 checkpoint 见 `p2prime_fits/`。

Q 辅助指标与选择终点明显分离。五源的 R3+R4_action 在 TF/FT 的 Q MSE 为 **0.0442/0.0627**，低于 R0 的 **0.0723/0.1401**；其 TF−FT 的 Controller×η 平均模式相关为 **0.883**，仍未产生可靠选择收益。静态 R3 在 FF/TT 与 TF/FT 对上完全相同，而独立 seed 半交叉估得两对的真实 Q 差平方约 **0.1276/0.1265**，相同输入的 pair-mean MSE 不可约下界约 **0.0319/0.0316**；见 `P2PRIME_R3_ALIASING_DIAGNOSTIC.json`。这是 R3 的**表示别名**证据，并非 R4 足以恢复闭环机制的证明。五源比三源在 R3 的 FT 上名义增加 +2.73pp，但 10 项 source-count 比较的 unknown 最坏同时 CI 为 **[−0.65,+6.51]pp**；其余表示也无一致正趋势，见 `P2PRIME_SOURCE_COUNT_ANALYSIS.json`。

数值 unknown 始终与失败区分。首次分析的独立上下界没有对“selector 与 baseline 在同 state/η 上使用同一 rollout”做配对抵消，虽保守却过宽；旧结果完整保留为 `P2PRIME_RESULTS_UNCOUPLED_UNKNOWN.json`。只读修正使用同样 5,000 个抽样和原统计门槛，修正后的 `P2PRIME_RESULTS.json` **不改变门槛结论**；详见 `P2PRIME_UNKNOWN_HANDLING_SUPPLEMENT.md`、`P2PRIME_PAIRWISE_UNKNOWN_FIX.md` 和哈希收据。这个修正未重训、未生成新标签。

## P3：失败诊断

状态：**已完成，事后触发的探索性诊断；新增 rollout 0。** P2′ 主门槛失败后，按 `PREREGISTRATION_P3.md` 在部署时可得信息上计算 η=0 投影动作的一步解析运动 `p+dt·a0`，随后静态查询投影动作与 Jacobian；全程封锁 `env.step`。对 7 个新 Controller 和历史 FF512 共 8 组静态特征无查询错误，约 50.5 秒；另用 8 个 GPU shard 配额完成 R4_t0/R4_openloop × {3,5} 源 × 3 fit seed 的 **12 次拟合**，约 42.2 秒。特征收据、逐模型预测、结果分别见 `P3_FEATURE_RECEIPT.json`、`p3_features/`、`p2prime_fits/C*_R4_*/` 与 `P3_RESULTS.json`。

五源时，R0 的 TF/FT 闭环成功率为 95.96/77.34%；R4_t0 为 **93.49/78.39%**，R4_openloop 为 **93.88/76.69%**。相对源固定 η13，open-loop probe 在 TF 为 +2.86pp、FT 为 −1.82pp，普通配对 95% CI 分别 [−1.43,+8.20]pp、[−4.82,+0.78]pp；没有选择修复。相比 R0，R4_openloop 的 Controller×η 平均模式 MSE 从 **0.0246** 降至 **0.0091**，但 FT 成功率反而低于固定 η13 的 78.52%。三源与五源、t0 与 open-loop probe 的 Q level/η 模式/三方残差分解见 `P3_RESULTS.json`；这些 Q 分解在同一已用于触发 P3 的 held test 上，属于失败定位而非新的独立确认。数据不支持“仅需短时开环静态探测即可修复选择”的说法，也不足以区分优化问题、训练 Controller 覆盖不足和完整闭环表示缺口。

## Controller–Q Relationship and Transferability

**Controller 对 Q 的影响。** P1 在固定 Ring state、几何、η、horizon、FF 和评价器时改变真实 Flow checkpoint，发现 Controller 平均 Q 的标准差为 0.119。独立 seed 半交叉分解得到非零 controller×η 与 state×controller×η 能量；这为“Controller 不仅造成 Q 整体偏移”提供开发样本中的证据，但数值大小仍受 state 分布与 Q4 精度限制，尚非独立确认。

**Controller 对 η 排序与 success basin 的影响。** P1 的 cross-controller 固定 η 和 state oracle 遗憾说明离散 K16 上的偏好会随 Controller 变化；两者都是后验选择的诊断，不是未见 Controller 可部署选择器的成绩。无法由 K16 离散点恢复连续 η success basin 的形状。

**表示充分性与机制证据。** R3 对 FF/TT 和 TF/FT 的静态输入逐点相同，而 paired Q 确实不同；因此 R3 在这两个 placement 对上有可量化的表示别名。R4_action 的投影 Jacobian 能区别 placement，且 FT 的 matched 描述比错配描述提高 9.90pp，但 R4_action 最终仍只与固定 η13 打平。R4_t0/open-loop probe 改进了部分 Q 模式指标，却未修复 FT 的选择。局部响应对完整 FF 积分、投影非线性及未来状态演化没有精确等价性；现有实验还不能将失败单因归于表示、优化或训练 Controller 覆盖。

**Held-out Controller 的预测与选择。** 原 P2 的真实 Flow 留出实验按预注册门槛未执行。P2′ 的 TF/FT 是**新 state + 未见 placement**、但**相同 Flow checkpoint/geometry**的独立评估。R3+R4_action 在 FT 相对源固定 η13 有小幅配对改善，却未同时超过 R0、错配表示或目标 K16 hindsight 固定；R4_action 对描述错配敏感，却不提高固定 η13 成功率。真实 Flow 的跨 checkpoint selector 泛化因此仍 **INCONCLUSIVE**，不能被 P2′ 的负结果直接否定或验证。三源到五源无同时 CI 下的稳定选择改善，表明当前增加两个 κ 源不能保证迁移。

## 假设评价、局限与复现

**SUPPORTED（DEVELOPMENT）**：P0 的 256 state × 8 η 标签分配在同等预算下比 512×4 与 128×16 的平均选择率略高；NLL checkpoint 没有被验证集 selected-success 规则可靠超越。**SUPPORTED（DEVELOPMENT）**：P1 的真实 Flow Controller 会改变 Q 水平和离散 η 偏好，且合格子集上 cross-controller regret 为正。**REJECTED（资格门）**：14 个真实 Flow 中至少 8 个具有 40%–95% 成功率；实际只有 7 个。**SUPPORTED（静态/独立标签）**：R3 对 FF/TT、TF/FT 表示别名；R4_action 可静态区分，并使 FT 的 matched/错配选择有差异。**REJECTED（预注册主终点，在本 P2′ 配置下）**：任一可用表示对未见 TF/FT 的 state-conditioned η selection 同时可靠超过源固定、最近源固定、R0、错配描述及全部 K16 固定 η；六个机会均未达到联合门槛。**INCONCLUSIVE**：真实 Flow checkpoint 上的 held-out selector 泛化、增加 3→5 源 Controller 是否一般提高迁移、连续 success basin 形状和跨几何机制不变性。

限制：P0/P1 使用已经打开的 256 开发 state，不能当作新确认；P1 的 Q4 和特定 24 state 对部分 Flow 饱和，缺乏正式 P2 资格。P2′ 虽用全新 state 和独立 seed，只有 Ring、固定原生 v10 Flow、源 FF/TT/κ、held TF/FT、K16、H700、Q8；不涵盖未见真实 Flow、其他安全投影、任意 η 连续值或不同几何。新测试 state 上的 4 个 seed 使 K16 hindsight 固定 η 不确定，已通过同时 CI 而非单次 hindsight 点估计判定。P3 是主失败后同一 held 面板的诊断，不能升级成第二次独立确认。数值 unknown 837 条保留并使用配对最坏界，结果不等于完整成功/失败标签下的精确 Q。

复现从 `RUN_COMMANDS.md`、`PREREGISTRATION*.md` 及 UTC/SHA256 收据开始；精确请求与数据来源是 `P1_CASES.json`、`P2PRIME_CASES.json`、`P1_CACHE_PREFLIGHT.json`、`P2PRIME_CACHE_PREFLIGHT.json`，原始结果为 `p1_raw/`、`p2prime/raw/`、`P1_LABELS.npz`、`P2PRIME_LABELS.npz`。逐拟合 checkpoint、验证 NLL 历史和预测在 `p0_fits/`、`p2prime_fits/`；统计重算入口为 `p0_analyze.py`、`p1_analyze.py`、`p2prime_analyze.py`、`p2prime_source_count_analysis.py`、`p3_analyze.py`。环境和源码哈希见 `ENVIRONMENT.json`、预注册收据及最终 `ARTIFACT_INDEX.json`；`FINAL_CONSISTENCY_AUDIT.json` 对四份预注册哈希、raw/DB计数、unknown、阶段门和报告哈希给出 **PASS**。本研究属于纯 Q 实验，按 `AGENTS.md` 无视频验收；未改动任何历史原始资产或 Git 跟踪文件，未做 commit。

# Controller-conditioned η selection：独立研究交接

## 核心问题与背景

在未见过的真实 Ring Flow controller 上，模型是否能只用部署时可算的初始state和controller静态查询，为每个state选η，并在**确实需要controller信息的折**上可靠超过源固定η、最近源固定η和无controller输入的critic？上一轮P1已证实Flow改变K16的偏好，但P2′留出TF/FT与源先验同为η13，无法检验controller表示。详见 `REPORT.md` 与 `../controller_q_transfer_round3_v1/HANDOFF.md`。

## 实际实施

七个P1合格Flow做LOCO，固定Ring几何、FF、H700、K16、安全投影、评价器和state。全新128训练+64测试自然state通过历史hash去重；训练每state8个η×Q4，测试K16×Q8。已有P1的24验证state标签被复用；Stage A新增86,016个raw（883个unknown），全阶段新增122,880个，累计原始episode约209.32 CPU小时。Stage A 35组模型任务、105个fit seed checkpoint。先用test seed0–3做Controller必要性门，再仅用seed4–7作主终点；按物理state配对max-T同时置信区间。分支决定：STAGE B THEN STAGE C (prospective addendum)。执行身份、哈希、preflight、merge和运行命令见 `REPORT.md` B/G节。

## 最重要的结果

必要折 **2/7**：[1, 4]。Stage A过主门表示：无。Stage C独立确认：无 Stage B 13源开发面板过门：R3, R3_R4_action 最终分类：**(ii) 必要折上未得到独立确认；Stage B 开发集有正信号**。

| 折 | 留出真实 Flow | η_src index | train最佳固定η | 目标固定 η（交叉拟合） | L 下界 | 95% CI | 必要折 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | base_u_plus_w_v4_wide_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 1 | base_u_macflow | 10 | 13 | 13/13 | +28.52pp | [+17.19pp, +39.06pp] | 是 |
| 2 | base_u_plus_w_v3_macflow | 10 | 2 | 10/2 | -1.95pp | [-5.86pp, +1.95pp] | 否 |
| 3 | base_u_plus_w_v4_train_targeted_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 4 | base_u_v10_local_macflow | 10 | 13 | 13/13 | +34.38pp | [+24.60pp, +44.92pp] | 是 |
| 5 | base_u_plus_w_v4_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 6 | base_u_v9_local_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |

| 表示 | 配对差 | 均值 | 95% 同时 CI |
| --- | --- | --- | --- |
| R3 | src | +5.73pp | [-0.58pp, +12.04pp] |
| R3 | near | +5.73pp | [-0.58pp, +12.04pp] |
| R3 | R0 | +4.43pp | [-1.44pp, +10.30pp] |
| R3 | mismatch | +6.22pp | [+2.26pp, +10.18pp] |
| R4_action | src | +0.72pp | [-4.10pp, +5.53pp] |
| R4_action | near | +0.72pp | [-4.10pp, +5.53pp] |
| R4_action | R0 | -0.59pp | [-4.82pp, +3.65pp] |
| R4_action | mismatch | -0.37pp | [-3.16pp, +2.42pp] |
| R4_openloop | src | +0.39pp | [-3.35pp, +4.13pp] |
| R4_openloop | near | +0.39pp | [-3.35pp, +4.13pp] |
| R4_openloop | R0 | -0.91pp | [-4.19pp, +2.36pp] |
| R4_openloop | mismatch | -1.15pp | [-4.19pp, +1.89pp] |
| R3_R4_action | src | +7.49pp | [+0.86pp, +14.12pp] |
| R3_R4_action | near | +7.49pp | [+0.86pp, +14.12pp] |
| R3_R4_action | R0 | +6.18pp | [-0.08pp, +12.45pp] |
| R3_R4_action | mismatch | +7.77pp | [+3.47pp, +12.07pp] |

Stage B/C的量化结论与controller-prior分解：

Stage B（13源，DEVELOPMENT）：

| 表示 | 对照 | 差值 | 95%同时CI |
| --- | --- | --- | --- |
| R3 | src | +6.64pp | [+2.57pp, +10.71pp] |
| R3 | near | +6.64pp | [+2.57pp, +10.71pp] |
| R3 | R0 | +7.36pp | [+3.14pp, +11.57pp] |
| R3 | mismatch | +9.54pp | [+5.71pp, +13.36pp] |
| R3_R4_action | src | +6.97pp | [+2.43pp, +11.50pp] |
| R3_R4_action | near | +6.97pp | [+2.43pp, +11.50pp] |
| R3_R4_action | R0 | +7.68pp | [+3.05pp, +12.31pp] |
| R3_R4_action | mismatch | +9.52pp | [+5.67pp, +13.36pp] |

增加源后的R3/组合表示相对Stage A总体成功率无显著改善，R0反而下降；不能据过门认定controller覆盖是唯一或主要根因。

Stage C（冻结Stage B模型，32个全新物理state、两个controller、64个面板）：

| 表示 | 对照 | 差值 | 95%同时CI |
| --- | --- | --- | --- |
| R3 | src | +5.21pp | [-0.66pp, +11.08pp] |
| R3 | near | +5.21pp | [-0.66pp, +11.08pp] |
| R3 | R0 | +5.73pp | [-0.13pp, +11.59pp] |
| R3 | mismatch | +7.51pp | [+2.53pp, +12.49pp] |
| R3_R4_action | src | +3.78pp | [-2.40pp, +9.96pp] |
| R3_R4_action | near | +3.78pp | [-2.40pp, +9.96pp] |
| R3_R4_action | R0 | +4.30pp | [-1.75pp, +10.34pp] |
| R3_R4_action | mismatch | +5.38pp | [+0.78pp, +9.99pp] |

**stage_a（EXPLORATORY）**

| 表示 | 逐state选择−学习到的controller固定η | 95%同时CI |
| --- | --- | --- |
| R3 | -3.84pp | [-6.84pp, -0.85pp] |
| R4_action | +0.72pp | [-3.71pp, +5.14pp] |
| R4_openloop | +0.39pp | [-3.10pp, +3.88pp] |
| R3_R4_action | +2.21pp | [-0.01pp, +4.44pp] |

**stage_b（EXPLORATORY）**

| 表示 | 逐state选择−学习到的controller固定η | 95%同时CI |
| --- | --- | --- |
| R3 | -2.08pp | [-4.28pp, +0.12pp] |
| R4_action | +0.07pp | [-5.04pp, +5.17pp] |
| R4_openloop | -4.49pp | [-8.96pp, -0.03pp] |
| R3_R4_action | -1.76pp | [-5.14pp, +1.63pp] |

**stage_c（EXPLORATORY）**

| 表示 | 逐state选择−学习到的controller固定η | 95%同时CI |
| --- | --- | --- |
| R3 | -1.30pp | [-4.56pp, +1.95pp] |
| R3_R4_action | -2.73pp | [-6.70pp, +1.23pp] |

## 假设与根因边界

INCONCLUSIVE：Stage B 的13源设置在已打开面板上过门，但冻结模型未在新state同时超过全部对照；不能宣称稳健的跨Controller选择迁移。 Q水平、η排序和selection是不同命题。Q分解与排序相关见 `CONTROLLER_Q_DECOMPOSITION.json`，逐表示闭环成功率、配对差、Q MSE、少样本目标校准见 `results.json`。若模型失效，可支持的陈述仅限本配置下的静态表示或源Controller覆盖；不能把失败直接归因于网络结构，也不能把成功直接推广到跨几何/连续success basin。Stage B若执行，仅为同一已打开目标面板的开发证据；Stage C才是新的自然state确认。

## 后续任务入口

独立研究Agent应先读 `REPORT.md`、`PREREGISTRATION_STAGE_A.md`、`PREREGISTRATION_CONDITIONAL_BRANCHES.md`、`results.json`及 `FINAL_CONSISTENCY_AUDIT.json`。可直接复用 `protocol.json`、`LABELS.npz`、`fits/`逐seed checkpoint/predictions、`static_features/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、条件分支目录和 `RUN_COMMANDS.md`。下一步最有价值的问题是：在固定闭环语义下，哪些controller静态或短时响应真正决定η排序；若七个Flow的必要折太少，需先设计能独立改变η优选的真实controller样本，再验证跨geometry。连续success basin需要额外密集η证据，本研究不作形状结论。

补充入口：`PREREGISTRATION_STAGE_C_AFTER_B.md`说明独立确认分支修订；`STAGE_A_RESULTS_FROZEN.json`、`stage_b/results.json`、`stage_c/results.json`是原始分阶段统计，最终 `results.json.study_summary` 为完整聚合。`controller_prior/`区分无目标标签的controller η regime识别与逐state选择；`TRAINING_PROVENANCE_AUDIT.json`记录源-only归一化和135个checkpoint，`RAW_CONTENT_INDEX_STAGE_*.jsonl`逐episode内容哈希。最重要的未解问题是controller表示是否仅识别全局η regime、哪些state×controller交互能迁移；增加controller覆盖的因果改善并未由Stage B−A比较证实。

已核查的关键覆盖缺口：新增7个源在训练标签中的经验最佳η为10/2/4，没有增加η13经验最优的controller。因此每个必要折6→13源后，η13最佳源例数仍只有1，见 `SOURCE_REGIME_COVERAGE.json`。下一任务应先复用现有标签/预测检验η-regime覆盖、Q拟合与排序目标、以及state残差选择；不要把controller数量增加直接等同于机制覆盖。独立确认的R3相对源固定η为+5.21pp但CI[-0.66,+11.08]pp，相对R0为+5.73pp但CI[-0.13,+11.59]pp；正确描述优于错配+7.51pp、CI[+2.53,+12.49]pp。这支持controller信息影响决策，但未达到可靠迁移主门，亦未证明真实效应为零。逐state选择相对无标签controller-prior也未得到独立正收益。

Q预测单独判断：必要折平均MSE从Stage B的R0 0.1046降至R3/组合0.0762，这是辅助拟合收益；其与选择确认失败同时成立。目标标签选出的固定η13在Stage C两个折分别64.84%/89.06%，明显高于零样本R3的31.25%/69.01%，但它使用了目标标签。1/2/4-state calibration只做固定预选子集的描述性对照，不能声称普遍样本效率。复用数据为21,504个P1 seed格（21,232有效、272 unknown）；全任务122,880新episode，1,307 unknown均保留，0 missing/0 quarantine。

证据开封边界：本轮Stage A的64个及Stage C的32个已评估物理state现已打开。下一任务若据此选模型/表示/超参，其结果只能作为DEVELOPMENT。Stage C协议中其余32个初态尚无rollout标签（-2），但已存在静态特征和冻结模型预测；使用它们补证必须另写事前协议和预算，不能覆盖本轮证据。本轮停止新增数据。

注意数字巧合：本轮Stage C的R3 +5.21pp是相对两个留出controller上的源固定η10；它与历史Ring Q512相对本controller最佳η13的+5.21pp属于不同数据、controller与baseline，不能互相替代。

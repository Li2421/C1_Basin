# Controller-conditioned η selection：独立研究交接

## 核心问题与背景

在未见过的真实 Ring Flow controller 上，模型是否能只用部署时可算的初始state和controller静态查询，为每个state选η，并在**确实需要controller信息的折**上可靠超过源固定η、最近源固定η和无controller输入的critic？上一轮P1已证实Flow改变K16的偏好，但P2′留出TF/FT与源先验同为η13，无法检验controller表示。详见 `REPORT.md` 与 `../controller_q_transfer_round3_v1/HANDOFF.md`。

## 实际实施

七个P1合格Flow做LOCO，固定Ring几何、FF、H700、K16、安全投影、评价器和state。全新128训练+64测试自然state通过历史hash去重；训练每state8个η×Q4，测试K16×Q8。已有P1的24验证state标签被复用；Stage A新增86,016个raw（883个unknown），全阶段新增114,688个，累计原始episode约196.70 CPU小时。Stage A 35组模型任务、105个fit seed checkpoint。先用test seed0–3做Controller必要性门，再仅用seed4–7作主终点；按物理state配对max-T同时置信区间。分支决定：STAGE B: EXTRA SOURCE CONTROLLERS。执行身份、哈希、preflight、merge和运行命令见 `REPORT.md` B/G节。

## 最重要的结果

必要折 **2/7**：[1, 4]。Stage A过主门表示：无。Stage C未执行。 Stage B 13源开发面板过门：R3, R3_R4_action 最终分类：**(iii) 增加源 Controller 后有开发集收益**。

| 折 | 留出真实 Flow | η_src index | 目标固定 η（交叉拟合） | L 下界 | 95% CI | 必要折 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | base_u_plus_w_v4_wide_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 1 | base_u_macflow | 10 | 13/13 | +28.52pp | [+17.19pp, +39.06pp] | 是 |
| 2 | base_u_plus_w_v3_macflow | 10 | 10/2 | -1.95pp | [-5.86pp, +1.95pp] | 否 |
| 3 | base_u_plus_w_v4_train_targeted_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 4 | base_u_v10_local_macflow | 10 | 13/13 | +34.38pp | [+24.60pp, +44.92pp] | 是 |
| 5 | base_u_plus_w_v4_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 6 | base_u_v9_local_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |

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

## 假设与根因边界

PARTIALLY SUPPORTED：Stage B 的13源设置通过同一已打开测试面板的联合门，但未有独立确认；不能把覆盖增加解释为因果修复。 Q水平、η排序和selection是不同命题。Q分解与排序相关见 `CONTROLLER_Q_DECOMPOSITION.json`，逐表示闭环成功率、配对差、Q MSE、少样本目标校准见 `results.json`。若模型失效，可支持的陈述仅限本配置下的静态表示或源Controller覆盖；不能把失败直接归因于网络结构，也不能把成功直接推广到跨几何/连续success basin。Stage B若执行，仅为同一已打开目标面板的开发证据；Stage C才是新的自然state确认。

## 后续任务入口

独立研究Agent应先读 `REPORT.md`、`PREREGISTRATION_STAGE_A.md`、`PREREGISTRATION_CONDITIONAL_BRANCHES.md`、`results.json`及 `FINAL_CONSISTENCY_AUDIT.json`。可直接复用 `protocol.json`、`LABELS.npz`、`fits/`逐seed checkpoint/predictions、`static_features/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、条件分支目录和 `RUN_COMMANDS.md`。下一步最有价值的问题是：在固定闭环语义下，哪些controller静态或短时响应真正决定η排序；若七个Flow的必要折太少，需先设计能独立改变η优选的真实controller样本，再验证跨geometry。连续success basin需要额外密集η证据，本研究不作形状结论。

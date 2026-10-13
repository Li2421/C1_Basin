# Real Flow LOCO：Controller-conditioned η selection 最终报告

## A. Research Overview

主问题：在源合并固定 η 对目标真实 Flow 明显不合适的留出 controller 上，部署时可静态计算的 controller 表示能否提高每个 state 的 η 选择闭环成功率？本研究把 Q 预测与 η 选择分开，主终点为独立 seed 的闭环成功率和预注册同时置信区间。上一轮 P1 在24个开发 state 上测到合格 Flow 的全局跨 controller η 遗憾中位数17.19pp （同时 CI [3.13,28.39]pp），但该面板已经打开，不能当作本轮新确认。

本轮分类：**(iii) 增加源 Controller 后有开发集收益**。PARTIALLY SUPPORTED：Stage B 的13源设置通过同一已打开测试面板的联合门，但未有独立确认；不能把覆盖增加解释为因果修复。

## B. Implementation Summary

先审计 `AGENTS.md`、Round 3 原始协议、14 个 Ring Flow checkpoint、shared DB 和历史标签。七个 [40%,95%] Flow 的 UID、checkpoint SHA、local/global observation frame 与 JAX x64=false 见 `EXECUTION_IDENTITY_AUDIT.json`。复用 Round 3 的 adapter、原生 v10 runtime、R3/R4_action/R4_openloop 静态特征和训练设置。所有新增文件在本目录；历史数据未修改、未做 Git commit。纯 Q 实验不要求视频。

用新的自然 state 序列生成128个 S_train、64个 S_test，七个 Flow 共用；全库/历史 state hash 查重为零，见 `STATE_DISJOINT_RECEIPT.json`。每个训练 state 分配8/16个 η、Q4；test 为K16×Q8，其中seed0–3仅用于目标固定 η 参照和必要性门，seed4–7仅用于最终评价。P1 的24个开发 state 仅供源验证 NLL 选步。训练为 physical23+表示→16-head MLP，3个 fit seed；只对有标签的头计损失。R0、R3、R4_action、R4_openloop、R3+R4_action用同一模型、数据、预算；R4基于η=0投影执行动作及投影后Jacobian，静态查询禁止 `env.step`。所有新 outcome 产生前的设计、决策门和时间/SHA凭证见下列预注册文件。

| 预注册文件 | UTC 时间 | SHA256 |
| --- | --- | --- |
| PREREGISTRATION_CONDITIONAL_BRANCHES.md | 2026-10-12T03:47:51.656833+00:00 | c4a087b6657fd242107f3219550e082ee37d80da2e4032ca6d936c5e533b6c53 |
| PREREGISTRATION_RESOURCE_SUPPLEMENT.md | 2026-10-12T03:32:56.265221+00:00 | 7447d22da0c9132bdfcb66223efd9a5142d2707597acf24ee2a28ed0d82b9922 |
| PREREGISTRATION_STAGE_A.md | 2026-10-12T03:31:34.392184+00:00 | 695e015d9d0ebb799bda27700fca770b9679bfd4b0c4469b0ee40d7e67d36d53 |

86,016个 Stage A 精确请求经官方 `python -m shared_rollout_db.plan` preflight，复用0、缺失86,016；DB experiment 行在 merge 前登记。98次计时 pilot、按 timing 的无结果重排、native v10 特征等价核查分别记录于 `PILOT_SAMPLING_DEVIATION.md`、`SCHEDULING_ADJUSTMENT.md`、`NATIVE_VALIDATION_FEATURE_AUDIT.json`。新 raw/journal 经单进程 merge，再做 postflight 和UID标签张量；数值失败保留为unknown。Stage A 86,016个新 raw、85,133 valid、883 unknown、0 missing、0 quarantine。七组静态特征查询均为0次 `env.step`。35组训练任务×3 fit seed＝105个 checkpoint；source-only归一化与P1验证选步。总复用P1验证标签约21,504个已有逐seed格标签；本轮所有阶段新增114,688个 raw，原始episode时间合计约196.70 CPU小时（未包含静态查询、训练、合并和排队时间）。

## C. Experimental Results

### Stage A：必要性门和 Controller–Q 关系

必要性门只读 S_test seed0–3：目标固定η用0–1↔2–3双向交叉拟合；源η来自六个源 Flow 的 S_train；对 unknown 使用配对保守下界。64个物理state成组bootstrap 2000次，门为 L≥5pp 且95%下界>0。必要折 2/7：[1, 4]。

| 折 | 留出真实 Flow | η_src index | 目标固定 η（交叉拟合） | L 下界 | 95% CI | 必要折 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | base_u_plus_w_v4_wide_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 1 | base_u_macflow | 10 | 13/13 | +28.52pp | [+17.19pp, +39.06pp] | 是 |
| 2 | base_u_plus_w_v3_macflow | 10 | 10/2 | -1.95pp | [-5.86pp, +1.95pp] | 否 |
| 3 | base_u_plus_w_v4_train_targeted_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 4 | base_u_v10_local_macflow | 10 | 13/13 | +34.38pp | [+24.60pp, +44.92pp] | 是 |
| 5 | base_u_plus_w_v4_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 6 | base_u_v9_local_macflow | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |

下表为全局层跨 controller 遗憾（源 Controller 的训练最优固定η在目标上的损失）；它与 state 层后验 oracle 和零样本模型选择并非同一指标。

| 源 Flow | 目标 Flow | 源 η | 目标 η | 遗憾下界 |
| --- | --- | --- | --- | --- |
| base_u_macflow | base_u_plus_w_v4_wide_macflow | 13 | 10 | +37.11pp |
| base_u_plus_w_v3_macflow | base_u_plus_w_v4_wide_macflow | 2 | 10 | +41.02pp |
| base_u_plus_w_v4_train_targeted_macflow | base_u_plus_w_v4_wide_macflow | 10 | 10 | +0.00pp |
| base_u_v10_local_macflow | base_u_plus_w_v4_wide_macflow | 13 | 10 | +37.11pp |
| base_u_plus_w_v4_macflow | base_u_plus_w_v4_wide_macflow | 10 | 10 | +0.00pp |
| base_u_v9_local_macflow | base_u_plus_w_v4_wide_macflow | 10 | 10 | +0.00pp |
| base_u_plus_w_v4_wide_macflow | base_u_macflow | 10 | 13 | +29.69pp |
| base_u_plus_w_v3_macflow | base_u_macflow | 2 | 13 | +50.39pp |
| base_u_plus_w_v4_train_targeted_macflow | base_u_macflow | 10 | 13 | +29.69pp |
| base_u_v10_local_macflow | base_u_macflow | 13 | 13 | +0.00pp |
| base_u_plus_w_v4_macflow | base_u_macflow | 10 | 13 | +29.69pp |
| base_u_v9_local_macflow | base_u_macflow | 10 | 13 | +29.69pp |
| base_u_plus_w_v4_wide_macflow | base_u_plus_w_v3_macflow | 10 | 2 | -0.39pp |
| base_u_macflow | base_u_plus_w_v3_macflow | 13 | 2 | +30.86pp |
| base_u_plus_w_v4_train_targeted_macflow | base_u_plus_w_v3_macflow | 10 | 2 | -0.39pp |
| base_u_v10_local_macflow | base_u_plus_w_v3_macflow | 13 | 2 | +30.86pp |
| base_u_plus_w_v4_macflow | base_u_plus_w_v3_macflow | 10 | 2 | -0.39pp |
| base_u_v9_local_macflow | base_u_plus_w_v3_macflow | 10 | 2 | -0.39pp |
| base_u_plus_w_v4_wide_macflow | base_u_plus_w_v4_train_targeted_macflow | 10 | 10 | +0.00pp |
| base_u_macflow | base_u_plus_w_v4_train_targeted_macflow | 13 | 10 | +39.84pp |
| base_u_plus_w_v3_macflow | base_u_plus_w_v4_train_targeted_macflow | 2 | 10 | +42.97pp |
| base_u_v10_local_macflow | base_u_plus_w_v4_train_targeted_macflow | 13 | 10 | +39.84pp |
| base_u_plus_w_v4_macflow | base_u_plus_w_v4_train_targeted_macflow | 10 | 10 | +0.00pp |
| base_u_v9_local_macflow | base_u_plus_w_v4_train_targeted_macflow | 10 | 10 | +0.00pp |
| base_u_plus_w_v4_wide_macflow | base_u_v10_local_macflow | 10 | 13 | +34.38pp |
| base_u_macflow | base_u_v10_local_macflow | 13 | 13 | +0.00pp |
| base_u_plus_w_v3_macflow | base_u_v10_local_macflow | 2 | 13 | +60.55pp |
| base_u_plus_w_v4_train_targeted_macflow | base_u_v10_local_macflow | 10 | 13 | +34.38pp |
| base_u_plus_w_v4_macflow | base_u_v10_local_macflow | 10 | 13 | +34.38pp |
| base_u_v9_local_macflow | base_u_v10_local_macflow | 10 | 13 | +34.38pp |
| base_u_plus_w_v4_wide_macflow | base_u_plus_w_v4_macflow | 10 | 10 | +0.00pp |
| base_u_macflow | base_u_plus_w_v4_macflow | 13 | 10 | +39.84pp |
| base_u_plus_w_v3_macflow | base_u_plus_w_v4_macflow | 2 | 10 | +41.41pp |
| base_u_plus_w_v4_train_targeted_macflow | base_u_plus_w_v4_macflow | 10 | 10 | +0.00pp |
| base_u_v10_local_macflow | base_u_plus_w_v4_macflow | 13 | 10 | +39.84pp |
| base_u_v9_local_macflow | base_u_plus_w_v4_macflow | 10 | 10 | +0.00pp |
| base_u_plus_w_v4_wide_macflow | base_u_v9_local_macflow | 10 | 10 | +0.00pp |
| base_u_macflow | base_u_v9_local_macflow | 13 | 10 | +16.80pp |
| base_u_plus_w_v3_macflow | base_u_v9_local_macflow | 2 | 10 | +76.17pp |
| base_u_plus_w_v4_train_targeted_macflow | base_u_v9_local_macflow | 10 | 10 | +0.00pp |
| base_u_v10_local_macflow | base_u_v9_local_macflow | 13 | 10 | +16.80pp |
| base_u_plus_w_v4_macflow | base_u_v9_local_macflow | 10 | 10 | +0.00pp |

独立 evaluation seed 的完整逐折结果如下。成功率及差值采用unknown保守下界；Q MSE只是辅助指标。最近源由**源数据归一化后的 R4_action**均值距离决定。目标自身固定η使用目标 seed0–3，只作目标监督参照。

| 折 | 表示 | 必要 | 成功率下界 | vs η_src | vs 最近源 | vs R0 | matched−mismatched | oracle regret下界 | Q MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | R0 | 否 | 51.04% | -10.68pp | -10.68pp | +0.00pp | +0.00pp | +28.26pp | 0.0514 |
| 0 | R3 | 否 | 60.55% | +0.00pp | +0.00pp | +9.11pp | +16.82pp | +18.75pp | 0.0581 |
| 0 | R4_action | 否 | 50.26% | -10.81pp | -10.81pp | -1.04pp | +0.95pp | +29.04pp | 0.0647 |
| 0 | R4_openloop | 否 | 50.52% | -10.29pp | -10.29pp | -0.65pp | +4.01pp | +28.78pp | 0.0622 |
| 0 | R3_R4_action | 否 | 60.55% | +0.00pp | +0.00pp | +9.11pp | +15.62pp | +18.75pp | 0.0578 |
| 1 | R0 | 是 | 24.48% | +0.65pp | +0.65pp | +0.00pp | +0.00pp | +43.10pp | 0.0614 |
| 1 | R3 | 是 | 17.06% | -7.16pp | -7.16pp | -7.81pp | -3.17pp | +50.52pp | 0.0674 |
| 1 | R4_action | 是 | 22.79% | -1.56pp | -1.56pp | -2.21pp | -0.20pp | +44.79pp | 0.0826 |
| 1 | R4_openloop | 是 | 23.05% | -1.43pp | -1.43pp | -2.08pp | -0.61pp | +44.53pp | 0.0863 |
| 1 | R3_R4_action | 是 | 19.14% | -4.95pp | -4.95pp | -5.60pp | -0.54pp | +48.44pp | 0.0637 |
| 2 | R0 | 否 | 77.34% | -9.77pp | -9.77pp | +0.00pp | +0.00pp | +22.27pp | 0.1997 |
| 2 | R3 | 否 | 85.42% | -1.30pp | -1.30pp | +8.07pp | +12.15pp | +14.19pp | 0.2732 |
| 2 | R4_action | 否 | 76.17% | -11.07pp | -11.07pp | -1.56pp | -1.37pp | +23.44pp | 0.2073 |
| 2 | R4_openloop | 否 | 79.17% | -8.20pp | -8.20pp | +1.30pp | +1.93pp | +20.44pp | 0.1970 |
| 2 | R3_R4_action | 否 | 84.51% | -2.21pp | -2.21pp | +7.16pp | +11.48pp | +15.10pp | 0.2757 |
| 3 | R0 | 否 | 35.16% | -10.81pp | -10.81pp | +0.00pp | +0.00pp | +14.06pp | 0.0689 |
| 3 | R3 | 否 | 39.32% | -6.25pp | -6.25pp | +3.91pp | +18.36pp | +9.90pp | 0.0283 |
| 3 | R4_action | 否 | 37.11% | -8.59pp | -8.59pp | +1.82pp | +3.39pp | +12.11pp | 0.0743 |
| 3 | R4_openloop | 否 | 29.82% | -16.15pp | -16.15pp | -5.47pp | +0.74pp | +19.40pp | 0.0722 |
| 3 | R3_R4_action | 否 | 37.11% | -8.59pp | -8.59pp | +1.69pp | +15.67pp | +12.11pp | 0.0298 |
| 4 | R0 | 是 | 66.02% | +1.95pp | +1.95pp | +0.00pp | +0.00pp | +33.98pp | 0.1407 |
| 4 | R3 | 是 | 82.68% | +18.62pp | +18.62pp | +16.67pp | +15.60pp | +17.32pp | 0.1079 |
| 4 | R4_action | 是 | 67.06% | +2.99pp | +2.99pp | +1.04pp | -0.54pp | +32.94pp | 0.1425 |
| 4 | R4_openloop | 是 | 66.28% | +2.21pp | +2.21pp | +0.26pp | -1.69pp | +33.72pp | 0.1436 |
| 4 | R3_R4_action | 是 | 83.98% | +19.92pp | +19.92pp | +17.97pp | +16.08pp | +16.02pp | 0.1066 |
| 5 | R0 | 否 | 37.37% | -9.51pp | +31.90pp | +0.00pp | +0.00pp | +14.19pp | 0.0631 |
| 5 | R3 | 否 | 44.14% | -2.86pp | +38.67pp | +6.64pp | +18.29pp | +7.42pp | 0.0339 |
| 5 | R4_action | 否 | 40.76% | -6.12pp | +35.29pp | +3.39pp | +3.45pp | +10.81pp | 0.0751 |
| 5 | R4_openloop | 否 | 42.97% | -4.04pp | +37.50pp | +5.47pp | +4.19pp | +8.59pp | 0.0773 |
| 5 | R3_R4_action | 否 | 44.27% | -2.73pp | +38.80pp | +6.77pp | +17.77pp | +7.29pp | 0.0347 |
| 6 | R0 | 否 | 92.84% | -1.69pp | +14.32pp | +0.00pp | +0.00pp | +3.65pp | 0.1460 |
| 6 | R3 | 否 | 83.59% | -10.94pp | +5.08pp | -9.24pp | +3.78pp | +12.89pp | 0.1089 |
| 6 | R4_action | 否 | 89.19% | -5.34pp | +10.68pp | -3.65pp | +0.22pp | +7.29pp | 0.1446 |
| 6 | R4_openloop | 否 | 92.06% | -2.47pp | +13.54pp | -0.78pp | +2.02pp | +4.43pp | 0.1425 |
| 6 | R3_R4_action | 否 | 83.07% | -11.46pp | +4.56pp | -9.77pp | +2.30pp | +13.41pp | 0.1075 |

主检验仅汇总必要折。每个表示需同时超过源固定η、最近源固定η、R0 critic，且正确描述优于六个错配描述；16项比较用同一物理state配对max-T bootstrap 5000次控制同时区间。Stage A 通过表示：无。

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

少样本目标校准使用不同于 S_test 的前1/2/4个 S_train state，属于有目标标签的参照；不是zero-shot。`results.json`保留全部逐fit成功率、unknown上界、oracle regret和 η 排序辅助指标。

| 折 | 表示 | 目标 train state 数 | 目标固定 η 成功率 | Critic+offset 成功率 | 观测 η 头数 |
| --- | --- | --- | --- | --- | --- |
| 1 | R0 | 1 | 55.47% | 55.47% | 8 |
| 1 | R0 | 2 | 28.91% | 28.91% | 16 |
| 1 | R0 | 4 | 23.83% | 23.83% | 16 |
| 1 | R3 | 1 | 55.47% | 51.95% | 8 |
| 1 | R3 | 2 | 28.91% | 28.91% | 16 |
| 1 | R3 | 4 | 23.83% | 23.44% | 16 |
| 1 | R4_action | 1 | 55.47% | 42.97% | 8 |
| 1 | R4_action | 2 | 28.91% | 28.91% | 16 |
| 1 | R4_action | 4 | 23.83% | 23.31% | 16 |
| 1 | R4_openloop | 1 | 55.47% | 45.57% | 8 |
| 1 | R4_openloop | 2 | 28.91% | 28.39% | 16 |
| 1 | R4_openloop | 4 | 23.83% | 24.09% | 16 |
| 1 | R3_R4_action | 1 | 55.47% | 50.91% | 8 |
| 1 | R3_R4_action | 2 | 28.91% | 28.91% | 16 |
| 1 | R3_R4_action | 4 | 23.83% | 24.09% | 16 |
| 4 | R0 | 1 | 90.23% | 90.23% | 8 |
| 4 | R0 | 2 | 90.23% | 90.23% | 16 |
| 4 | R0 | 4 | 90.23% | 90.23% | 16 |
| 4 | R3 | 1 | 90.23% | 90.23% | 8 |
| 4 | R3 | 2 | 90.23% | 90.23% | 16 |
| 4 | R3 | 4 | 90.23% | 88.28% | 16 |
| 4 | R4_action | 1 | 90.23% | 90.23% | 8 |
| 4 | R4_action | 2 | 90.23% | 90.23% | 16 |
| 4 | R4_action | 4 | 90.23% | 90.36% | 16 |
| 4 | R4_openloop | 1 | 90.23% | 90.23% | 8 |
| 4 | R4_openloop | 2 | 90.23% | 90.23% | 16 |
| 4 | R4_openloop | 4 | 90.23% | 90.76% | 16 |
| 4 | R3_R4_action | 1 | 90.23% | 90.23% | 8 |
| 4 | R3_R4_action | 2 | 90.23% | 90.23% | 16 |
| 4 | R3_R4_action | 4 | 90.23% | 87.50% | 16 |

辅助 Q 分解：七个 Controller 的平均Q水平标准差 0.1292；controller×η 交叉半样本能量 0.028200；state×controller×η 能量 0.022679。各自bootstrap区间、pairwise η rank相关和unknown比例见 `CONTROLLER_Q_DECOMPOSITION.json`。这些离散K16量不识别连续success basin。

### 条件分支

Stage B **完成**：同一 S_test 面板上增加七个外窗口真实 Flow 作为源；该面板已被 Stage A 打开，属于 DEVELOPMENT。独立的新增 rollout/unknown 见 `stage_b/POSTFLIGHT.json`。

| 表示 | 13源−6源成功率下界 | 95% 同时 CI |
| --- | --- | --- |
| R0 | -2.02pp | [-4.02pp, -0.01pp] |
| R3 | +0.72pp | [-2.53pp, +3.96pp] |
| R4_action | -0.78pp | [-4.57pp, +3.01pp] |
| R4_openloop | +0.07pp | [-4.56pp, +4.69pp] |
| R3_R4_action | -0.65pp | [-4.81pp, +3.51pp] |

Stage B 联合判据通过表示：R3, R3_R4_action。

## D. Hypothesis Evaluation

- **真实 Flow 会改变 Q level**：SUPPORTED（七个固定几何和执行条件下的描述性均值）；数值和区间见 Q 分解。
- **真实 Flow 会改变η排序**：SUPPORTED（必要折 2/7；全局与state层需分开，不能由Q MSE代替）。
- **现有静态表示能在未见真实 Flow 上可靠改善选择**：PARTIALLY SUPPORTED：Stage B 的13源设置通过同一已打开测试面板的联合门，但未有独立确认；不能把覆盖增加解释为因果修复。
- **更多源 Controller 能修复失败**：PARTIALLY SUPPORTED（同一已打开面板）。
- **短时静态探测表示优于投影动作/Jacobian**：仅看相应预注册配对差；若同时区间未过门则 REJECTED（本设计内），不得以Q MSE替代。

## E. Controller–Q Relationship and Transferability

固定 Ring geometry、FF placement、OrthoFlow3、硬安全投影、K16、H700、同一state及seed，只改变真实Flow checkpoint。Q水平差见分解；η排序变化见必要性门、跨 Controller 遗憾及pairwise rank。R3为compact64，R4_action为η=0投影动作与投影Jacobian，R4_openloop为静态短时探测；局部Jacobian不是完整闭环的精确线性化。匹配/错配表示的成功率差是机制表示的行为检验；Q MSE不证明selection。结果在七个真实Flow和必要折范围内解释，不能推至未知几何、placement或连续η空间。源 Controller 数从6→13只在Stage B实际执行时可检验，并且同一目标面板上的改善仍需独立确认。

## F. Limitations and Next Steps

七个Flow来自已有checkpoint库，不代表所有Controller；只固定Ring原生几何、FF、H700、K16及自然开发state分布。新64个 S_test state共享于七折，bootstrap按物理state成组；必要折由独立seed挑选，主终点只用seed4–7，但这仍是条件性结论。unknown按保守边界处理，代价是置信区间可能偏宽。Stage C受10,000新rollout上限约束，总64个state–controller面板会降低单折精度。零样本机制若失败，应首先诊断η排序与controller覆盖、真实闭环响应是否可由t0静态特征表征；不得从离散K16推断连续basin形状。

## G. Reproducibility and Evidence

工作根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`；记录的Git HEAD `a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e`，模型/runtime均JAX x64=false。精确controller checkpoint身份、state UID、η、seed、代码哈希在 `protocol.json`、`ENVIRONMENT.json`、`EXECUTION_IDENTITY_AUDIT.json`。运行命令与Slurm依赖见 `RUN_COMMANDS.md`；原始证据在 `raw/`、`static_features/`、`LABELS.npz`、`fits/fold*_{arm}/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、`results.json`及触发分支目录。`POSTFLIGHT.json`、`LABEL_AUDIT.json`、`FINAL_CONSISTENCY_AUDIT.json`和 `ARTIFACT_INDEX.json`供独立核查。所有报告中的事实均来自这些机器文件；机制解释为局部/统计解释，跨几何或连续basin推断仍属待验证假设。

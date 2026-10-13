# Real Flow LOCO：Controller-conditioned η selection 最终报告

## A. Research Overview

主问题：在源合并固定 η 对目标真实 Flow 明显不合适的留出 controller 上，部署时可静态计算的 controller 表示能否提高每个 state 的 η 选择闭环成功率？本研究把 Q 预测与 η 选择分开，主终点为独立 seed 的闭环成功率和预注册同时置信区间。上一轮 P1 在24个开发 state 上测到合格 Flow 的全局跨 controller η 遗憾中位数17.19pp （同时 CI [3.13,28.39]pp），但该面板已经打开，不能当作本轮新确认。

本轮分类：**(ii) 必要折上未得到独立确认；Stage B 开发集有正信号**。INCONCLUSIVE：Stage B 的13源设置在已打开面板上过门，但冻结模型未在新state同时超过全部对照；不能宣称稳健的跨Controller选择迁移。

| 阶段 | 执行状态 | 新rollout | 数值unknown | 决策 |
| --- | --- | --- | --- | --- |
| Stage A | 完成 | 86016 | 883 | 2必要折；主联合判据无表示通过 |
| Stage B | 完成 | 28672 | 324 | R3, R3_R4_action |
| Stage C | 完成 | 8192 | 100 | FAILED INDEPENDENT CONFIRMATION |

![Controller selection gains](figures/controller_selection_gains.png)

图中每阶段使用各自的同时CI族；单个对比为正不代表四项联合判据通过。最后一行为探索性的逐state选择相对模型自身controller固定ηprior的差值。可导出的矢量图为 `figures/controller_selection_gains.pdf`。

## B. Implementation Summary

先审计 `AGENTS.md`、Round 3 原始协议、14 个 Ring Flow checkpoint、shared DB 和历史标签。七个 [40%,95%] Flow 的 UID、checkpoint SHA、local/global observation frame 与 JAX x64=false 见 `EXECUTION_IDENTITY_AUDIT.json`。复用 Round 3 的 adapter、原生 v10 runtime、R3/R4_action/R4_openloop 静态特征和训练设置。所有新增文件在本目录；历史数据未修改、未做 Git commit。纯 Q 实验不要求视频。

用新的自然 state 序列生成128个 S_train、64个 S_test，七个 Flow 共用；全库/历史 state hash 查重为零，见 `STATE_DISJOINT_RECEIPT.json`。每个训练 state 分配8/16个 η、Q4；test 为K16×Q8，其中seed0–3仅用于目标固定 η 参照和必要性门，seed4–7仅用于最终评价。P1 的24个开发 state 仅供源验证 NLL 选步。训练为 physical23+表示→16-head MLP，3个 fit seed；只对有标签的头计损失。R0、R3、R4_action、R4_openloop、R3+R4_action用同一模型、数据、预算；R4基于η=0投影执行动作及投影后Jacobian，静态查询禁止 `env.step`。所有新 outcome 产生前的设计、决策门和时间/SHA凭证见下列预注册文件。

固定Ring为4-agent单积分器：中心障碍半径0.9、外边界半径3.25、agent半径0.16、最大速度0.52、dt=0.05、H700即35秒、到达容差0.09；安全/连续碰撞检查和成功终止语义沿用P1。η在episode内固定，K16完整保留（η13为下表index13，未删除强固定候选）。

| η index（0-based） | 三维episode级η |
| --- | --- |
| 0 | 0.562171, -0.361577, 0.696127 |
| 1 | 1.019353, 0.219025, 0.073114 |
| 2 | 1.142078, -0.111841, 0.535476 |
| 3 | 0.873041, 0.469239, 0.198283 |
| 4 | 0.703716, -0.246198, 0.113809 |
| 5 | 1.161976, 0.357536, 0.620183 |
| 6 | 0.908966, -0.496935, 0.367701 |
| 7 | 0.640320, 0.108254, 0.401121 |
| 8 | 0.643073, -0.038241, 0.295862 |
| 9 | 0.935523, 0.426934, 0.426504 |
| 10 | 1.246393, -0.288999, 0.186473 |
| 11 | 0.765795, 0.177650, 0.594161 |
| 12 | 0.782398, -0.415476, 0.514806 |
| 13 | 1.074506, 0.058060, 0.266061 |
| 14 | 1.011107, -0.165731, 0.717436 |
| 15 | 0.529389, 0.308303, 0.004511 |

| 预注册文件 | UTC 时间 | SHA256 |
| --- | --- | --- |
| PREREGISTRATION_CONDITIONAL_BRANCHES.md | 2026-10-12T03:47:51.656833+00:00 | c4a087b6657fd242107f3219550e082ee37d80da2e4032ca6d936c5e533b6c53 |
| PREREGISTRATION_RESOURCE_SUPPLEMENT.md | 2026-10-12T03:32:56.265221+00:00 | 7447d22da0c9132bdfcb66223efd9a5142d2707597acf24ee2a28ed0d82b9922 |
| PREREGISTRATION_STAGE_A.md | 2026-10-12T03:31:34.392184+00:00 | 695e015d9d0ebb799bda27700fca770b9679bfd4b0c4469b0ee40d7e67d36d53 |
| PREREGISTRATION_STAGE_C_AFTER_B.md | 2026-10-12T15:27:19.026032+00:00 | 24a3aae75d967151fccabc400ed6ef7d1255dc32e66a113bcbfd5b3d8a25b33f |

86,016个 Stage A 精确请求经官方 `python -m shared_rollout_db.plan` preflight，复用0、缺失86,016；DB experiment 行在 merge 前登记。98次计时 pilot、按 timing 的无结果重排、native v10 特征等价核查分别记录于 `PILOT_SAMPLING_DEVIATION.md`、`SCHEDULING_ADJUSTMENT.md`、`NATIVE_VALIDATION_FEATURE_AUDIT.json`。新 raw/journal 经单进程 merge，再做 postflight 和UID标签张量；数值失败保留为unknown。Stage A 86,016个新 raw、85,133 valid、883 unknown、0 missing、0 quarantine。七组静态特征查询均为0次 `env.step`。35组训练任务×3 fit seed＝105个 checkpoint；source-only归一化与P1验证选步。总复用P1验证标签约21,504个已有逐seed格标签；本轮所有阶段新增122,880个 raw，原始episode时间合计约209.32 CPU小时（未包含静态查询、训练、合并和排队时间）。

Stage B复用同一128个训练初态和η分配，为7个额外兼容Flow补28,672个确切缺失请求，10组必要折×表示任务产生30个fit-seed checkpoint；总计135次模型拟合。`TRAINING_PROVENANCE_AUDIT.json`逐一复算45组模型的源归一化，验证与保存值完全一致、留出controller没有进入源训练/验证列表，并记录外部代码、P1标签和物理特征资产哈希。静态表示与checkpoint推理阶段均为零env.step；新controller输入编码使用既有标签，没有另补rollout。

已有P1数据复用的21,504个逐seed格中，21,232为有效标签、272为数值unknown；unknown没有计入训练/验证NLL的成功或失败，见 `HISTORICAL_LABEL_REUSE_COUNTS.json`。每折的验证列表只含其源controller。

执行调整：原分支预注册把Stage C限定为Stage A过门，Stage B过门后按用户原始“有表示达到主判据”要求补写 `PREREGISTRATION_STAGE_C_AFTER_B.md`，冻结Stage B模型后才生成新的确认outcome。该补充是事前于Stage C、事后于Stage B的明确修订。最终资源授权为使用所有空闲CPU/GPU；因19-way采集分片已运行，保持其UID所有权，并把冻结推理/审计放入剩余CPU槽，与采集并行，GPU并发上限8。没有为了资源调整重复或中断rollout。实际调度记录在 `STAGE_C_RESOURCE_UPDATE.json`、`STAGE_C_AFTER_B_DISPATCH.json`，每个raw的内容哈希在 `RAW_CONTENT_INDEX_STAGE_AB.jsonl` / `RAW_CONTENT_INDEX_STAGE_C.jsonl`。

## C. Experimental Results

### Stage A：必要性门和 Controller–Q 关系

必要性门只读 S_test seed0–3：目标固定η用0–1↔2–3双向交叉拟合；源η来自六个源 Flow 的 S_train；对 unknown 使用配对保守下界。64个物理state成组bootstrap 2000次，门为 L≥5pp 且95%下界>0。必要折 2/7：[1, 4]。

| 折 | 留出真实 Flow | η_src index | train最佳固定η | 目标固定 η（交叉拟合） | L 下界 | 95% CI | 必要折 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | base_u_plus_w_v4_wide_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 1 | base_u_macflow | 10 | 13 | 13/13 | +28.52pp | [+17.19pp, +39.06pp] | 是 |
| 2 | base_u_plus_w_v3_macflow | 10 | 2 | 10/2 | -1.95pp | [-5.86pp, +1.95pp] | 否 |
| 3 | base_u_plus_w_v4_train_targeted_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 4 | base_u_v10_local_macflow | 10 | 13 | 13/13 | +34.38pp | [+24.60pp, +44.92pp] | 是 |
| 5 | base_u_plus_w_v4_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |
| 6 | base_u_v9_local_macflow | 10 | 10 | 10/10 | +0.00pp | [+0.00pp, +0.00pp] | 否 |

state层交叉拟合oracle（同一个已有state在独立seed上选择和评价，不是未见state的可部署selector）：

| 折 | state-oracle交叉拟合成功率下界 | vs源固定η下界 |
| --- | --- | --- |
| 0 | 68.75% | +5.47pp |
| 1 | 47.66% | +24.22pp |
| 2 | 89.84% | -1.56pp |
| 3 | 40.23% | -2.73pp |
| 4 | 85.16% | +27.34pp |
| 5 | 37.89% | -4.30pp |
| 6 | 85.94% | -10.16pp |

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

辅助 Q 分解：七个 Controller 的平均Q水平标准差 0.1275；controller×η 交叉半样本能量 0.028448；state×controller×η 能量 0.022738。各自bootstrap区间、pairwise η rank相关和unknown比例见 `CONTROLLER_Q_DECOMPOSITION.json`。这些离散K16量不识别连续success basin。

辅助分解的初版曾隐式将unknown按0参与能量，已在最终审计中修正并保存旧版本。现用unknown=[0,1]的中点及正交投影/Cauchy界进行敏感性分析，bootstrap区间扩展以覆盖unknown完成值，不声称任意缺失情况下严格无偏。修正后的controller×η能量区间[0.023235,0.034666]，state×controller×η区间[0.011014,0.034132]，仍为正；详见 `DECOMPOSITION_UNKNOWN_POLICY_AUDIT.md`。主判据和模型没有受到该辅助修正影响。

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

Stage B 的完整必要折终点与主对比（开发证据）：

| 折 | 表示 | 成功率下界 | vs源固定η | vs R0 | matched−mismatched |
| --- | --- | --- | --- | --- | --- |
| 1 | R0 | 22.40% | -1.43pp | +0.00pp | +0.00pp |
| 1 | R3 | 25.26% | +1.30pp | +2.73pp | +7.27pp |
| 1 | R4_action | 21.88% | -2.47pp | -1.04pp | +0.07pp |
| 1 | R4_openloop | 24.48% | -0.26pp | +1.17pp | +2.15pp |
| 1 | R3_R4_action | 29.69% | +5.60pp | +7.03pp | +11.89pp |
| 4 | R0 | 64.06% | +0.00pp | +0.00pp | +0.00pp |
| 4 | R3 | 76.04% | +11.98pp | +11.98pp | +11.81pp |
| 4 | R4_action | 66.67% | +2.60pp | +2.60pp | +1.17pp |
| 4 | R4_openloop | 65.36% | +1.30pp | +1.30pp | -1.04pp |
| 4 | R3_R4_action | 72.40% | +8.33pp | +8.33pp | +7.14pp |

| 表示 | 对照 | 差值 | 95%同时CI |
| --- | --- | --- | --- |
| R3 | src | +6.64pp | [+2.57pp, +10.71pp] |
| R3 | near | +6.64pp | [+2.57pp, +10.71pp] |
| R3 | R0 | +7.36pp | [+3.14pp, +11.57pp] |
| R3 | mismatch | +9.54pp | [+5.71pp, +13.36pp] |
| R4_action | src | +0.07pp | [-5.30pp, +5.43pp] |
| R4_action | near | +0.07pp | [-5.30pp, +5.43pp] |
| R4_action | R0 | +0.78pp | [-4.15pp, +5.71pp] |
| R4_action | mismatch | +0.62pp | [-3.21pp, +4.45pp] |
| R4_openloop | src | +0.52pp | [-5.60pp, +6.64pp] |
| R4_openloop | near | +0.52pp | [-5.60pp, +6.64pp] |
| R4_openloop | R0 | +1.24pp | [-4.74pp, +7.22pp] |
| R4_openloop | mismatch | +0.55pp | [-4.38pp, +5.49pp] |
| R3_R4_action | src | +6.97pp | [+2.43pp, +11.50pp] |
| R3_R4_action | near | +6.97pp | [+2.43pp, +11.50pp] |
| R3_R4_action | R0 | +7.68pp | [+3.05pp, +12.31pp] |
| R3_R4_action | mismatch | +9.52pp | [+5.67pp, +13.36pp] |

不能将Stage B过门直接解释为源覆盖不足被修复：R3和组合表示的Stage B−A同时区间均覆盖0；R0下降约2pp，组合表示的总体成功率也略降。fold1变好而fold4变差，源增加并非逐折单调改善。两个必要折中源先验与最近源均选择η10，因此(a)/(b)实际相同；目标自身固定η使用目标标签，仍明显更强。

| 折 | 表示 | 目标state数 | 目标固定η成功率 | critic+offset成功率 |
| --- | --- | --- | --- | --- |
| 1 | R0 | 1 | 55.47% | 55.47% |
| 1 | R0 | 2 | 28.91% | 28.91% |
| 1 | R0 | 4 | 23.83% | 23.83% |
| 1 | R3 | 1 | 55.47% | 55.47% |
| 1 | R3 | 2 | 28.91% | 28.91% |
| 1 | R3 | 4 | 23.83% | 23.83% |
| 1 | R4_action | 1 | 55.47% | 31.25% |
| 1 | R4_action | 2 | 28.91% | 28.91% |
| 1 | R4_action | 4 | 23.83% | 23.83% |
| 1 | R4_openloop | 1 | 55.47% | 36.72% |
| 1 | R4_openloop | 2 | 28.91% | 28.91% |
| 1 | R4_openloop | 4 | 23.83% | 24.09% |
| 1 | R3_R4_action | 1 | 55.47% | 50.91% |
| 1 | R3_R4_action | 2 | 28.91% | 28.91% |
| 1 | R3_R4_action | 4 | 23.83% | 23.83% |
| 4 | R0 | 1 | 90.23% | 90.23% |
| 4 | R0 | 2 | 90.23% | 90.23% |
| 4 | R0 | 4 | 90.23% | 90.23% |
| 4 | R3 | 1 | 90.23% | 90.23% |
| 4 | R3 | 2 | 90.23% | 90.23% |
| 4 | R3 | 4 | 90.23% | 88.80% |
| 4 | R4_action | 1 | 90.23% | 90.23% |
| 4 | R4_action | 2 | 90.23% | 90.23% |
| 4 | R4_action | 4 | 90.23% | 90.23% |
| 4 | R4_openloop | 1 | 90.23% | 90.23% |
| 4 | R4_openloop | 2 | 90.23% | 90.23% |
| 4 | R4_openloop | 4 | 90.23% | 90.23% |
| 4 | R3_R4_action | 1 | 90.23% | 90.23% |
| 4 | R3_R4_action | 2 | 90.23% | 90.23% |
| 4 | R3_R4_action | 4 | 90.23% | 86.59% |

Stage C **完成**：按 `PREREGISTRATION_STAGE_C_AFTER_B.md` 的前瞻性补充规则，在新自然 state 上使用冻结 Stage B checkpoint、归一化和源 prior。共 64 个 state–controller 面板，状态 **FAILED INDEPENDENT CONFIRMATION**；确认表示：无。

| 表示 | 配对差 | 均值 | 95% 同时 CI |
| --- | --- | --- | --- |
| R3 | src | +5.21pp | [-0.66pp, +11.08pp]  |
| R3 | near | +5.21pp | [-0.66pp, +11.08pp]  |
| R3 | R0 | +5.73pp | [-0.13pp, +11.59pp]  |
| R3 | mismatch | +7.51pp | [+2.53pp, +12.49pp]  |
| R3_R4_action | src | +3.78pp | [-2.40pp, +9.96pp]  |
| R3_R4_action | near | +3.78pp | [-2.40pp, +9.96pp]  |
| R3_R4_action | R0 | +4.30pp | [-1.75pp, +10.34pp]  |
| R3_R4_action | mismatch | +5.38pp | [+0.78pp, +9.99pp]  |

| 折 | 表示 | 确认state数 | 选择成功率下界 | 源固定η | R0 | 目标标签固定η |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | R3 | 32 | 31.25% | 29.69% | 28.65% | 64.84% |
| 1 | R3_R4_action | 32 | 30.21% | 29.69% | 28.65% | 64.84% |
| 4 | R3 | 32 | 69.01% | 60.16% | 60.16% | 89.06% |
| 4 | R3_R4_action | 32 | 67.45% | 60.16% | 60.16% | 89.06% |

Stage C生成64个去重初态，实际两个controller各标注相同前32个初态：32个独立物理state、64个state–controller面板、8,192请求。原任务每controller64个state需16,384次，会超过10,000硬上限；此缩减在新outcome前冻结。Stage C只检验Stage B冻结的R3和组合表示以及R0对照，不重新训练、调参或选择。

### Controller级prior与逐state选择的分解（探索性）

使用冻结模型在128个S_train初始state上的Q预测取均值，选择一个controller级固定η，完全不读取该controller的目标标签。η及checkpoint哈希在Stage C新outcome读取前冻结，见 `EXPLORATORY_CONTROLLER_PRIOR_PROTOCOL.md`、`controller_prior/`。这是使用无标签目标初态的zero-target-label prior估计，与1/2/4个目标state的有监督calibration不同；其统计对比不改变主判据。以下比较才直接检验逐state选择是否额外优于模型自身已识别的controller η regime。

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

### Q预测与源η-regime覆盖（描述性/探索性）

| 阶段 | 表示 | 必要折平均Q MSE（辅助） |
| --- | --- | --- |
| Stage A | R0 | 0.1011 |
| Stage A | R3 | 0.0877 |
| Stage A | R4_action | 0.1126 |
| Stage A | R4_openloop | 0.1150 |
| Stage A | R3_R4_action | 0.0851 |
| Stage B | R0 | 0.1046 |
| Stage B | R3 | 0.0762 |
| Stage B | R4_action | 0.1159 |
| Stage B | R4_openloop | 0.1117 |
| Stage B | R3_R4_action | 0.0762 |

R3/组合表示的Q误差在Stage B优于R0，R4动作/短时探测并未因此获得选择过门证据。Q标签只有4个endpoint seed且含unknown，MSE是同一已观察面板上的辅助误差，不能解释成真实连续Q的精确误差或替代Stage C选择确认。

| Controller | 组 | 训练经验最佳η（成功率下界） | 最佳η成功率下界 | η13成功率下界 |
| --- | --- | --- | --- | --- |
| base_u_plus_w_v4_wide_macflow | original | 10 | 65.62% | 25.78% |
| base_u_macflow | original | 13 | 57.03% | 57.03% |
| base_u_plus_w_v3_macflow | original | 2 | 96.88% | 66.02% |
| base_u_plus_w_v4_train_targeted_macflow | original | 10 | 35.94% | 7.03% |
| base_u_v10_local_macflow | original | 13 | 91.41% | 91.41% |
| base_u_plus_w_v4_macflow | original | 10 | 34.77% | 4.69% |
| base_u_v9_local_macflow | original | 10 | 96.48% | 80.47% |
| base_u_v4_macflow | added | 10 | 48.83% | 21.09% |
| base_u_plus_w_v6_macflow | added | 10 | 45.70% | 5.08% |
| base_u_v3_macflow | added | 2 | 91.80% | 61.72% |
| base_u_v2_macflow | added | 10 | 26.56% | 23.05% |
| base_u_plus_w_v5_macflow | added | 10 | 35.94% | 8.98% |
| base_u_v7_local_macflow | added | 10 | 95.31% | 35.94% |
| base_u_v8_local_macflow | added | 4 | 99.61% | 38.28% |

7个新增Flow在本轮训练标签上的经验最佳η为10/2/4，没有新增经验最优η13的controller。每个必要折原有6源中的η13经验最优源只有另一个必要controller；增加到13源后仍只有这一个。故Stage B增加了controller总数，却没有增加这个关键η-regime的经验最佳controller例数；这是可审计的覆盖缺口，尚不能证明它是唯一失败原因。窗口资格来自P1的24个分层开发state，不按本轮自然state成功率重新筛选Flow。完整曲线、unknown上下界及候选分配在 `SOURCE_REGIME_COVERAGE.json`。

## D. Hypothesis Evaluation

- **真实 Flow 会改变 Q level**：SUPPORTED（七个固定几何和执行条件下的描述性均值）；数值和区间见 Q 分解。
- **真实 Flow 会改变η排序**：SUPPORTED（必要折 2/7；全局与state层需分开，不能由Q MSE代替）。
- **Controller仅改变Q整体水平、η排序不变**：REJECTED（本固定Ring面板）；两个必要折的固定η交叉拟合损失及controller×η交互不支持纯level解释。
- **本轮表示已达到独立确认主判据**：REJECTED（预注册操作性成功条件）；这不等同于拒绝所有正效应，真实效应大小仍须按CI解释。
- **现有静态表示能在未见真实 Flow 上可靠改善选择**：INCONCLUSIVE：Stage B 的13源设置在已打开面板上过门，但冻结模型未在新state同时超过全部对照；不能宣称稳健的跨Controller选择迁移。
- **源controller覆盖是失败的主要根因**：INCONCLUSIVE；增加源的Stage B−A配对成功率差未显示R3/组合表示显著提升，且增加controller同时增加标签量并改变源分布。不能把Stage B过门等同于因果瓶颈被定位。
- **R4_action/R4_openloop提供可靠选择迁移**：本设计中没有通过预注册联合判据；它们的普遍充分性仍为INCONCLUSIVE。Q误差改善不能替代选择检验。

## E. Controller–Q Relationship and Transferability

固定 Ring geometry、FF placement、OrthoFlow3、硬安全投影、K16、H700、同一state及seed，只改变真实Flow checkpoint。Q水平差见分解；η排序变化见必要性门、跨 Controller 遗憾及pairwise rank。R3为四组compact64拼接（256维）；R4_action为η=0投影动作与有限差分Jacobian，查询t0及沿目标方向平移0.1的静态probe（总64维，差分eps=0.001）。R4_openloop拼接t0动作/Jacobian与一次p←p+dt·a0、v←a0的解析运动探测，无env.step，亦非真实短rollout。局部Jacobian不等同于完整闭环success basin。physical23包含初始径向距离、相位、目标径向距离/角偏差及局部速度；每臂输入统一补零至343维，网络64→64→16，12,000步，AdamW lr0.0008、wd0.0001、clip5，batch128。训练controller均衡采样，验证NLL按3个fit seed均值选共同checkpoint步；特征最终标准化仅使用源训练样本。

匹配/错配差是部署表示影响选择的行为证据，不单独识别因果机制；主终点由上面的联合对比和独立确认判定。新增source后的已打开面板与新state确认分开解释；逐state选择相对学习到的controller prior的增益另列探索性对比。未测试新几何、placement或连续η，不能推广；R4的局部作用不足并不证明静态表示原则必然无效。

事实：R4的Jacobian在η=0查询，但K16第一维约0.53–1.25且没有η=0候选。机制猜想：有限幅度FF积分、硬投影active-set变化及后续交互可能使局部响应不足以预测η排序；本轮没有针对这些环节的因果消融，不能将它们写成已定位的失败根因。错配仅替换critic输入的controller描述，实际闭环标签始终来自正确留出controller；Stage B/C错配仍对原来的其他6个controller取平均。

## F. Limitations and Next Steps

七个Flow来自已有checkpoint库，不代表所有Controller；只固定Ring原生几何、FF、H700、K16及自然开发state分布。新64个 S_test state共享于七折，bootstrap按物理state成组；必要折由独立seed挑选，主终点只用seed4–7，但这仍是条件性结论。unknown按保守边界处理，代价是置信区间可能偏宽。Stage C受10,000新rollout上限约束，总64个state–controller面板会降低单折精度。零样本机制若失败，应首先诊断η排序与controller覆盖、真实闭环响应是否可由t0静态特征表征；不得从离散K16推断连续basin形状。

置信区间描述这组固定模型、固定controller上的state/seed波动，未估计无限controller总体的泛化误差；两个必要折不能代表所有真实Flow。3个fit seed先各自选择η并评价，再平均成功率，主终点不是logit ensemble；逐fit差异保留在结果与预测。1/2/4-state calibration只使用预先固定的前k个训练初态，没有重复抽取calibration子集，因此其样本效率为描述性。学习到的controller固定ηprior的增益分解是新增探索性分析，不能替换确认主判据，也不能由其零目标标签性能声称物理因果机制。

Stage C按冻结规则未通过可靠迁移主门，并不证明真实收益为零：R3的同时CI仍包含约+11pp的正效应。之后若基于这些结果修改模型，Stage A的64个和Stage C的32个已打开物理state都应作为DEVELOPMENT，不可重新包装为独立确认。`stage_c/protocol.json`中另32个生成初态只做了静态查询、没有rollout标签，`LABELS.npz`相应位置为-2（unrequested）；本任务不追加，未来如补证需新协议/预算、重新preflight并记录该出处。

## G. Reproducibility and Evidence

工作根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`；记录的Git HEAD `a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e`，模型/runtime均JAX x64=false。精确controller checkpoint身份、state UID、η、seed、代码哈希在 `protocol.json`、`ENVIRONMENT.json`、`EXECUTION_IDENTITY_AUDIT.json`。运行命令与Slurm依赖见 `RUN_COMMANDS.md`；原始证据在 `raw/`、`static_features/`、`LABELS.npz`、`fits/fold*_{arm}/`、`GATE_RESULTS.json`、`CONTROLLER_Q_DECOMPOSITION.json`、`results.json`及触发分支目录。`POSTFLIGHT.json`、`LABEL_AUDIT.json`、`FINAL_CONSISTENCY_AUDIT.json`和 `ARTIFACT_INDEX.json`供独立核查。所有报告中的事实均来自这些机器文件；机制解释为局部/统计解释，跨几何或连续basin推断仍属待验证假设。

`STAGE_A_RESULTS_FROZEN.json`保留Stage A原始分析；`stage_b/results.json`与`stage_c/results.json`保留分阶段结果，最终 `results.json` 的 `study_summary` / `conditional_stages` 聚合全任务，原顶层Stage A rows/primary为兼容保留。重算统计与报告无需新rollout，运行本目录 `analyze.py`（Stage A）或 `analyze.py --stage-b`、`stage_b_compare.py`、`stage_c_analyze.py`、`controller_prior_analyze.py`、`render_report.py`、`final_audit.py`。现成checkpoint可直接用于新初态静态查询；没有target label输入时使用 `stage_c_predict.py` 的冻结推理路径，不能把目标校准offset默认当zero-shot。

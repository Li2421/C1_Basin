# Controller pathway fix：方法偏置、伤害消除与零样本迁移

状态 **COMPLETE**；更新时间UTC 2026-10-13T03:47:40.335143+00:00；全部结果 **DEVELOPMENT**。新增rollout **0**，共享DB写入 **0**。

## A. Research Overview

目标是解释并修复Q(s,C,η)的controller通路在未见controller上过度外推、argmax选择被高估η的风险。本轮使用已经打开的Ring/FF/H700/K16标签，检验降维、向R0收缩、悲观/保守选择和两段式regime模型，主终点是实际缓存闭环success，而非Q误差。修复能够消除伤害，按预注册操作映射支持H_method。融合族SPARSE的有界残差+悲观/保守规则有正收益；真实Flow尚无同时满足安全与正收益的规则。高维过参数化是否为唯一根因仍INCONCLUSIVE。

先核查round5：3个必要controller，原主门要求4，故当时INCONCLUSIVE；原R3最大损失−57.81pp、oracle−52.08pp；DENSE−SPARSE分别−9.90pp CI[−17.71,−2.08]、−26.91pp CI[−34.23,−19.59]。R0并非所有目标Q2 MSE都约.06，其最大.165；不能把少数例子推广到全部。核查详见BACKGROUND_FACT_AUDIT.json。round4七折中1/4两必要折，Stage C只有旧确认背景，不在本轮重新确认。14原Flow经验regime分布9/2/2/1沿用源标签核查。

## B. Implementation Summary

读取AGENTS.md、round6完整prompt、round4/5 HANDOFF/REPORT/results/protocol/fits/静态特征和P1验证标签，HISTORICAL_ASSET_INDEX.json逐文件hash。prepare.py登记38个冻结重放任务和48个融合新模型预测任务；replay.py直接加载原checkpoint、原输入与normalization，以JAX x64=false GPU推理。无Flow静态查询、无env.step调用，训练/终点脚本无DB连接与merger。只在本目录写文件，不修改历史数据，不commit，纯Q无视频。

阶段0：所有38任务argmax和原报告success均逐值一致。14个任务辅助logit误差超过自主追加的2e-5门，最大0.0016608；原log保持FAIL，不伪称bitwise重放。归一化重算逐值一致，numpy float32与历史CUDA前向也有小差异（Q最大0.000429），支持浮点/后端差异解释，但没有证明具体kernel是唯一原因。PREREGISTRATION_REPLAY_NUMERICS_ADDENDUM.md明确主用户复现门PASS、辅助逐值门未过；F3用原冻结预测缓存，F2用逐值一致的R0。此调整发生在修复训练前，未按新终点调参。

阶段1：util.Data只把源TRAIN标签用于训练与经验regime；controller级R3在共同96初态上求均值、source-only zscore/SVD取1/2/3维，目标仅投影、不参与PCA。F1-oracle-low输入9维已知融合坐标及源锚点regime。每个输入physical23+pad513，F1沿用64/64/16-head SiLU、AdamW lr.0008/wd.0001/clip5/12k步骤、17/23/41、3fit平均源验证NLL选step。F2先冻结同折R0，再训练16/16/16零输出初始化残差，tanh界±1、wd.01；只对valid heads计loss。F1/F2共48任务×3seed引用，但相同SPARSE源模型只优化一次并SHA复用。F4是source-only固定L2=1分类器，置信度.6，否则src prior。F3a κ1为主、.5为敏感性，全部固定；F3b三fit top1一致且平均预测优势≥.10才偏离，否则prior。F3相当于ensemble部署，原argmax每fit部署后平均，两者区分。统计首次只因NumPy int64无法JSON序列化而停止，转成Python int后从冻结预测重算，没有改统计值、调参或重训，见ANALYSIS_SERIALIZATION_REPAIR.json。

阶段2：资格由融合E1 simultaneous CI下界>−2pp、possible E3=0冻结到STAGE2_ELIGIBILITY.json。所有真实七折B1-F3无条件评价；合格架构只训一次再报告预定合格policies。oracle-low的真实Flow输入需目标自身regime，零样本不可得，故明确INAPPLICABLE，绝不读取目标标签补onehot。实际新MLP独立fit **174**，模型预测task **76**，小logistic tasks **15**。来源规模：训练94,208、P1验证21,504、融合test/discovery24,576、真实test/discovery57,344；去重复用 **197,632 seed标签槽位**（包含unknown，不等同全valid），无新增标签。

资源：与另一任务共享18CPU，本任务优先，GPU最多3shards且每job1CPU；实测3fits37.43s，成本门保留4倍余量低于8h，COST_GATE.json。Slurm执行commands在SLURM_SUBMISSIONS.json与分支dispatch。所有脚本/输入/参数/输出索引独立保存。

## C. Experimental Results

融合族SPARSE：F2-PCA2+κ=1悲观选择，在3个必要controller上相对源prior及R0的保守收益为+7.81pp（上界+8.33pp），描述性95% CI [1.56,14.06]pp；E1同时CI下界−1.48pp超过−2pp非劣界，E3=0/12。F2-PCA2+保守偏离收益+4.69pp，CI [1.04,8.33]pp，E1下界−1.04pp，E3=0/12。分别闭合target固定η差距55.6%和33.3%；因2次bootstrap分母非正，融合比例CI记NA。

真实Flow：F2-PCA2+保守偏离在2个必要controller上相对源prior提高+8.79pp，描述性CI [4.30,13.48]pp；相对R0提高+7.49pp，CI [3.06,12.11]pp。但在非必要的v9_local上相对源prior损失−10.16pp，E3=1/7，故不能判为安全正迁移。该收益来自v10_local（+17.58pp），base_u为0，两个η13型controller的不对称仍在。B1+保守偏离、F1-PCA1/2/3+保守偏离、F4可满足安全门，但在必要controller上与源prior完全相同（E2=0）。因此尚没有已验证“既安全又有效”的真实Flow修复，不应只看平均收益。

机制诊断：融合DENSE的旧R3/oracle被选η的平均预测高估上界分别+5.17/+12.99pp，凸包距离与相对R0损失的描述性Pearson为.647/.975；有界残差PCA2 argmax的对应高估为−6.57pp。降维/收缩大幅减少误选η1/η8；请求头数本身平衡，不能将错误简单归为这些head没标签。相关性不是因果，各目标明细及solver状态见D1–D3机器结果。

| 阶段 | 状态 |
| --- | --- |
| 0 | 完成：argmax/success复现；logits非bitwise且公开记录 |
| 1 | 完成：F1/F2/F3/F4、E1–E3、D1–D3 |
| 2 | 完成：28个真实拟合任务；oracle-low不适用 |

### 融合族

H_method 操作判定：**SUPPORTED: operational repair eliminates harm**。安全修复数18，正迁移候选4。每个controller/fold/fit的选择、unknown边界、source prior、target fixed及η频率保存在对应results.json的rows。

| 覆盖 | arm/规则 | 成功下界 | E1 vs R0同时CI | E1过门 | E2 vs源prior描述CI | E2 vsR0描述CI | gap闭合比例 | E3确定–可能 | 安全合格 | 规则证据类别 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | B0__argmax | 79.86% | [+0.00pp, +0.00pp] | True | [+0.00pp, +0.52pp] | [+0.00pp, +0.00pp] | 0.012 | 0–0/12 | True | 源门预定 |
| DENSE | B1__argmax | 64.76% | [-20.01pp, -10.42pp] | False | [-55.21pp, -36.81pp] | [-55.21pp, -36.98pp] | -3.296 | 4–4/12 | False | 源门预定 |
| DENSE | B1__pessimistic1 | 63.80% | [-21.46pp, -10.74pp] | False | [-50.00pp, -32.29pp] | [-50.00pp, -32.47pp] | -2.926 | 4–4/12 | False | 源门预定 |
| DENSE | B1__pessimistic0.5 | 63.67% | [-21.55pp, -10.91pp] | False | [-51.04pp, -32.81pp] | [-51.22pp, -33.16pp] | -3.000 | 4–4/12 | False | 敏感性 κ=.5 |
| DENSE | B1__conservative | 79.69% | [-3.41pp, +3.00pp] | False | [-9.90pp, -0.51pp] | [-10.07pp, -0.52pp] | -0.333 | 0–0/12 | False | 源门预定 |
| DENSE | B2__argmax | 60.94% | [-24.34pp, -13.62pp] | False | [-44.62pp, -28.30pp] | [-44.79pp, -28.65pp] | -2.605 | 6–6/12 | False | 源门预定 |
| DENSE | B2__pessimistic1 | 62.50% | [-25.98pp, -8.77pp] | False | [-45.83pp, -22.92pp] | [-46.01pp, -23.09pp] | -2.444 | 6–6/12 | False | 源门预定 |
| DENSE | B2__pessimistic0.5 | 63.41% | [-24.85pp, -8.29pp] | False | [-42.19pp, -21.35pp] | [-42.36pp, -21.52pp] | -2.259 | 6–6/12 | False | 敏感性 κ=.5 |
| DENSE | B2__conservative | 75.65% | [-8.50pp, +0.13pp] | False | [-16.67pp, -2.60pp] | [-16.84pp, -2.95pp] | -0.667 | 3–3/12 | False | 源门预定 |
| DENSE | F1_PCA1__argmax | 78.17% | [-4.59pp, +1.10pp] | False | [-3.82pp, -0.35pp] | [-3.99pp, -0.52pp] | -0.136 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA1__pessimistic1 | 78.39% | [-4.12pp, +1.21pp] | False | [-5.21pp, +1.04pp] | [-5.39pp, +0.69pp] | -0.148 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA1__pessimistic0.5 | 78.78% | [-3.55pp, +1.41pp] | False | [-2.08pp, +1.04pp] | [-2.43pp, +0.69pp] | -0.037 | 0–0/12 | False | 敏感性 κ=.5 |
| DENSE | F1_PCA1__conservative | 78.91% | [-3.10pp, +1.16pp] | False | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA2__argmax | 78.17% | [-4.19pp, +0.62pp] | False | [-4.17pp, +0.00pp] | [-4.34pp, -0.35pp] | -0.148 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA2__pessimistic1 | 78.12% | [-4.84pp, +1.12pp] | False | [-5.21pp, +1.04pp] | [-5.39pp, +0.69pp] | -0.148 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA2__pessimistic0.5 | 78.52% | [-4.21pp, +1.56pp] | False | [-2.08pp, +2.08pp] | [-2.08pp, +1.74pp] | 0.000 | 0–0/12 | False | 敏感性 κ=.5 |
| DENSE | F1_PCA2__conservative | 78.91% | [-3.10pp, +1.16pp] | False | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA3__argmax | 78.65% | [-3.78pp, +1.16pp] | False | [-2.43pp, +1.91pp] | [-2.43pp, +1.39pp] | -0.025 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA3__pessimistic1 | 78.26% | [-4.89pp, +1.38pp] | False | [-5.73pp, +1.56pp] | [-5.73pp, +1.56pp] | -0.111 | 0–0/12 | False | 源门预定 |
| DENSE | F1_PCA3__pessimistic0.5 | 78.39% | [-4.65pp, +1.39pp] | False | [-3.12pp, +1.04pp] | [-3.12pp, +0.52pp] | -0.074 | 0–0/12 | False | 敏感性 κ=.5 |
| DENSE | F1_PCA3__conservative | 78.91% | [-3.10pp, +1.16pp] | False | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | False | 源门预定 |
| DENSE | F1_oracle_low__argmax | 80.34% | [-4.71pp, +5.41pp] | False | [-6.77pp, +3.30pp] | [-6.94pp, +2.95pp] | -0.148 | 0–1/12 | False | 源门预定 |
| DENSE | F1_oracle_low__pessimistic1 | 79.95% | [-5.75pp, +5.60pp] | False | [-5.73pp, +5.73pp] | [-5.91pp, +5.21pp] | -0.037 | 0–0/12 | False | 源门预定 |
| DENSE | F1_oracle_low__pessimistic0.5 | 80.34% | [-5.77pp, +6.32pp] | False | [-5.73pp, +5.21pp] | [-5.73pp, +4.86pp] | -0.037 | 0–0/12 | False | 敏感性 κ=.5 |
| DENSE | F1_oracle_low__conservative | 78.91% | [-3.10pp, +1.16pp] | False | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | False | 源门预定 |
| DENSE | F2_PCA2__argmax | 81.29% | [-0.10pp, +2.94pp] | True | [+0.00pp, +0.87pp] | [+0.00pp, +0.52pp] | 0.025 | 0–0/12 | True | 源门预定 |
| DENSE | F2_PCA2__pessimistic1 | 81.12% | [-0.38pp, +2.89pp] | True | [+0.00pp, +3.12pp] | [+0.00pp, +2.60pp] | 0.074 | 0–0/12 | True | 源门预定 |
| DENSE | F2_PCA2__pessimistic0.5 | 81.25% | [-0.32pp, +3.09pp] | True | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | True | 敏感性 κ=.5 |
| DENSE | F2_PCA2__conservative | 80.34% | [-1.00pp, +1.87pp] | True | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| DENSE | F2_oracle_low__argmax | 81.38% | [-0.77pp, +3.60pp] | True | [-0.17pp, +6.42pp] | [-0.17pp, +5.90pp] | 0.210 | 0–0/12 | True | 源门预定 |
| DENSE | F2_oracle_low__pessimistic1 | 81.38% | [-1.62pp, +4.34pp] | True | [-1.04pp, +7.29pp] | [-1.04pp, +6.94pp] | 0.222 | 0–0/12 | True | 源门预定 |
| DENSE | F2_oracle_low__pessimistic0.5 | 81.77% | [-1.03pp, +4.50pp] | True | [+1.04pp, +7.81pp] | [+0.87pp, +7.47pp] | 0.296 | 0–0/12 | True | 敏感性 κ=.5 |
| DENSE | F2_oracle_low__conservative | 79.17% | [-2.66pp, +1.41pp] | False | [+0.00pp, +3.12pp] | [+0.00pp, +2.60pp] | 0.074 | 0–0/12 | False | 源门预定 |
| DENSE | F4__argmax | 80.08% | [-1.80pp, +2.24pp] | True | [+0.00pp, +0.00pp] | [-0.52pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| SPARSE | B0__argmax | 78.91% | [+0.00pp, +0.00pp] | True | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| SPARSE | B1__argmax | 66.28% | [-18.22pp, -7.28pp] | False | [-42.36pp, -30.73pp] | [-42.36pp, -30.73pp] | -2.605 | 3–3/12 | False | 源门预定 |
| SPARSE | B1__pessimistic1 | 63.93% | [-21.08pp, -9.23pp] | False | [-53.65pp, -34.90pp] | [-53.65pp, -34.90pp] | -3.148 | 4–4/12 | False | 源门预定 |
| SPARSE | B1__pessimistic0.5 | 63.41% | [-21.23pp, -10.08pp] | False | [-56.25pp, -39.06pp] | [-56.25pp, -39.06pp] | -3.370 | 4–4/12 | False | 敏感性 κ=.5 |
| SPARSE | B1__conservative | 77.60% | [-4.43pp, +1.75pp] | False | [-12.50pp, -2.60pp] | [-12.50pp, -2.60pp] | -0.519 | 1–1/12 | False | 源门预定 |
| SPARSE | B2__argmax | 71.40% | [-11.98pp, -3.25pp] | False | [-15.62pp, -3.82pp] | [-15.62pp, -3.82pp] | -0.704 | 3–3/12 | False | 源门预定 |
| SPARSE | B2__pessimistic1 | 72.92% | [-12.13pp, +0.05pp] | False | [-26.56pp, -8.85pp] | [-26.56pp, -8.85pp] | -1.259 | 3–4/12 | False | 源门预定 |
| SPARSE | B2__pessimistic0.5 | 72.79% | [-11.50pp, -0.85pp] | False | [-26.04pp, -9.38pp] | [-26.04pp, -9.38pp] | -1.259 | 3–4/12 | False | 敏感性 κ=.5 |
| SPARSE | B2__conservative | 79.30% | [-0.81pp, +1.59pp] | True | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| SPARSE | F1_PCA1__argmax | 78.26% | [-2.27pp, +0.65pp] | False | [-3.65pp, -0.17pp] | [-3.65pp, -0.17pp] | -0.123 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_PCA1__pessimistic1 | 78.39% | [-3.00pp, +1.48pp] | False | [-3.12pp, +1.56pp] | [-3.12pp, +1.56pp] | -0.037 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_PCA1__pessimistic0.5 | 78.39% | [-2.62pp, +1.58pp] | False | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | False | 敏感性 κ=.5 |
| SPARSE | F1_PCA1__conservative | 78.91% | [+0.00pp, +0.00pp] | True | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| SPARSE | F1_PCA2__argmax | 78.39% | [-2.49pp, +1.12pp] | False | [-3.99pp, +1.22pp] | [-3.99pp, +1.22pp] | -0.099 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_PCA2__pessimistic1 | 78.26% | [-3.29pp, +1.54pp] | False | [-4.69pp, +2.08pp] | [-4.69pp, +2.08pp] | -0.074 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_PCA2__pessimistic0.5 | 78.39% | [-2.92pp, +1.39pp] | False | [-3.12pp, +0.00pp] | [-3.12pp, +0.00pp] | -0.074 | 0–0/12 | False | 敏感性 κ=.5 |
| SPARSE | F1_PCA2__conservative | 78.91% | [+0.00pp, +0.00pp] | True | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| SPARSE | F1_PCA3__argmax | 78.60% | [-2.27pp, +1.33pp] | False | [-3.47pp, +1.22pp] | [-3.47pp, +1.22pp] | -0.074 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_PCA3__pessimistic1 | 78.52% | [-2.94pp, +1.60pp] | False | [-3.12pp, +0.00pp] | [-3.12pp, +0.00pp] | -0.074 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_PCA3__pessimistic0.5 | 78.52% | [-2.70pp, +1.41pp] | False | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | False | 敏感性 κ=.5 |
| SPARSE | F1_PCA3__conservative | 78.91% | [+0.00pp, +0.00pp] | True | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |
| SPARSE | F1_oracle_low__argmax | 80.73% | [-3.37pp, +7.10pp] | False | [-4.34pp, +7.12pp] | [-4.34pp, +7.12pp] | 0.086 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_oracle_low__pessimistic1 | 79.95% | [-5.49pp, +7.58pp] | False | [-5.74pp, +8.33pp] | [-5.74pp, +8.33pp] | 0.074 | 0–0/12 | False | 源门预定 |
| SPARSE | F1_oracle_low__pessimistic0.5 | 81.12% | [-3.59pp, +7.99pp] | False | [-2.08pp, +10.42pp] | [-2.08pp, +10.42pp] | 0.296 | 0–0/12 | False | 敏感性 κ=.5 |
| SPARSE | F1_oracle_low__conservative | 79.69% | [-1.46pp, +3.32pp] | True | [+0.00pp, +3.12pp] | [+0.00pp, +3.12pp] | 0.074 | 0–0/12 | True | 源门预定 |
| SPARSE | F2_PCA2__argmax | 81.55% | [-2.41pp, +7.52pp] | False | [-3.65pp, +8.33pp] | [-3.65pp, +8.33pp] | 0.173 | 0–0/12 | False | 源门预定 |
| SPARSE | F2_PCA2__pessimistic1 | 82.68% | [-1.48pp, +8.86pp] | True | [+1.56pp, +14.06pp] | [+1.56pp, +14.06pp] | 0.556 | 0–0/12 | True | 源门预定 |
| SPARSE | F2_PCA2__pessimistic0.5 | 82.81% | [-1.37pp, +9.02pp] | True | [+2.07pp, +14.58pp] | [+2.07pp, +14.58pp] | 0.593 | 0–0/12 | True | 敏感性 κ=.5 |
| SPARSE | F2_PCA2__conservative | 81.77% | [-1.04pp, +6.55pp] | True | [+1.04pp, +8.33pp] | [+1.04pp, +8.33pp] | 0.333 | 0–0/12 | True | 源门预定 |
| SPARSE | F2_oracle_low__argmax | 80.99% | [-5.81pp, +9.63pp] | False | [-5.73pp, +8.85pp] | [-5.73pp, +8.85pp] | 0.123 | 0–0/12 | False | 源门预定 |
| SPARSE | F2_oracle_low__pessimistic1 | 80.60% | [-5.32pp, +8.33pp] | False | [-5.73pp, +5.21pp] | [-5.73pp, +5.21pp] | -0.037 | 0–0/12 | False | 源门预定 |
| SPARSE | F2_oracle_low__pessimistic0.5 | 81.51% | [-4.63pp, +9.38pp] | False | [-6.77pp, +6.25pp] | [-6.77pp, +6.25pp] | -0.037 | 0–1/12 | False | 敏感性 κ=.5 |
| SPARSE | F2_oracle_low__conservative | 80.86% | [-1.59pp, +5.55pp] | True | [+0.00pp, +5.73pp] | [+0.00pp, +5.73pp] | 0.185 | 0–0/12 | True | 源门预定 |
| SPARSE | F4__argmax | 78.91% | [+0.00pp, +0.00pp] | True | [+0.00pp, +0.00pp] | [+0.00pp, +0.00pp] | 0.000 | 0–0/12 | True | 源门预定 |

E1同时CI family包含本测试床的全部实际评价方法及coverage臂；5000次同物理state共享成组bootstrap，3fit平均。E2仅既定必要controller（融合3、真实2），CI描述性；gap闭合比例不clip，bootstrap分母不正时CI为NA，次数见机器结果。E3范围保留unknown，合格要求possible E3=0。真实床不符合源门但同架构生成的其他固定policy只作辅助，最后列明确标记；不能以真实终点选择policy。

| 覆盖 | arm/规则 | D1凸包距离/损失Pearson | D2 selected Q高估上界 | D3选η1 | D3选η8 | 源η1/η8请求数 |
| --- | --- | --- | --- | --- | --- | --- |
| DENSE | B0__argmax | NA | -12.40pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | B1__argmax | 0.647 | +5.17pp | 12.85% | 2.95% | 5312/5312 |
| DENSE | B1__pessimistic1 | 0.559 | +4.75pp | 13.54% | 4.43% | 5312/5312 |
| DENSE | B1__pessimistic0.5 | 0.565 | +4.97pp | 14.32% | 4.43% | 5312/5312 |
| DENSE | B1__conservative | 0.247 | -14.42pp | 0.78% | 0.26% | 5312/5312 |
| DENSE | B2__argmax | 0.975 | +12.99pp | 9.81% | 7.73% | 5312/5312 |
| DENSE | B2__pessimistic1 | 0.966 | +7.60pp | 12.50% | 2.34% | 5312/5312 |
| DENSE | B2__pessimistic0.5 | 0.936 | +6.82pp | 10.42% | 5.21% | 5312/5312 |
| DENSE | B2__conservative | 0.934 | -12.02pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA1__argmax | NA | -8.24pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA1__pessimistic1 | NA | -8.57pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA1__pessimistic0.5 | NA | -8.92pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA1__conservative | NA | -9.06pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA2__argmax | 0.068 | -8.58pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA2__pessimistic1 | 0.180 | -8.94pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA2__pessimistic0.5 | 0.437 | -9.20pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA2__conservative | 0.258 | -9.62pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA3__argmax | 0.082 | -9.16pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA3__pessimistic1 | -0.007 | -9.19pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA3__pessimistic0.5 | 0.028 | -9.20pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_PCA3__conservative | -0.322 | -9.76pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_oracle_low__argmax | NA | -0.38pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_oracle_low__pessimistic1 | NA | -0.73pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_oracle_low__pessimistic0.5 | NA | -1.02pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F1_oracle_low__conservative | NA | -1.76pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_PCA2__argmax | 0.055 | -6.57pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_PCA2__pessimistic1 | 0.139 | -6.97pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_PCA2__pessimistic0.5 | 0.040 | -6.98pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_PCA2__conservative | -0.011 | -7.43pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_oracle_low__argmax | NA | -0.63pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_oracle_low__pessimistic1 | NA | -0.90pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_oracle_low__pessimistic0.5 | NA | -1.25pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F2_oracle_low__conservative | NA | +0.67pp | 0.00% | 0.00% | 5312/5312 |
| DENSE | F4__argmax | 0.311 | 非Q模型 | 0.00% | 0.00% | 5312/5312 |
| SPARSE | B0__argmax | NA | -20.40pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | B1__argmax | 0.624 | -0.41pp | 11.28% | 3.73% | 3584/3584 |
| SPARSE | B1__pessimistic1 | 0.575 | +0.73pp | 15.36% | 1.30% | 3584/3584 |
| SPARSE | B1__pessimistic0.5 | 0.577 | +1.37pp | 15.62% | 0.52% | 3584/3584 |
| SPARSE | B1__conservative | 0.534 | -20.59pp | 2.34% | 0.00% | 3584/3584 |
| SPARSE | B2__argmax | 0.210 | -3.40pp | 1.22% | 6.68% | 3584/3584 |
| SPARSE | B2__pessimistic1 | 0.048 | -7.60pp | 0.00% | 3.91% | 3584/3584 |
| SPARSE | B2__pessimistic0.5 | 0.081 | -7.14pp | 0.00% | 3.65% | 3584/3584 |
| SPARSE | B2__conservative | -0.464 | -21.04pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA1__argmax | NA | -16.99pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA1__pessimistic1 | NA | -17.79pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA1__pessimistic0.5 | NA | -17.61pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA1__conservative | NA | -18.08pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA2__argmax | 0.333 | -20.36pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA2__pessimistic1 | 0.452 | -21.34pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA2__pessimistic0.5 | 0.447 | -21.27pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA2__conservative | NA | -21.77pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA3__argmax | 0.328 | -20.03pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA3__pessimistic1 | 0.518 | -21.00pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA3__pessimistic0.5 | 0.644 | -20.83pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_PCA3__conservative | NA | -21.21pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_oracle_low__argmax | 0.083 | -29.15pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_oracle_low__pessimistic1 | -0.266 | -29.30pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_oracle_low__pessimistic0.5 | -0.096 | -30.25pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F1_oracle_low__conservative | 0.080 | -30.23pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_PCA2__argmax | 0.367 | -22.75pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_PCA2__pessimistic1 | 0.265 | -24.85pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_PCA2__pessimistic0.5 | 0.272 | -24.82pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_PCA2__conservative | 0.299 | -27.04pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_oracle_low__argmax | 0.343 | -28.08pp | 2.08% | 0.00% | 3584/3584 |
| SPARSE | F2_oracle_low__pessimistic1 | -0.022 | -29.37pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_oracle_low__pessimistic0.5 | 0.076 | -29.79pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F2_oracle_low__conservative | 0.562 | -31.84pp | 0.00% | 0.00% | 3584/3584 |
| SPARSE | F4__argmax | NA | 非Q模型 | 0.00% | 0.00% | 3584/3584 |

D1全部最近邻/凸包距离、solver状态、Pearson/Spearman在D1/D1_correlations。不同折的source normalization不同，controller与state非独立，相关性不等于因果。D2有unknown保守区间与valid-Q估计，F4概率是regime概率，不能冒充Q，故D2不适用。D3每个η头的请求/valid/unknown/源success/controller覆盖都保留。选择稀有好head不能自动解释为缺标签：请求分配平衡，而源成功率可能不平衡。
### 真实Flow LOCO

H_method 操作判定：**SUPPORTED: operational repair eliminates harm**。安全修复数5，正迁移候选0。每个controller/fold/fit的选择、unknown边界、source prior、target fixed及η频率保存在对应results.json的rows。

| 覆盖 | arm/规则 | 成功下界 | E1 vs R0同时CI | E1过门 | E2 vs源prior描述CI | E2 vsR0描述CI | gap闭合比例 | E3确定–可能 | 安全合格 | 规则证据类别 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCO | B0__argmax | 54.89% | [+0.00pp, +0.00pp] | True | [-0.20pp, +2.99pp] | [+0.00pp, +0.00pp] | 0.046 | 1–2/7 | False | 源门预定 |
| LOCO | B1__argmax | 58.97% | [+0.94pp, +7.43pp] | True | [+1.17pp, +10.61pp] | [+0.13pp, +8.92pp] | 0.204 | 1–1/7 | False | 源门预定 |
| LOCO | B1__pessimistic1 | 60.77% | [+2.62pp, +9.54pp] | True | [+5.66pp, +16.21pp] | [+4.36pp, +14.84pp] | 0.382 | 1–1/7 | False | 源门预定 |
| LOCO | B1__pessimistic0.5 | 60.83% | [+2.58pp, +9.69pp] | True | [+5.66pp, +16.21pp] | [+4.36pp, +14.84pp] | 0.382 | 1–1/7 | False | 敏感性 κ=.5 |
| LOCO | B1__conservative | 59.88% | [+2.15pp, +8.20pp] | True | [+0.00pp, +0.00pp] | [-2.99pp, +0.20pp] | 0.000 | 0–0/7 | True | 源门预定 |
| LOCO | F1_PCA1__argmax | 57.74% | [+0.84pp, +5.05pp] | True | [+0.20pp, +2.67pp] | [-1.37pp, +1.56pp] | 0.051 | 0–0/7 | True | 辅助；源门未过 |
| LOCO | F1_PCA1__pessimistic1 | 57.70% | [+0.47pp, +5.23pp] | True | [+0.20pp, +2.93pp] | [-1.50pp, +1.76pp] | 0.049 | 0–0/7 | True | 辅助；源门未过 |
| LOCO | F1_PCA1__pessimistic0.5 | 58.87% | [+1.43pp, +6.68pp] | True | [+0.00pp, +1.56pp] | [-2.47pp, +0.98pp] | 0.021 | 0–0/7 | True | 敏感性 κ=.5 |
| LOCO | F1_PCA1__conservative | 60.10% | [+2.45pp, +8.34pp] | True | [+0.00pp, +0.00pp] | [-2.99pp, +0.20pp] | 0.000 | 0–0/7 | True | 源门预定 |
| LOCO | F1_PCA2__argmax | 58.39% | [+1.32pp, +5.95pp] | True | [+0.72pp, +4.23pp] | [-0.59pp, +2.73pp] | 0.086 | 0–0/7 | True | 辅助；源门未过 |
| LOCO | F1_PCA2__pessimistic1 | 58.48% | [+0.64pp, +6.75pp] | True | [-2.15pp, +4.88pp] | [-3.12pp, +3.06pp] | 0.049 | 0–0/7 | True | 辅助；源门未过 |
| LOCO | F1_PCA2__pessimistic0.5 | 59.38% | [+1.71pp, +7.55pp] | True | [+0.78pp, +5.47pp] | [-0.07pp, +3.52pp] | 0.104 | 0–0/7 | True | 敏感性 κ=.5 |
| LOCO | F1_PCA2__conservative | 60.10% | [+2.45pp, +8.34pp] | True | [+0.00pp, +0.00pp] | [-2.99pp, +0.20pp] | 0.000 | 0–0/7 | True | 源门预定 |
| LOCO | F1_PCA3__argmax | 58.65% | [+1.43pp, +6.30pp] | True | [+0.13pp, +2.93pp] | [-1.37pp, +1.69pp] | 0.053 | 0–0/7 | True | 辅助；源门未过 |
| LOCO | F1_PCA3__pessimistic1 | 59.21% | [+1.71pp, +7.16pp] | True | [-1.37pp, +3.91pp] | [-2.60pp, +2.15pp] | 0.042 | 0–0/7 | True | 辅助；源门未过 |
| LOCO | F1_PCA3__pessimistic0.5 | 60.27% | [+2.75pp, +8.29pp] | True | [+0.00pp, +2.73pp] | [-1.69pp, +1.50pp] | 0.042 | 0–0/7 | True | 敏感性 κ=.5 |
| LOCO | F1_PCA3__conservative | 60.21% | [+2.45pp, +8.56pp] | True | [+0.00pp, +0.00pp] | [-2.99pp, +0.20pp] | 0.000 | 0–0/7 | True | 源门预定 |
| LOCO | F2_PCA2__argmax | 59.38% | [+2.00pp, +7.07pp] | True | [+6.25pp, +16.28pp] | [+5.27pp, +14.65pp] | 0.398 | 1–1/7 | False | 源门预定 |
| LOCO | F2_PCA2__pessimistic1 | 58.31% | [+0.60pp, +6.39pp] | True | [+4.69pp, +15.82pp] | [+3.51pp, +14.32pp] | 0.361 | 2–2/7 | False | 源门预定 |
| LOCO | F2_PCA2__pessimistic0.5 | 58.82% | [+1.02pp, +7.07pp] | True | [+5.86pp, +16.41pp] | [+4.56pp, +15.30pp] | 0.396 | 2–2/7 | False | 敏感性 κ=.5 |
| LOCO | F2_PCA2__conservative | 61.27% | [+3.09pp, +9.96pp] | True | [+4.30pp, +13.48pp] | [+3.06pp, +12.11pp] | 0.312 | 1–1/7 | False | 源门预定 |
| LOCO | F4__argmax | 60.21% | [+2.45pp, +8.56pp] | True | [+0.00pp, +0.00pp] | [-2.99pp, +0.20pp] | 0.000 | 0–0/7 | True | 源门预定 |

E1同时CI family包含本测试床的全部实际评价方法及coverage臂；5000次同物理state共享成组bootstrap，3fit平均。E2仅既定必要controller（融合3、真实2），CI描述性；gap闭合比例不clip，bootstrap分母不正时CI为NA，次数见机器结果。E3范围保留unknown，合格要求possible E3=0。真实床不符合源门但同架构生成的其他固定policy只作辅助，最后列明确标记；不能以真实终点选择policy。

| 覆盖 | arm/规则 | D1凸包距离/损失Pearson | D2 selected Q高估上界 | D3选η1 | D3选η8 | 源η1/η8请求数 |
| --- | --- | --- | --- | --- | --- | --- |
| LOCO | B0__argmax | NA | +1.64pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | B1__argmax | 0.554 | +0.70pp | 2.75% | 0.37% | 1536/1536 |
| LOCO | B1__pessimistic1 | 0.418 | -3.20pp | 0.45% | 0.00% | 1536/1536 |
| LOCO | B1__pessimistic0.5 | 0.412 | -2.82pp | 0.45% | 0.00% | 1536/1536 |
| LOCO | B1__conservative | 0.767 | -5.06pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA1__argmax | NA | -3.67pp | 0.00% | 0.37% | 1536/1536 |
| LOCO | F1_PCA1__pessimistic1 | NA | -4.12pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA1__pessimistic0.5 | NA | -5.18pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA1__conservative | NA | -6.92pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA2__argmax | -0.139 | -2.53pp | 0.00% | 0.30% | 1536/1536 |
| LOCO | F1_PCA2__pessimistic1 | -0.062 | -3.47pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA2__pessimistic0.5 | 0.063 | -4.17pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA2__conservative | -0.067 | -5.27pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA3__argmax | 0.567 | -7.22pp | 0.00% | 0.67% | 1536/1536 |
| LOCO | F1_PCA3__pessimistic1 | 0.632 | -8.52pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F1_PCA3__pessimistic0.5 | 0.737 | -9.43pp | 0.00% | 0.22% | 1536/1536 |
| LOCO | F1_PCA3__conservative | 0.719 | -9.43pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F2_PCA2__argmax | 0.344 | -4.82pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F2_PCA2__pessimistic1 | 0.276 | -4.30pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F2_PCA2__pessimistic0.5 | 0.299 | -4.62pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F2_PCA2__conservative | 0.312 | -8.55pp | 0.00% | 0.00% | 1536/1536 |
| LOCO | F4__argmax | -0.084 | 非Q模型 | 0.00% | 0.00% | 1536/1536 |

D1全部最近邻/凸包距离、solver状态、Pearson/Spearman在D1/D1_correlations。不同折的source normalization不同，controller与state非独立，相关性不等于因果。D2有unknown保守区间与valid-Q估计，F4概率是regime概率，不能冒充Q，故D2不适用。D3每个η头的请求/valid/unknown/源success/controller覆盖都保留。选择稀有好head不能自动解释为缺标签：请求分配平衡，而源成功率可能不平衡。


| 测试床 | 安全修复规则数（含敏感性） | 安全修复涉及架构 |
| --- | --- | --- |
| 融合 | 18 | B2, F1_PCA1, F1_PCA2, F1_PCA3, F1_oracle_low, F2_PCA2, F2_oracle_low, F4 |
| 真实 | 5 | B1, F1_PCA1, F1_PCA2, F1_PCA3, F4 |

| 测试床 | 主要正迁移候选（DEVELOPMENT） | κ=.5敏感性候选 |
| --- | --- | --- |
| 融合 | [{'coverage': 'SPARSE', 'method': 'F2_PCA2__pessimistic1'}, {'coverage': 'SPARSE', 'method': 'F2_PCA2__conservative'}] | [{'coverage': 'DENSE', 'method': 'F2_oracle_low__pessimistic0.5'}, {'coverage': 'SPARSE', 'method': 'F2_PCA2__pessimistic0.5'}] |
| 真实 | [] | [] |

全部图表在figures/：blend/real_E1,E2,E3以及D1,D2,D3各PNG/PDF，共24个文件；FIGURE_INDEX.json。图表使用固定旧matplotlib依赖只读加载，font cache写入本目录。

## D. Hypothesis Evaluation

| 假设 | 判定 |
| --- | --- |
| H_method_harm | SUPPORTED: operational repair eliminates harm |
| H_positive_blend_transfer | SUPPORTED DEVELOPMENT |
| H_positive_real_Flow_transfer | INCONCLUSIVE: no primary eligible arm established positive real transfer |
| H_high_dimension_is_unique_rootcause | INCONCLUSIVE: repair interventions combine representation, shrinkage, uncertainty and fallback; no unique causal identification |
| H_statewise_generalization | INCONCLUSIVE: not a prospective state-selection confirmation |

上述H_method按用户给定“安全修复+基线灾难”映射判定，是方法偏置解释伤害的操作性证据。单纯回退固定η成功不等于critic已学会controller关系。F1同时降维和移除descriptor逐state变化；F2同时缩小网络、强weight decay及限制残差；F3三seed分歧不是校准的Bayesian不确定性。不能把这些干预当成严格证明高维参数化是唯一因果机制。

## E. Controller–Q Relationship and Transferability

明确分开：Q拟合/选择高估(D2)、给定controller的η选择(E1/E2)、controller级regime/prior校准(F4/回退)。恢复安全可支持修复controller通路的归纳偏置，只有E2同时优于src prior与R0才算正迁移候选。全部修复性能由原终点seed评价，没有目标标签offset/调参。源regime信息与特权融合组成不是新真实Flow的oracle输入。必要controller只有3不再被包装成round5旧主门通过；本轮E2按预定规则仅作DEVELOPMENT描述。

## F. Limitations and Next Steps

范围为Ring、固定geometry、FF/H700/K16、给定真实Flow与合成速度融合族；离散η不能刻画完整连续success basin。本轮全部数据早已打开，multiple arms固定并用simultaneous E1保护安全检验，仍非独立确认。真实LOCO仅在该折优化中留出controller，方法资格来自含这些原Flow的旧融合源面板，不能称完全独立新controller确认。未发现正迁移不证明其不存在；恢复安全也不排除覆盖/表示不足。D1相关不因果；D2末端标签噪声/unknown；D3计数不等于有效controller覆盖。源验证只含原Flow，其NLL最优step未必优化融合selection；本轮不打开终点调参。oracle-low是特权压缩regime表示，并非完整真坐标。三fit置信区间按物理state抽样，不包括对新controller总体或大量训练seed的随机抽样不确定性。

最小前瞻方案（未执行）：先从本轮候选冻结一个模型/规则；至少两个新真实Flow controller，部署只用无标签静态表示，不读目标TRAIN regime；每controller64个全新自然state×K16×Q8=16,384闭环请求（先DB精确复用审计、另行成本/预注册），seed0–3仅选独立fixed参照与门，4–7仅评价；source checkpoint冻结，不重新选policy。若只存在安全无收益修复，则下一步是源数据内CV检查controller级regime可辨识性与NLL/selection目标偏移，使用现有标签，不直接扩rollout。该建议不在本轮自动生成数据。

## G. Reproducibility

最终全库hash审计最初FAIL，原记录保留为FINAL_AUDIT_INITIAL_FAIL.json。只读调查发现本轮期间Drone3D独立任务导入66条source行标签，并登记其独立experiment；本目录experiment/source行均0。DB_CONCURRENT_ACTIVITY.json保留基线/当前hash及外部来源。无法证明共享DB每个字节变化的完整归属，故全库不变性标NOT VERIFIED；本轮实际使用的737个历史NPZ/checkpoint/代码文件哈希全一致，新增脚本不写DB，最后调查仅以mode=ro、immutable、query_only执行SELECT。最终审计以限定范围PASS_WITH_NOTES交付，不能伪称共享DB全局未变化。

Python /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；工作目录/home/zhihan/research/Basin_C1；git a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e；JAX x64=false。随机拟合17/23/41，bootstrap5000、种子20261013061/62，PCA/logistic deterministic。数据源hash和DB只读hash、checkpointSHA、source normalization/PCA参数、每seed logits均在本目录。

| 文件 | UTC | sha256 |
| --- | --- | --- |
| PREREGISTRATION.md | 2026-10-13T03:12:43.349563+00:00 | 95ad710301e9cd2db86a7d14ecdf9fea081af0ab3fdbefbb605b8792b8cd36f3 |
| PREREGISTRATION_REPLAY_NUMERICS_ADDENDUM.md | 2026-10-13T03:20:20.197950+00:00 | c70514a08e5d42da51344b665b77959d8544e883164a657b04d2e7b6d8eea809 |
| PREREGISTRATION_RESOURCE_ADDENDUM.md | 2026-10-13T03:16:57.964020+00:00 | 13afd05fdcb25b543fac78f6a69f42aa384adbf5f4216f8178665f6b25fb686d |

```bash
cd /home/zhihan/research/Basin_C1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/verify_statistics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/evaluate.py blend
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/evaluate.py real
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/plots.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/report.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/audit.py
```

重训通过Slurm，参见submit.py与各dispatch commands；完成checkpoint缓存不重复优化，禁止改终点seed或读取目标标签构造输入。最终审计FINAL_AUDIT.json独立验证收据、历史输入未修改、全部已执行模型/图表/结果；全库不变性与本任务零写入分开登记。失败记录保留，未解决的审计FAIL不可标交付完成。

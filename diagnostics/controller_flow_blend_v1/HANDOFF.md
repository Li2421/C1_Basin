# 独立Handoff：Controller Flow Blend Round5

当前UTC：2026-10-13T03:00:57.431150+00:00；任务状态 **COMPLETE**。本文件可脱离对话读取，未完成阶段明确标注。

## 背景和研究目标

目标：从物理state和controller查询学会Q(s,C,η)，在未见controller下获得η选择闭环成功收益。上一轮round4真实Flow LOCO必要折仅2/7，其留出controller经验η13、源priorη10；增加7源没有增加η13样例。独立Stage C的R3 gain+5.21pp CI[-.66,11.08]，错配差+7.51pp CI[2.53,12.49]；未过联合选择门。因此无法唯一定位覆盖不足、表示不足或方法缺陷。

## 实际实施和已完成证据

Stage0已执行，读14个原Flow共享128state平衡8-headQ4标签、静态特征和冻结模型；复用57,344训练seed格，零新增rollout/critic拟合。Stage1实现world-field Flow convex blend，A=base_u/v10、B=wide/v9，4pairs×λ.25/.5/.75。初值同ξ，两chart速度在共同h处转换世界坐标融合，保留FF/OrthoFlow3/投影/H700。端点静态公式及原runtime检查误差0，禁止env.step；x64=false。stage_1/protocol.json冻结4,608K4Q4请求；preflight全部确切缺失、登记experiment先于merge。98timing包含在pilot标签内。

| 阶段 | 当前状态 |
| --- | --- |
| Stage0 | COMPLETE EXPLORATORY |
| Stage1 | 完成；门通过 |
| Stage2 | 完成 |

14Flow经验regimeη10=9、η13=2、η2=2、η4=1。R3 NN/logistic LOO准确率64.3%=多数类；R4_action NN35.7%，logistic64.3%；组合也是64.3%。分类只对14个controller均值描述，无调参，不支持源外独立确认。

重要事实纠正：fold4冻结R3 controller prior实际选η4，3fit都如此，而不是摘要的η13。它在旧测试面板很好；fold1 prior[4,10,10]失败。精确index分类与选择success不可混同。旧报告中也存在Stage A组合逐state−modelprior point+2.21pp CI[-.01,4.44]，因此“所有point都不高于prior”过强；没有可靠正的state残差选择证据。

## 本轮最重要结果

| pair | λ | η2下界 | η4下界 | η10下界 | η13下界 | 13−10配对CI | 明确赢家 | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.25 | 0.00% | 27.08% | 28.12% | 52.08% | [+3.12pp, +43.75pp] | 13 | 3 |
| 0 | 0.5 | 1.04% | 46.88% | 47.92% | 67.71% | [+0.00pp, +38.54pp] | 13 | 1 |
| 0 | 0.75 | 10.42% | 42.71% | 61.46% | 45.83% | [-34.38pp, +4.17pp] | 10 | 2 |
| 1 | 0.25 | 0.00% | 59.38% | 92.71% | 98.96% | [+0.00pp, +14.58pp] | 13 | 1 |
| 1 | 0.5 | 0.00% | 95.83% | 82.29% | 96.88% | [+4.17pp, +21.88pp] | 13 | 2 |
| 1 | 0.75 | 0.00% | 98.96% | 93.75% | 92.71% | [-12.50pp, +10.42pp] | 10 | 0 |
| 2 | 0.25 | 10.42% | 51.04% | 73.96% | 86.46% | [-6.25pp, +31.25pp] | 13 | 0 |
| 2 | 0.5 | 0.00% | 60.42% | 90.62% | 86.46% | [-19.79pp, +9.38pp] | 10 | 0 |
| 2 | 0.75 | 1.04% | 80.21% | 86.46% | 97.92% | [+2.08pp, +22.92pp] | 13 | 0 |
| 3 | 0.25 | 34.38% | 98.96% | 93.75% | 97.92% | [-4.17pp, +13.54pp] | 13 | 0 |
| 3 | 0.5 | 29.17% | 100.00% | 100.00% | 98.96% | [-3.12pp, +0.00pp] | 10 | 0 |
| 3 | 0.75 | 16.67% | 100.00% | 100.00% | 94.79% | [-13.54pp, +0.00pp] | 10 | 0 |

| pair | η13胜任λ | η10胜任λ | 通过内部λ门 |
| --- | --- | --- | --- |
| 0 | [0.25, 0.5] | [0.75] | True |
| 1 | [0.25, 0.5] | [0.75] | True |
| 2 | [0.25, 0.75] | [0.5] | True |
| 3 | [0.25] | [0.5, 0.75] | True |

12个内部controller共4608请求；valid=4599、unknown=9。内部λ门4/4，要求≥3/4。锚点复用768个K4 seed格；每η仅12/24state，详见stage_1/results.json.original_endpoint_partial_curves。

98次pilot平均10.39s、median 9.37s、p90 22.84s；全Stage1初估13.30CPU小时/0.55h@24CPU，Stage2完整标签初估177.38CPU小时/7.39h@24CPU。实际全量尾部、特征、训练、队列与合并另计，不能视作任务完成时间保证。

Stage2正式成本门：复用2,304个pilot seed格（2,299 valid、5 unknown不重跑），新增59136个完整H700 episode及832个短probe，共59968、cap65,000。按全pilot平均10.14s估计166.53CPU小时，@24CPU=6.94h、@18CPU=9.25h；K16包含更多可能慢的η，仍非完成时间保证。

实际全研究新增63,744次完整闭环请求及832次短probe，总计64,576次。Stage2已合并61,440个格子标签（含2,304个pilot缓存）：有效61,009、unknown 431、缺失/隔离均0。probe有效统计831、unknown 1；R5整体未训练，保留失败与partial trace。历史TRAIN/验证/Stage0旧test分别57,344/21,504/16,384 seed格，共95,232；2,304个本轮pilot复用不计入历史独立标签数。

### 实际训练与部署设置

新增32自然test、16probe初态均做全DB及全部历史protocol物理hash去重；旧96train为DEVELOPMENT。每原Flow复用128state8-headQ4，共57,344TRAIN seed格；P1 14controller24stateK16Q4共21,504验证seed格复用。14原Flow始终为源，融合源按LOPO/LOAO排除目标。模型physical23+pad513→64/64/16 MLP，3fitseeds17/23/41，AdamW/12,000steps/NLL选步沿用round4；源controller均匀抽样，source-only归一化。源验证只含原Flow，没有融合源验证标签，须注意源验证目标与融合任务的偏移。

实际独立优化84次fit、144个per-task checkpoint引用；SPARSE相同源训练重复折直接复用SHA一致冻结模型。R0/R3/R4_action/R_oracle/R5输入网络参数预算相同；R_oracle已知[λ,R3_A,R3_B]是特权坐标，原Flow自融合[0,R3_C,R3_C]。R5每C用16probe×2η×200step，160个无终局标签统计特征，需要模拟器，可部署假设不同。

probe已执行832次，其中831次提供完整统计、1次数值失败。按事前分支，全部12个R5预测task明确NOT EXECUTED；另外48个task完成，实际84次独立fit。partial trace/unknown保留，不删controller/probe、不填0、不重跑。无标签probe在DB仅登记UNLABELED_SHORT_PROBE source_file与独立UID/manifest，不制造success=0，不修改schema。见probes/POSTFLIGHT.json、R5_FEATURES.npz和RAW_TRACE_INDEX.json。

### 必要性门（仅seed0–1；不读终点）

| 覆盖 | pair/fold | 目标C | η_src | targetCFη | L保守下界 | 95%CI | 必要 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | 0 | blend_pair0_lambda25 | 10 | [13, 13] | +7.81pp | [-9.38pp, +23.44pp] | False |
| DENSE | 0 | blend_pair0_lambda50 | 10 | [13, 13] | +12.50pp | [-7.81pp, +31.25pp] | False |
| DENSE | 0 | blend_pair0_lambda75 | 10 | [10, 10] | +0.00pp | [+0.00pp, +0.00pp] | False |
| SPARSE | 0 | blend_pair0_lambda25 | 10 | [13, 13] | +7.81pp | [-9.38pp, +23.44pp] | False |
| SPARSE | 0 | blend_pair0_lambda50 | 10 | [13, 13] | +12.50pp | [-7.81pp, +31.25pp] | False |
| SPARSE | 0 | blend_pair0_lambda75 | 10 | [10, 10] | +0.00pp | [+0.00pp, +0.00pp] | False |
| DENSE | 1 | blend_pair1_lambda25 | 10 | [13, 13] | +3.12pp | [-4.69pp, +12.50pp] | False |
| DENSE | 1 | blend_pair1_lambda50 | 10 | [4, 4] | +14.06pp | [+4.69pp, +25.00pp] | True |
| DENSE | 1 | blend_pair1_lambda75 | 10 | [4, 4] | +1.56pp | [-3.12pp, +7.81pp] | False |
| SPARSE | 1 | blend_pair1_lambda25 | 10 | [13, 13] | +3.12pp | [-4.69pp, +12.50pp] | False |
| SPARSE | 1 | blend_pair1_lambda50 | 10 | [4, 4] | +14.06pp | [+4.69pp, +25.00pp] | True |
| SPARSE | 1 | blend_pair1_lambda75 | 10 | [4, 4] | +1.56pp | [-3.12pp, +7.81pp] | False |
| DENSE | 2 | blend_pair2_lambda25 | 10 | [13, 13] | +35.94pp | [+18.75pp, +51.56pp] | True |
| DENSE | 2 | blend_pair2_lambda50 | 10 | [13, 13] | +15.62pp | [+4.69pp, +26.56pp] | True |
| DENSE | 2 | blend_pair2_lambda75 | 10 | [13, 13] | +9.38pp | [-1.56pp, +21.88pp] | False |
| SPARSE | 2 | blend_pair2_lambda25 | 10 | [13, 13] | +35.94pp | [+18.75pp, +51.56pp] | True |
| SPARSE | 2 | blend_pair2_lambda50 | 10 | [13, 13] | +15.62pp | [+4.69pp, +26.56pp] | True |
| SPARSE | 2 | blend_pair2_lambda75 | 10 | [13, 13] | +9.38pp | [-1.56pp, +21.88pp] | False |
| DENSE | 3 | blend_pair3_lambda25 | 10 | [4, 4] | +4.69pp | [-4.69pp, +14.06pp] | False |
| DENSE | 3 | blend_pair3_lambda50 | 10 | [10, 10] | +0.00pp | [+0.00pp, +0.00pp] | False |
| DENSE | 3 | blend_pair3_lambda75 | 10 | [10, 10] | +0.00pp | [+0.00pp, +0.00pp] | False |
| SPARSE | 3 | blend_pair3_lambda25 | 10 | [4, 4] | +4.69pp | [-4.69pp, +14.06pp] | False |
| SPARSE | 3 | blend_pair3_lambda50 | 10 | [10, 10] | +0.00pp | [+0.00pp, +0.00pp] | False |
| SPARSE | 3 | blend_pair3_lambda75 | 10 | [10, 10] | +0.00pp | [+0.00pp, +0.00pp] | False |

### 覆盖×表示主结果（仅新state seed2–3）

| 覆盖 | 表示 | 必要C数 | vs源prior同时CI | vs最近源同时CI | vsR0同时CI | matched−mismatch同时CI | 联合门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | R3 | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| DENSE | R4_action | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| DENSE | R_oracle | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| DENSE | R5_behavior | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| SPARSE | R3 | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| SPARSE | R4_action | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| SPARSE | R_oracle | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |
| SPARSE | R5_behavior | 3 | 未检验 | 未检验 | 未检验 | 未检验 | INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS |

每表示内部四项配对max-T同时CI、5000次shared-state cluster bootstrap；3fit先平均。主检验必要C≥4；缺输入不伪造结果。跨表示不宣称整个研究familywise error已经控制。Q2有标签噪声，不能代替success选择证据。

### DENSE−SPARSE直接覆盖效应

| 目标集合 | C数 | 表示 | 配对差 | 跨表示同时CI |
| --- | --- | --- | --- | --- |
| union | 3 | R0 | +0.17pp | [-0.26pp, +0.60pp] |
| union | 3 | R3 | -9.90pp | [-17.71pp, -2.08pp] |
| union | 3 | R4_action | +1.39pp | [-2.17pp, +4.95pp] |
| union | 3 | R_oracle | -26.91pp | [-34.23pp, -19.59pp] |
| intersection | 3 | R0 | +0.17pp | [-0.26pp, +0.60pp] |
| intersection | 3 | R3 | -9.90pp | [-17.71pp, -2.08pp] |
| intersection | 3 | R4_action | +1.39pp | [-2.17pp, +4.95pp] |
| intersection | 3 | R_oracle | -26.91pp | [-34.23pp, -19.59pp] |
| all | 12 | R0 | +0.87pp | [-0.68pp, +2.42pp] |
| all | 12 | R3 | -1.65pp | [-4.06pp, +0.76pp] |
| all | 12 | R4_action | +2.08pp | [-0.97pp, +5.14pp] |
| all | 12 | R_oracle | -10.59pp | [-14.06pp, -7.12pp] |

主union由两臂discovery必要门定义，两个臂在同一目标/物理state上比较；intersection/all为敏感性。额外源同时增加label量，不把这个实验当纯无混杂的controller覆盖因果估计。

### Controller prior与逐state增益

| 覆盖 | 表示 | state选择success | 模型controller固定prior success | state残差保守差 | 模型priorη/3fits |
| --- | --- | --- | --- | --- | --- |
| DENSE | R0 | 78.82% | 78.65% | +0.17pp | [[10, 10, 10], [10, 10, 10], [10, 10, 10]] |
| DENSE | R3 | 32.29% | 35.59% | -3.30pp | [[13, 8, 1], [4, 4, 4], [1, 4, 1]] |
| DENSE | R4_action | 79.69% | 82.29% | -2.60pp | [[13, 13, 13], [10, 10, 10], [10, 10, 10]] |
| DENSE | R_oracle | 42.01% | 45.31% | -3.30pp | [[8, 4, 1], [4, 4, 4], [4, 4, 4]] |
| SPARSE | R0 | 78.65% | 78.65% | +0.00pp | [[10, 10, 10], [10, 10, 10], [10, 10, 10]] |
| SPARSE | R3 | 42.01% | 33.68% | +8.33pp | [[1, 1, 1], [13, 4, 13], [4, 4, 1]] |
| SPARSE | R4_action | 78.30% | 79.86% | -1.56pp | [[13, 10, 10], [10, 10, 10], [10, 10, 10]] |
| SPARSE | R_oracle | 68.75% | 72.57% | -3.82pp | [[8, 13, 13], [13, 4, 10], [13, 4, 10]] |

prior只用held controller在96旧train初态的静态/无标签行为输入预测均值，不用目标标签。以上分解描述性，与主门分开；精确argmax index与闭环goodη不等价。

### 目标少样本校准参照（使用目标标签）

| 覆盖 | 表示 | k个目标state | fullrollout数 | 校准固定η success | critic+offset success |
| --- | --- | --- | --- | --- | --- |
| DENSE | R0 | 1 | 32 | 83.33% | 83.33% |
| DENSE | R0 | 2 | 64 | 83.33% | 84.72% |
| DENSE | R0 | 4 | 128 | 83.33% | 85.07% |
| DENSE | R3 | 1 | 32 | 83.33% | 83.33% |
| DENSE | R3 | 2 | 64 | 83.33% | 84.38% |
| DENSE | R3 | 4 | 128 | 83.33% | 84.72% |
| DENSE | R4_action | 1 | 32 | 83.33% | 83.33% |
| DENSE | R4_action | 2 | 64 | 83.33% | 84.38% |
| DENSE | R4_action | 4 | 128 | 83.33% | 84.38% |
| DENSE | R_oracle | 1 | 32 | 83.33% | 83.33% |
| DENSE | R_oracle | 2 | 64 | 83.33% | 84.20% |
| DENSE | R_oracle | 4 | 128 | 83.33% | 84.72% |
| SPARSE | R0 | 1 | 32 | 83.33% | 83.33% |
| SPARSE | R0 | 2 | 64 | 83.33% | 83.85% |
| SPARSE | R0 | 4 | 128 | 83.33% | 84.72% |
| SPARSE | R3 | 1 | 32 | 83.33% | 83.33% |
| SPARSE | R3 | 2 | 64 | 83.33% | 84.20% |
| SPARSE | R3 | 4 | 128 | 83.33% | 84.55% |
| SPARSE | R4_action | 1 | 32 | 83.33% | 83.33% |
| SPARSE | R4_action | 2 | 64 | 83.33% | 83.85% |
| SPARSE | R4_action | 4 | 128 | 83.33% | 84.20% |
| SPARSE | R_oracle | 1 | 32 | 83.33% | 83.33% |
| SPARSE | R_oracle | 2 | 64 | 83.33% | 84.20% |
| SPARSE | R_oracle | 4 | 128 | 83.33% | 84.38% |

取预定前k个S_train，未观测head回退源prior；不读seed2–3。k1=32个full H700 episode，R5=32个200step短probe，episode数量相同而CPU/步数不同，不能称成本精确相等。

### 每fold/每表示结果（含LOAO次要划分）

| 划分 | 覆盖 | fold | 目标C | 表示 | 必要 | success下界 | vs源prior | vs最近源 | vsR0 | Q2 MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOPO | DENSE | 0 | blend_pair0_lambda25 | R0 | False | 57.29% | +3.12pp | -0.52pp | +0.00pp | 0.0629 |
| LOPO | DENSE | 0 | blend_pair0_lambda50 | R0 | False | 50.00% | +1.56pp | -12.50pp | +0.00pp | 0.0582 |
| LOPO | DENSE | 0 | blend_pair0_lambda75 | R0 | False | 53.12% | +4.69pp | +4.69pp | +0.00pp | 0.0576 |
| LOPO | DENSE | 0 | blend_pair0_lambda25 | R3 | False | 59.38% | +4.69pp | +1.56pp | +1.56pp | 0.0557 |
| LOPO | DENSE | 0 | blend_pair0_lambda50 | R3 | False | 52.08% | +3.12pp | -10.42pp | +1.56pp | 0.0571 |
| LOPO | DENSE | 0 | blend_pair0_lambda75 | R3 | False | 50.52% | +2.08pp | +2.08pp | -2.60pp | 0.0602 |
| LOPO | DENSE | 0 | blend_pair0_lambda25 | R4_action | False | 59.90% | +5.21pp | +2.08pp | +2.08pp | 0.0650 |
| LOPO | DENSE | 0 | blend_pair0_lambda50 | R4_action | False | 63.02% | +13.02pp | +0.52pp | +11.46pp | 0.0610 |
| LOPO | DENSE | 0 | blend_pair0_lambda75 | R4_action | False | 48.44% | -0.52pp | -0.52pp | -5.21pp | 0.0572 |
| LOPO | DENSE | 0 | blend_pair0_lambda25 | R_oracle | False | 51.56% | -1.56pp | -6.25pp | -5.73pp | 0.0633 |
| LOPO | DENSE | 0 | blend_pair0_lambda50 | R_oracle | False | 50.00% | +1.56pp | -12.50pp | -0.00pp | 0.0589 |
| LOPO | DENSE | 0 | blend_pair0_lambda75 | R_oracle | False | 49.48% | +0.52pp | +0.52pp | -4.17pp | 0.0585 |
| LOPO | SPARSE | 0 | blend_pair0_lambda25 | R0 | False | 53.12% | +0.00pp | -4.69pp | +0.00pp | 0.0606 |
| LOPO | SPARSE | 0 | blend_pair0_lambda50 | R0 | False | 48.44% | +0.00pp | -14.06pp | +0.00pp | 0.0582 |
| LOPO | SPARSE | 0 | blend_pair0_lambda75 | R0 | False | 48.44% | +0.00pp | +0.00pp | +0.00pp | 0.0563 |
| LOPO | SPARSE | 0 | blend_pair0_lambda25 | R3 | False | 58.33% | +3.65pp | +0.52pp | +3.65pp | 0.0641 |
| LOPO | SPARSE | 0 | blend_pair0_lambda50 | R3 | False | 57.29% | +8.33pp | -5.21pp | +8.33pp | 0.0647 |
| LOPO | SPARSE | 0 | blend_pair0_lambda75 | R3 | False | 47.92% | -0.52pp | -0.52pp | -0.52pp | 0.0641 |
| LOPO | SPARSE | 0 | blend_pair0_lambda25 | R4_action | False | 50.00% | -4.17pp | -7.81pp | -4.17pp | 0.0639 |
| LOPO | SPARSE | 0 | blend_pair0_lambda50 | R4_action | False | 50.52% | +2.08pp | -11.98pp | +2.08pp | 0.0609 |
| LOPO | SPARSE | 0 | blend_pair0_lambda75 | R4_action | False | 50.52% | +2.08pp | +2.08pp | +2.08pp | 0.0560 |
| LOPO | SPARSE | 0 | blend_pair0_lambda25 | R_oracle | False | 40.10% | -13.54pp | -17.71pp | -13.54pp | 0.0755 |
| LOPO | SPARSE | 0 | blend_pair0_lambda50 | R_oracle | False | 42.19% | -6.25pp | -20.31pp | -6.25pp | 0.0763 |
| LOPO | SPARSE | 0 | blend_pair0_lambda75 | R_oracle | False | 39.58% | -9.38pp | -9.38pp | -9.38pp | 0.0731 |
| LOPO | DENSE | 1 | blend_pair1_lambda25 | R0 | False | 86.46% | +0.52pp | -13.54pp | +0.00pp | 0.0613 |
| LOPO | DENSE | 1 | blend_pair1_lambda50 | R0 | True | 84.38% | +0.00pp | -10.94pp | +0.00pp | 0.0660 |
| LOPO | DENSE | 1 | blend_pair1_lambda75 | R0 | False | 95.31% | +0.00pp | +0.00pp | +0.00pp | 0.0736 |
| LOPO | DENSE | 1 | blend_pair1_lambda25 | R3 | False | 97.92% | +11.98pp | -2.08pp | +11.46pp | 0.0810 |
| LOPO | DENSE | 1 | blend_pair1_lambda50 | R3 | True | 30.73% | -53.65pp | -64.58pp | -53.65pp | 0.2438 |
| LOPO | DENSE | 1 | blend_pair1_lambda75 | R3 | False | 43.75% | -54.17pp | -54.17pp | -54.17pp | 0.1567 |
| LOPO | DENSE | 1 | blend_pair1_lambda25 | R4_action | False | 95.31% | +9.38pp | -4.69pp | +8.85pp | 0.0719 |
| LOPO | DENSE | 1 | blend_pair1_lambda50 | R4_action | True | 87.50% | +3.12pp | -7.81pp | +3.12pp | 0.0719 |
| LOPO | DENSE | 1 | blend_pair1_lambda75 | R4_action | False | 90.62% | -6.25pp | -6.25pp | -6.25pp | 0.0754 |
| LOPO | DENSE | 1 | blend_pair1_lambda25 | R_oracle | False | 55.73% | -30.21pp | -44.27pp | -30.73pp | 0.1766 |
| LOPO | DENSE | 1 | blend_pair1_lambda50 | R_oracle | True | 53.12% | -31.25pp | -42.19pp | -31.25pp | 0.1757 |
| LOPO | DENSE | 1 | blend_pair1_lambda75 | R_oracle | False | 54.69% | -43.23pp | -43.23pp | -43.23pp | 0.1780 |
| LOPO | SPARSE | 1 | blend_pair1_lambda25 | R0 | False | 85.94% | +0.00pp | -14.06pp | +0.00pp | 0.0767 |
| LOPO | SPARSE | 1 | blend_pair1_lambda50 | R0 | True | 84.38% | +0.00pp | +0.00pp | +0.00pp | 0.0826 |
| LOPO | SPARSE | 1 | blend_pair1_lambda75 | R0 | False | 95.31% | +0.00pp | +0.00pp | +0.00pp | 0.0903 |
| LOPO | SPARSE | 1 | blend_pair1_lambda25 | R3 | False | 85.42% | -0.52pp | -14.58pp | -0.52pp | 0.1050 |
| LOPO | SPARSE | 1 | blend_pair1_lambda50 | R3 | True | 26.56% | -57.81pp | -57.81pp | -57.81pp | 0.2442 |
| LOPO | SPARSE | 1 | blend_pair1_lambda75 | R3 | False | 53.12% | -45.31pp | -45.31pp | -45.31pp | 0.1317 |
| LOPO | SPARSE | 1 | blend_pair1_lambda25 | R4_action | False | 84.90% | -1.04pp | -15.10pp | -1.04pp | 0.0850 |
| LOPO | SPARSE | 1 | blend_pair1_lambda50 | R4_action | True | 81.25% | -3.12pp | -3.12pp | -3.12pp | 0.0918 |
| LOPO | SPARSE | 1 | blend_pair1_lambda75 | R4_action | False | 90.10% | -6.25pp | -6.25pp | -6.25pp | 0.0958 |
| LOPO | SPARSE | 1 | blend_pair1_lambda25 | R_oracle | False | 83.33% | -2.60pp | -16.67pp | -2.60pp | 0.1396 |
| LOPO | SPARSE | 1 | blend_pair1_lambda50 | R_oracle | True | 82.81% | -1.56pp | -1.56pp | -1.56pp | 0.1386 |
| LOPO | SPARSE | 1 | blend_pair1_lambda75 | R_oracle | False | 77.08% | -20.83pp | -20.83pp | -20.83pp | 0.1440 |
| LOPO | DENSE | 2 | blend_pair2_lambda25 | R0 | True | 65.62% | +1.56pp | +18.75pp | +0.00pp | 0.0638 |
| LOPO | DENSE | 2 | blend_pair2_lambda50 | R0 | True | 86.46% | -1.04pp | -1.04pp | +0.00pp | 0.0511 |
| LOPO | DENSE | 2 | blend_pair2_lambda75 | R0 | False | 85.94% | +0.00pp | +0.00pp | +0.00pp | 0.0469 |
| LOPO | DENSE | 2 | blend_pair2_lambda25 | R3 | True | 36.46% | -27.60pp | -10.42pp | -29.17pp | 0.1500 |
| LOPO | DENSE | 2 | blend_pair2_lambda50 | R3 | True | 29.69% | -57.81pp | -57.81pp | -56.77pp | 0.1685 |
| LOPO | DENSE | 2 | blend_pair2_lambda75 | R3 | False | 87.50% | +1.56pp | +1.56pp | +1.56pp | 0.0794 |
| LOPO | DENSE | 2 | blend_pair2_lambda25 | R4_action | True | 66.67% | +2.60pp | +19.79pp | +1.04pp | 0.0692 |
| LOPO | DENSE | 2 | blend_pair2_lambda50 | R4_action | True | 84.90% | -2.60pp | -2.60pp | -1.56pp | 0.0558 |
| LOPO | DENSE | 2 | blend_pair2_lambda75 | R4_action | False | 83.85% | -2.08pp | -2.08pp | -2.08pp | 0.0553 |
| LOPO | DENSE | 2 | blend_pair2_lambda25 | R_oracle | True | 37.50% | -26.56pp | -9.38pp | -28.12pp | 0.1838 |
| LOPO | DENSE | 2 | blend_pair2_lambda50 | R_oracle | True | 35.42% | -52.08pp | -52.08pp | -51.04pp | 0.1848 |
| LOPO | DENSE | 2 | blend_pair2_lambda75 | R_oracle | False | 52.08% | -33.85pp | -33.85pp | -33.85pp | 0.1825 |
| LOPO | SPARSE | 2 | blend_pair2_lambda25 | R0 | True | 64.06% | +0.00pp | -21.88pp | +0.00pp | 0.0722 |
| LOPO | SPARSE | 2 | blend_pair2_lambda50 | R0 | True | 87.50% | +0.00pp | +0.00pp | +0.00pp | 0.0611 |
| LOPO | SPARSE | 2 | blend_pair2_lambda75 | R0 | False | 85.94% | +0.00pp | +0.00pp | +0.00pp | 0.0585 |
| LOPO | SPARSE | 2 | blend_pair2_lambda25 | R3 | True | 64.06% | +0.00pp | -21.88pp | +0.00pp | 0.1148 |
| LOPO | SPARSE | 2 | blend_pair2_lambda50 | R3 | True | 35.42% | -52.08pp | -52.08pp | -52.08pp | 0.1386 |
| LOPO | SPARSE | 2 | blend_pair2_lambda75 | R3 | False | 79.17% | -6.77pp | -6.77pp | -6.77pp | 0.0833 |
| LOPO | SPARSE | 2 | blend_pair2_lambda25 | R4_action | True | 68.23% | +4.17pp | -17.71pp | +4.17pp | 0.0724 |
| LOPO | SPARSE | 2 | blend_pair2_lambda50 | R4_action | True | 85.42% | -2.08pp | -2.08pp | -2.08pp | 0.0618 |
| LOPO | SPARSE | 2 | blend_pair2_lambda75 | R4_action | False | 84.90% | -1.04pp | -1.04pp | -1.04pp | 0.0613 |
| LOPO | SPARSE | 2 | blend_pair2_lambda25 | R_oracle | True | 56.77% | -7.29pp | -29.17pp | -7.29pp | 0.1267 |
| LOPO | SPARSE | 2 | blend_pair2_lambda50 | R_oracle | True | 66.67% | -20.83pp | -20.83pp | -20.83pp | 0.1211 |
| LOPO | SPARSE | 2 | blend_pair2_lambda75 | R_oracle | False | 80.21% | -5.73pp | -5.73pp | -5.73pp | 0.1178 |
| LOPO | DENSE | 3 | blend_pair3_lambda25 | R0 | False | 93.75% | +0.00pp | -1.56pp | +0.00pp | 0.1514 |
| LOPO | DENSE | 3 | blend_pair3_lambda50 | R0 | False | 100.00% | +0.00pp | +0.00pp | +0.00pp | 0.1539 |
| LOPO | DENSE | 3 | blend_pair3_lambda75 | R0 | False | 100.00% | +0.00pp | +0.00pp | +0.00pp | 0.1527 |
| LOPO | DENSE | 3 | blend_pair3_lambda25 | R3 | False | 96.88% | +3.12pp | +1.56pp | +3.12pp | 0.0822 |
| LOPO | DENSE | 3 | blend_pair3_lambda50 | R3 | False | 96.35% | -3.65pp | -3.65pp | -3.65pp | 0.0800 |
| LOPO | DENSE | 3 | blend_pair3_lambda75 | R3 | False | 95.83% | -4.17pp | -4.17pp | -4.17pp | 0.0775 |
| LOPO | DENSE | 3 | blend_pair3_lambda25 | R4_action | False | 91.15% | -2.60pp | -4.17pp | -2.60pp | 0.1444 |
| LOPO | DENSE | 3 | blend_pair3_lambda50 | R4_action | False | 96.88% | -3.12pp | -3.12pp | -3.12pp | 0.1502 |
| LOPO | DENSE | 3 | blend_pair3_lambda75 | R4_action | False | 96.88% | -3.12pp | -3.12pp | -3.12pp | 0.1521 |
| LOPO | DENSE | 3 | blend_pair3_lambda25 | R_oracle | False | 98.44% | +4.69pp | +3.12pp | +4.69pp | 0.0958 |
| LOPO | DENSE | 3 | blend_pair3_lambda50 | R_oracle | False | 97.92% | -2.08pp | -2.08pp | -2.08pp | 0.0988 |
| LOPO | DENSE | 3 | blend_pair3_lambda75 | R_oracle | False | 95.31% | -4.69pp | -4.69pp | -4.69pp | 0.1013 |
| LOPO | SPARSE | 3 | blend_pair3_lambda25 | R0 | False | 93.75% | +0.00pp | -1.56pp | +0.00pp | 0.1630 |
| LOPO | SPARSE | 3 | blend_pair3_lambda50 | R0 | False | 100.00% | +0.00pp | +7.81pp | +0.00pp | 0.1651 |
| LOPO | SPARSE | 3 | blend_pair3_lambda75 | R0 | False | 100.00% | +0.00pp | +0.00pp | +0.00pp | 0.1635 |
| LOPO | SPARSE | 3 | blend_pair3_lambda25 | R3 | False | 97.92% | +4.17pp | +2.60pp | +4.17pp | 0.0717 |
| LOPO | SPARSE | 3 | blend_pair3_lambda50 | R3 | False | 95.31% | -4.69pp | +3.12pp | -4.69pp | 0.0730 |
| LOPO | SPARSE | 3 | blend_pair3_lambda75 | R3 | False | 94.79% | -5.21pp | -5.21pp | -5.21pp | 0.0720 |
| LOPO | SPARSE | 3 | blend_pair3_lambda25 | R4_action | False | 93.23% | -0.52pp | -2.08pp | -0.52pp | 0.1598 |
| LOPO | SPARSE | 3 | blend_pair3_lambda50 | R4_action | False | 98.44% | -1.56pp | +6.25pp | -1.56pp | 0.1644 |
| LOPO | SPARSE | 3 | blend_pair3_lambda75 | R4_action | False | 98.96% | -1.04pp | -1.04pp | -1.04pp | 0.1648 |
| LOPO | SPARSE | 3 | blend_pair3_lambda25 | R_oracle | False | 96.88% | +3.12pp | +1.56pp | +3.12pp | 0.0736 |
| LOPO | SPARSE | 3 | blend_pair3_lambda50 | R_oracle | False | 96.88% | -3.12pp | +4.69pp | -3.12pp | 0.0765 |
| LOPO | SPARSE | 3 | blend_pair3_lambda75 | R_oracle | False | 94.27% | -5.73pp | -5.73pp | -5.73pp | 0.0785 |
| LOAO | DENSE | 0 | blend_pair0_lambda25 | R0 | False | 57.29% | +2.60pp | -0.52pp | +0.00pp | 0.0634 |
| LOAO | DENSE | 0 | blend_pair0_lambda50 | R0 | False | 50.00% | +1.56pp | -12.50pp | +0.00pp | 0.0587 |
| LOAO | DENSE | 0 | blend_pair0_lambda75 | R0 | False | 53.65% | +5.21pp | -7.29pp | +0.00pp | 0.0578 |
| LOAO | DENSE | 0 | blend_pair1_lambda25 | R0 | False | 89.58% | +3.65pp | -10.42pp | +0.00pp | 0.0585 |
| LOAO | DENSE | 0 | blend_pair1_lambda50 | R0 | True | 88.54% | +4.17pp | -6.77pp | +0.00pp | 0.0625 |
| LOAO | DENSE | 0 | blend_pair1_lambda75 | R0 | False | 95.83% | +0.00pp | +0.00pp | +0.00pp | 0.0700 |
| LOAO | DENSE | 0 | blend_pair0_lambda25 | R3 | False | 56.77% | +2.08pp | -1.04pp | -0.52pp | 0.0619 |
| LOAO | DENSE | 0 | blend_pair0_lambda50 | R3 | False | 60.42% | +10.94pp | -2.08pp | +9.38pp | 0.0622 |
| LOAO | DENSE | 0 | blend_pair0_lambda75 | R3 | False | 50.52% | +2.08pp | -9.38pp | -3.13pp | 0.0597 |
| LOAO | DENSE | 0 | blend_pair1_lambda25 | R3 | False | 97.92% | +11.98pp | -2.08pp | +8.33pp | 0.1038 |
| LOAO | DENSE | 0 | blend_pair1_lambda50 | R3 | True | 26.04% | -58.33pp | -69.27pp | -62.50pp | 0.2270 |
| LOAO | DENSE | 0 | blend_pair1_lambda75 | R3 | False | 34.90% | -63.54pp | -63.54pp | -63.54pp | 0.1408 |
| LOAO | DENSE | 0 | blend_pair0_lambda25 | R4_action | False | 53.65% | -1.04pp | -4.17pp | -3.65pp | 0.0670 |
| LOAO | DENSE | 0 | blend_pair0_lambda50 | R4_action | False | 53.65% | +5.21pp | -8.85pp | +3.65pp | 0.0608 |
| LOAO | DENSE | 0 | blend_pair0_lambda75 | R4_action | False | 50.52% | +2.08pp | -8.85pp | -3.13pp | 0.0556 |
| LOAO | DENSE | 0 | blend_pair1_lambda25 | R4_action | False | 89.06% | +3.12pp | -10.94pp | -0.52pp | 0.0676 |
| LOAO | DENSE | 0 | blend_pair1_lambda50 | R4_action | True | 82.81% | -1.56pp | -12.50pp | -5.73pp | 0.0721 |
| LOAO | DENSE | 0 | blend_pair1_lambda75 | R4_action | False | 90.62% | -6.25pp | -6.25pp | -6.25pp | 0.0748 |
| LOAO | DENSE | 0 | blend_pair0_lambda25 | R_oracle | False | 42.19% | -12.50pp | -15.62pp | -15.10pp | 0.0696 |
| LOAO | DENSE | 0 | blend_pair0_lambda50 | R_oracle | False | 50.00% | +0.52pp | -12.50pp | -1.04pp | 0.0690 |
| LOAO | DENSE | 0 | blend_pair0_lambda75 | R_oracle | False | 46.88% | -2.08pp | -12.50pp | -7.29pp | 0.0657 |
| LOAO | DENSE | 0 | blend_pair1_lambda25 | R_oracle | False | 56.25% | -29.69pp | -43.75pp | -33.33pp | 0.1887 |
| LOAO | DENSE | 0 | blend_pair1_lambda50 | R_oracle | True | 48.44% | -35.94pp | -46.87pp | -40.10pp | 0.1890 |
| LOAO | DENSE | 0 | blend_pair1_lambda75 | R_oracle | False | 44.79% | -53.65pp | -53.65pp | -53.65pp | 0.1930 |
| LOAO | SPARSE | 0 | blend_pair0_lambda25 | R0 | False | 53.12% | +0.00pp | -4.69pp | +0.00pp | 0.0606 |
| LOAO | SPARSE | 0 | blend_pair0_lambda50 | R0 | False | 48.44% | +0.00pp | -14.06pp | +0.00pp | 0.0582 |
| LOAO | SPARSE | 0 | blend_pair0_lambda75 | R0 | False | 48.44% | +0.00pp | +0.00pp | +0.00pp | 0.0563 |
| LOAO | SPARSE | 0 | blend_pair1_lambda25 | R0 | False | 85.94% | +0.00pp | -14.06pp | +0.00pp | 0.0767 |
| LOAO | SPARSE | 0 | blend_pair1_lambda50 | R0 | True | 84.38% | +0.00pp | +0.00pp | +0.00pp | 0.0826 |
| LOAO | SPARSE | 0 | blend_pair1_lambda75 | R0 | False | 95.31% | +0.00pp | +0.00pp | +0.00pp | 0.0903 |
| LOAO | SPARSE | 0 | blend_pair0_lambda25 | R3 | False | 58.33% | +3.65pp | +0.52pp | +3.65pp | 0.0641 |
| LOAO | SPARSE | 0 | blend_pair0_lambda50 | R3 | False | 57.29% | +8.33pp | -5.21pp | +8.33pp | 0.0647 |
| LOAO | SPARSE | 0 | blend_pair0_lambda75 | R3 | False | 47.92% | -0.52pp | -0.52pp | -0.52pp | 0.0641 |
| LOAO | SPARSE | 0 | blend_pair1_lambda25 | R3 | False | 85.42% | -0.52pp | -14.58pp | -0.52pp | 0.1050 |
| LOAO | SPARSE | 0 | blend_pair1_lambda50 | R3 | True | 26.56% | -57.81pp | -57.81pp | -57.81pp | 0.2442 |
| LOAO | SPARSE | 0 | blend_pair1_lambda75 | R3 | False | 53.12% | -45.31pp | -45.31pp | -45.31pp | 0.1317 |
| LOAO | SPARSE | 0 | blend_pair0_lambda25 | R4_action | False | 50.00% | -4.17pp | -7.81pp | -4.17pp | 0.0639 |
| LOAO | SPARSE | 0 | blend_pair0_lambda50 | R4_action | False | 50.52% | +2.08pp | -11.98pp | +2.08pp | 0.0609 |
| LOAO | SPARSE | 0 | blend_pair0_lambda75 | R4_action | False | 50.52% | +2.08pp | +2.08pp | +2.08pp | 0.0560 |
| LOAO | SPARSE | 0 | blend_pair1_lambda25 | R4_action | False | 84.90% | -1.04pp | -15.10pp | -1.04pp | 0.0850 |
| LOAO | SPARSE | 0 | blend_pair1_lambda50 | R4_action | True | 81.25% | -3.12pp | -3.12pp | -3.12pp | 0.0918 |
| LOAO | SPARSE | 0 | blend_pair1_lambda75 | R4_action | False | 90.10% | -6.25pp | -6.25pp | -6.25pp | 0.0958 |
| LOAO | SPARSE | 0 | blend_pair0_lambda25 | R_oracle | False | 40.10% | -13.54pp | -17.71pp | -13.54pp | 0.0755 |
| LOAO | SPARSE | 0 | blend_pair0_lambda50 | R_oracle | False | 42.19% | -6.25pp | -20.31pp | -6.25pp | 0.0763 |
| LOAO | SPARSE | 0 | blend_pair0_lambda75 | R_oracle | False | 39.58% | -9.38pp | -9.38pp | -9.38pp | 0.0731 |
| LOAO | SPARSE | 0 | blend_pair1_lambda25 | R_oracle | False | 83.33% | -2.60pp | -16.67pp | -2.60pp | 0.1396 |
| LOAO | SPARSE | 0 | blend_pair1_lambda50 | R_oracle | True | 82.81% | -1.56pp | -1.56pp | -1.56pp | 0.1386 |
| LOAO | SPARSE | 0 | blend_pair1_lambda75 | R_oracle | False | 77.08% | -20.83pp | -20.83pp | -20.83pp | 0.1440 |
| LOAO | DENSE | 1 | blend_pair2_lambda25 | R0 | True | 64.06% | +0.00pp | -21.88pp | +0.00pp | 0.0618 |
| LOAO | DENSE | 1 | blend_pair2_lambda50 | R0 | True | 85.42% | -2.08pp | -2.08pp | +0.00pp | 0.0494 |
| LOAO | DENSE | 1 | blend_pair2_lambda75 | R0 | False | 85.94% | +0.00pp | +0.00pp | +0.00pp | 0.0457 |
| LOAO | DENSE | 1 | blend_pair3_lambda25 | R0 | False | 93.23% | -0.52pp | -2.08pp | +0.00pp | 0.1502 |
| LOAO | DENSE | 1 | blend_pair3_lambda50 | R0 | False | 98.96% | -1.04pp | +6.77pp | +0.00pp | 0.1522 |
| LOAO | DENSE | 1 | blend_pair3_lambda75 | R0 | False | 100.00% | +0.00pp | +0.00pp | +0.00pp | 0.1512 |
| LOAO | DENSE | 1 | blend_pair2_lambda25 | R3 | True | 38.02% | -26.04pp | -47.92pp | -26.04pp | 0.1233 |
| LOAO | DENSE | 1 | blend_pair2_lambda50 | R3 | True | 38.02% | -49.48pp | -49.48pp | -47.40pp | 0.1424 |
| LOAO | DENSE | 1 | blend_pair2_lambda75 | R3 | False | 84.38% | -1.56pp | -1.56pp | -1.56pp | 0.0785 |
| LOAO | DENSE | 1 | blend_pair3_lambda25 | R3 | False | 98.44% | +4.69pp | +3.12pp | +5.21pp | 0.0730 |
| LOAO | DENSE | 1 | blend_pair3_lambda50 | R3 | False | 96.88% | -3.12pp | +4.69pp | -2.08pp | 0.0775 |
| LOAO | DENSE | 1 | blend_pair3_lambda75 | R3 | False | 95.31% | -4.69pp | -4.69pp | -4.69pp | 0.0795 |
| LOAO | DENSE | 1 | blend_pair2_lambda25 | R4_action | True | 68.23% | +4.17pp | -17.71pp | +4.17pp | 0.0644 |
| LOAO | DENSE | 1 | blend_pair2_lambda50 | R4_action | True | 86.98% | -0.52pp | -0.52pp | +1.56pp | 0.0522 |
| LOAO | DENSE | 1 | blend_pair2_lambda75 | R4_action | False | 85.94% | +0.00pp | +0.00pp | +0.00pp | 0.0519 |
| LOAO | DENSE | 1 | blend_pair3_lambda25 | R4_action | False | 92.19% | -1.56pp | -3.12pp | -1.04pp | 0.1475 |
| LOAO | DENSE | 1 | blend_pair3_lambda50 | R4_action | False | 97.40% | -2.60pp | +5.21pp | -1.56pp | 0.1525 |
| LOAO | DENSE | 1 | blend_pair3_lambda75 | R4_action | False | 97.92% | -2.08pp | -2.08pp | -2.08pp | 0.1537 |
| LOAO | DENSE | 1 | blend_pair2_lambda25 | R_oracle | True | 44.79% | -19.27pp | -41.15pp | -19.27pp | 0.1439 |
| LOAO | DENSE | 1 | blend_pair2_lambda50 | R_oracle | True | 51.56% | -35.94pp | -35.94pp | -33.85pp | 0.1429 |
| LOAO | DENSE | 1 | blend_pair2_lambda75 | R_oracle | False | 61.98% | -23.96pp | -23.96pp | -23.96pp | 0.1423 |
| LOAO | DENSE | 1 | blend_pair3_lambda25 | R_oracle | False | 98.44% | +4.69pp | +3.12pp | +5.21pp | 0.0752 |
| LOAO | DENSE | 1 | blend_pair3_lambda50 | R_oracle | False | 96.88% | -3.12pp | +4.69pp | -2.08pp | 0.0765 |
| LOAO | DENSE | 1 | blend_pair3_lambda75 | R_oracle | False | 95.31% | -4.69pp | -4.69pp | -4.69pp | 0.0792 |
| LOAO | SPARSE | 1 | blend_pair2_lambda25 | R0 | True | 64.06% | +0.00pp | -21.88pp | +0.00pp | 0.0722 |
| LOAO | SPARSE | 1 | blend_pair2_lambda50 | R0 | True | 87.50% | +0.00pp | +0.00pp | +0.00pp | 0.0611 |
| LOAO | SPARSE | 1 | blend_pair2_lambda75 | R0 | False | 85.94% | +0.00pp | +0.00pp | +0.00pp | 0.0585 |
| LOAO | SPARSE | 1 | blend_pair3_lambda25 | R0 | False | 93.75% | +0.00pp | -1.56pp | +0.00pp | 0.1630 |
| LOAO | SPARSE | 1 | blend_pair3_lambda50 | R0 | False | 100.00% | +0.00pp | +7.81pp | +0.00pp | 0.1651 |
| LOAO | SPARSE | 1 | blend_pair3_lambda75 | R0 | False | 100.00% | +0.00pp | +0.00pp | +0.00pp | 0.1635 |
| LOAO | SPARSE | 1 | blend_pair2_lambda25 | R3 | True | 64.06% | +0.00pp | -21.88pp | +0.00pp | 0.1148 |
| LOAO | SPARSE | 1 | blend_pair2_lambda50 | R3 | True | 35.42% | -52.08pp | -52.08pp | -52.08pp | 0.1386 |
| LOAO | SPARSE | 1 | blend_pair2_lambda75 | R3 | False | 79.17% | -6.77pp | -6.77pp | -6.77pp | 0.0833 |
| LOAO | SPARSE | 1 | blend_pair3_lambda25 | R3 | False | 97.92% | +4.17pp | +2.60pp | +4.17pp | 0.0717 |
| LOAO | SPARSE | 1 | blend_pair3_lambda50 | R3 | False | 95.31% | -4.69pp | +3.12pp | -4.69pp | 0.0730 |
| LOAO | SPARSE | 1 | blend_pair3_lambda75 | R3 | False | 94.79% | -5.21pp | -5.21pp | -5.21pp | 0.0720 |
| LOAO | SPARSE | 1 | blend_pair2_lambda25 | R4_action | True | 68.23% | +4.17pp | -17.71pp | +4.17pp | 0.0724 |
| LOAO | SPARSE | 1 | blend_pair2_lambda50 | R4_action | True | 85.42% | -2.08pp | -2.08pp | -2.08pp | 0.0618 |
| LOAO | SPARSE | 1 | blend_pair2_lambda75 | R4_action | False | 84.90% | -1.04pp | -1.04pp | -1.04pp | 0.0613 |
| LOAO | SPARSE | 1 | blend_pair3_lambda25 | R4_action | False | 93.23% | -0.52pp | -2.08pp | -0.52pp | 0.1598 |
| LOAO | SPARSE | 1 | blend_pair3_lambda50 | R4_action | False | 98.44% | -1.56pp | +6.25pp | -1.56pp | 0.1644 |
| LOAO | SPARSE | 1 | blend_pair3_lambda75 | R4_action | False | 98.96% | -1.04pp | -1.04pp | -1.04pp | 0.1648 |
| LOAO | SPARSE | 1 | blend_pair2_lambda25 | R_oracle | True | 56.77% | -7.29pp | -29.17pp | -7.29pp | 0.1267 |
| LOAO | SPARSE | 1 | blend_pair2_lambda50 | R_oracle | True | 66.67% | -20.83pp | -20.83pp | -20.83pp | 0.1211 |
| LOAO | SPARSE | 1 | blend_pair2_lambda75 | R_oracle | False | 80.21% | -5.73pp | -5.73pp | -5.73pp | 0.1178 |
| LOAO | SPARSE | 1 | blend_pair3_lambda25 | R_oracle | False | 96.88% | +3.12pp | +1.56pp | +3.12pp | 0.0736 |
| LOAO | SPARSE | 1 | blend_pair3_lambda50 | R_oracle | False | 96.88% | -3.12pp | +4.69pp | -3.12pp | 0.0765 |
| LOAO | SPARSE | 1 | blend_pair3_lambda75 | R_oracle | False | 94.27% | -5.73pp | -5.73pp | -5.73pp | 0.0785 |

LOAO只留出包含某A的融合pair，原A checkpoint仍在14源中；不是未见原锚点/真实checkpoint的验证。

### 根因分类

**INCONCLUSIVE**。按预注册映射分类，必要controller少于4优先INCONCLUSIVE。若oracle没过联合门，必须检查是否只有最近源baseline已正确选regime这一项使门失败，不得据此声称映射无法学。统计未显著不证明效应为零；小样本、源验证、优化与近邻天花板边界同时保留。


## 成立、否定和未解决

| 假设 | 判定 |
| --- | --- |
| real_flow_eta_regime_variation | SUPPORTED — existing sparse TRAIN evidence; not new confirmation |
| mean_static_features_LOO_outperform_majority | REJECTED IN THIS DESCRIPTIVE TEST |
| blend_internal_regime_coverage | SUPPORTED |
| coverage_vs_representation_vs_method | INCONCLUSIVE |
| held_blend_selection_transfer | INCONCLUSIVE |

Stage0未超过多数类不能证明表示没有regime信息；pilot不检验selection泛化。本轮主析因已执行；两臂均仅3个必要controller，低于事前要求4个，主结论INCONCLUSIVE。R4_action的必要controller描述性success为DENSE79.69%、SPARSE78.30%，没有取得正式联合迁移确认。R3与特权坐标的DENSE−SPARSE分别−9.90pp、−26.91pp（同时CI均小于0），因此本设定没有支持增加融合源即可修复；这仍不能唯一归因表示或方法。R5有一个unknown探测，完整保留而未填补，整臂未执行。 合成family的结果不能直接覆盖真实未见checkpoint、跨geometry或连续success basin。

## 可直接复用资产及下一步入口

目录：/home/zhihan/research/Basin_C1/diagnostics/controller_flow_blend_v1。先读REPORT.md、results.json和PREREGISTRATION_STAGE_1.md；STAGE_0_RESULTS.json给14条曲线CI/距离/混淆/冻结prior/regret分解；DATA_REUSE_INDEX.json给历史hash。ENDPOINT_EQUIVALENCE_AUDIT.json给语义检查。stage_1/含协议、CASES/MISSING_CASES、CACHE_PREFLIGHT、PILOT_TIMING、raw和labels/results（只有真正产生后存在）；shared DB的experiment为exp_controller_flow_blend_v1_stage_1。

重放分析命令：/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_flow_blend_v1/stage0.py；/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_flow_blend_v1/stage1_analyze.py（仅postflight成功后）；/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/controller_flow_blend_v1/render_report.py。历史round4可复用/home/zhihan/research/Basin_C1/diagnostics/controller_loco_real_flow_v1/LABELS.npz、stage_b/LABELS.npz、static_features/、controller_prior/，勿覆盖。

本轮已完成Stage0–2及所有适用分支，新增标签、模型和预测实际存在。必要controller仅3个，低于4个；禁止把描述性失败改写成正式方法缺陷结论。优先复用现有预测检查源验证NLL为何选中R3/oracle的很早训练步（50/100等），以及held-pair输入是否在源特征支持范围内。这些是后续待验证假设，不是本轮已确认根因；不得在已打开终点上调参后冒充新确认。R5失败的一个probe原样保存，不重跑、不填0；因此闭环行为表示优于静态表示的假设仍未检验。未来独立确认另需新state、重新预注册及预算。 Stage2证据：stage_2/GATE_RESULTS.json、results.json、TRAIN_LABELS.npz、DISCOVERY_LABELS.npz、ENDPOINT_LABELS.npz、static_features/、fits/、shared_sparse_fits/、probes/RAW_TRACE_INDEX.json；复算使用analyze.py，再render_report.py和final_audit.py。完整证据审计在FINAL_CONSISTENCY_AUDIT.json。

资源：明天2026-10-14 08:00+08撤6CPU/4GPU额外容量，剩18CPU/4GPU可继续任务。资源授权/调度在RESOURCE_AUTHORIZATION.json、SLURM_SUBMISSIONS.json；只撤本人本任务r5x_作业。所有旧round4 state结果DEVELOPMENT，未来确认另需全新state。

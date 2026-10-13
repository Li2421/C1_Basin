# Controller Flow Blending：覆盖、表示和方法的辨识研究

更新时间UTC：2026-10-13T03:00:57.431150+00:00。整体状态：**COMPLETE**。

## A. Research Overview

核心目标是学到可迁移的Q(s,C,η)中controller作用，在未见controller上选择闭环成功率更高的η。上一轮真实Flow LOCO未获独立选择确认，但必要折均为η13经验最优、源先验为η10；增加七个源没有增加η13经验最优样例。本轮先复核controller η regime，再用连续world-space Flow融合跨越regime，满足pilot门后才做覆盖×表示析因。Q误差、regime分类与逐state选择不是同一终点。

## B. Implementation Summary

读取AGENTS.md、完整round5 prompt、round4 HANDOFF/REPORT/results/SOURCE_REGIME_COVERAGE及shared DB规范。新代码只在本目录；历史数据、checkpoint和报告只读、不commit。纯Q不要求视频。

Stage0由stage0.py执行，复用14个controller共享128state的随机平衡8-head Q4标签、静态特征以及冻结的controller_prior/predictions；共57344个TRAIN seed格（有效56750、unknown594），零rollout、零重训critic。每η64state，按128个共享初态bootstrap5000次，保留head请求mask；经验bestη同数据选择，仅探索性，报告unknown上下界和14×2 max-T区间。NN/logistic只训练regime分类器，LOO只由13源做normalization、L2=1固定无调参。

blend_runtime.py复用旧RingFlowAdapter、native v10 runtime、FF积分和安全投影。在每个共同world h处调用两个Flow自己的chart/normalizer，凸组合世界速度；h0用共同ξ凸组合两种反归一化。basis参考nominal为融合场无修正积分后终端chart clipping的凸组合；这个原prompt未具体定义的接口选择已事前明确。边界λ0/1保留原float32运行顺序；ENDPOINT_EQUIVALENCE_AUDIT.json的160次静态η/state/seed检查，端点最大误差为0，绕过分支的world公式也为0，邻近λ误差≤7.1e−7；禁止env.step，不以端点分支替代公式检查。global/local环境schema名称不同，其余物理配置逐项相同；初次严格schema检查停止，排除仅chart名称后继续，尚未产生outcome。

Stage1设计为4个固定锚点pair×3内部λ×24旧S_train×K4[2,4,10,13]×Q4=4608，98timing计入其中。新UID包括两个锚点完整execution identity、λ、runtimeSHA、nominal/chart规则。design.py先登记experiment/UID，再官方DBpreflight（4608确切缺失、0reuse、0ambiguous），仅执行missing；rollout.py写raw+journal，单进程merger、postflight、stage1_analyze.py。固定物理Ring/H700/FF/安全/评价器/JAX x64=false与seed语义。

Stage1翻转门事前明确检查三个内部λ：它们才是Stage2的12个目标/新增源；仅端点已知差异不证明内部新controller提供覆盖。原锚点标签前24state只有随机8头，K4每η12state，没有补跑缺失锚点。此操作化比把端点计入的解释更严格，报告实际定义，不伪称两种解释完全相同。正式预注册与receipt已冻结，采集、特征、拟合、评价均完成。R5未训练属于事前输入完整性分支，并非零性能。

执行调整：历史协议扫描遇到states为计数整数而非列表，首次Stage2设计在新outcome前停止，修正解析后继续；未修改设计/预注册。GPU静态作业首次以JAX_PLATFORMS=gpu请求后端，当前JAX展开到不存在的ROCm而初始化失败，未生成任何特征；改为显式cuda并重提，等待训练的作业也在未启动时替换/重接依赖。见STAGE2_PREPARATION_RESUME.log、stage_2/GPU_FEATURE_REPAIR.json、GPU_TRAINING_REPAIR.json、code_versions/。CPU新标签未重跑，模拟执行identity没有改变。CPU标签收尾时，提前释放原有probe数组的调度依赖，利用空闲CPU槽位；probe不读取闭环标签，其单进程merge仍等待标签merge结束。见stage_2/PROBE_EARLY_START.json。

## C. Experimental Results

| 阶段 | 状态 | 新增rollout/模型 |
| --- | --- | --- |
| Stage0 | COMPLETE EXPLORATORY | 0 rollout；无critic重训 |
| Stage1 | 完成；门通过 | 当前raw=4608/4608 |
| Stage2 | 完成 | 见正式阶段证据；未执行不填0性能 |

### Stage0：14真实Flow regime

| Flow | 经验bestη | best成功下界 | η10下界 | η13下界 | best−10同时CI | best−13同时CI |
| --- | --- | --- | --- | --- | --- | --- |
| base_u_plus_w_v4_wide_macflow | 10 | 65.62% | 65.62% | 25.78% | [+0.00pp, +0.00pp] | [+22.45pp, +55.67pp] |
| base_u_macflow | 13 | 57.03% | 25.78% | 57.03% | [+13.07pp, +48.65pp] | [+0.00pp, +0.00pp] |
| base_u_plus_w_v3_macflow | 2 | 96.88% | 89.06% | 66.02% | [-2.32pp, +17.16pp] | [+18.51pp, +40.86pp] |
| base_u_plus_w_v4_train_targeted_macflow | 10 | 35.94% | 35.94% | 7.03% | [+0.00pp, +0.00pp] | [+13.04pp, +43.99pp] |
| base_u_v10_local_macflow | 13 | 91.41% | 59.77% | 91.41% | [+13.73pp, +49.56pp] | [+0.00pp, +0.00pp] |
| base_u_plus_w_v4_macflow | 10 | 34.77% | 34.77% | 4.69% | [+0.00pp, +0.00pp] | [+13.43pp, +45.16pp] |
| base_u_v9_local_macflow | 10 | 96.48% | 96.48% | 80.47% | [+0.00pp, +0.00pp] | [+4.23pp, +27.80pp] |
| base_u_v4_macflow | 10 | 48.83% | 48.83% | 21.09% | [+0.00pp, +0.00pp] | [+8.30pp, +45.60pp] |
| base_u_plus_w_v6_macflow | 10 | 45.70% | 45.70% | 5.08% | [+0.00pp, +0.00pp] | [+23.38pp, +56.31pp] |
| base_u_v3_macflow | 2 | 91.80% | 73.05% | 61.72% | [-0.54pp, +31.79pp] | [+12.35pp, +43.90pp] |
| base_u_v2_macflow | 10 | 26.56% | 26.56% | 23.05% | [+0.00pp, +0.00pp] | [-14.47pp, +19.94pp] |
| base_u_plus_w_v5_macflow | 10 | 35.94% | 35.94% | 8.98% | [+0.00pp, +0.00pp] | [+9.52pp, +44.38pp] |
| base_u_v7_local_macflow | 10 | 95.31% | 95.31% | 35.94% | [+0.00pp, +0.00pp] | [+46.37pp, +72.38pp] |
| base_u_v8_local_macflow | 4 | 99.61% | 99.22% | 38.28% | [-1.62pp, +2.40pp] | [+49.50pp, +73.16pp] |

regime计数η2/4/10/13分别2/1/9/2。表中率为unknown保守下界，不是将unknown标成失败；best定义来自valid均值。比较区间是保守配对/按请求mask的探索性max-T，best同标签选出，不能称独立最优确认。sparse随机head意味着不同η覆盖不同state，随机分配不等于每state配对全K16。

| 表示 | NN LOO | Logistic LOO | 多数类LOO |
| --- | --- | --- | --- |
| R3 | 64.29% | 64.29% | 64.29% |
| R4_action | 35.71% | 64.29% | 64.29% |
| R3_R4_action | 64.29% | 64.29% | 64.29% |

分类混淆矩阵和14×13源归一化距离在STAGE_0_RESULTS.json。没有任何表示的这个LOO分类器超过多数类；14小样本、严重不平衡、η4类只有1例，不能推出表示不包含任何regime信息。

### fold1/fold4：现象、解释及regret分解

| 表示 | base_u最近邻 | v10_local最近邻 |
| --- | --- | --- |
| R3 | base_u_v2_macflow | base_u_v9_local_macflow |
| R4_action | base_u_plus_w_v5_macflow | base_u_v9_local_macflow |
| R3_R4_action | base_u_v2_macflow | base_u_v9_local_macflow |

| 阶段/折 | R3 prior | R4_action prior | R3+R4 prior |
| --- | --- | --- | --- |
| stage_a_fold1 | [4, 10, 10] | [10, 10, 10] | [4, 10, 10] |
| stage_a_fold4 | [4, 4, 4] | [10, 10, 10] | [4, 4, 10] |
| stage_b_fold1 | [10, 10, 10] | [10, 10, 10] | [10, 10, 10] |
| stage_b_fold4 | [4, 4, 10] | [10, 10, 10] | [4, 4, 10] |

**核查纠正**：上一轮摘要称fold4 R3“选η13”并不等于冻结代码的输出。Stage A fold4的模型自身prior为η4（3fit均如此），fold1为[η4,η10,η10]；Stage B也没有直接学到η13 prior。η4在v10旧测试state表现很好，故“经验η13-type”是标签regime的描述，不能代替模型实际所选η。成功决策不要求精确复现同一数据上经验argmax的index。

| 阶段 | 折 | 表示 | target固定−model固定 | model固定−逐state | 总regret |
| --- | --- | --- | --- | --- | --- |
| stage_a | 1 | R3 | +38.28pp | +0.13pp | +38.41pp |
| stage_a | 1 | R4_action | +31.64pp | +1.04pp | +32.68pp |
| stage_a | 1 | R3_R4_action | +38.28pp | -1.95pp | +36.33pp |
| stage_a | 4 | R3 | +0.00pp | +7.55pp | +7.55pp |
| stage_a | 4 | R4_action | +26.17pp | -2.99pp | +23.18pp |
| stage_a | 4 | R3_R4_action | +8.72pp | -2.47pp | +6.25pp |
| stage_b | 1 | R3 | +31.64pp | -1.43pp | +30.21pp |
| stage_b | 1 | R4_action | +31.64pp | +1.95pp | +33.59pp |
| stage_b | 1 | R3_R4_action | +31.64pp | -5.86pp | +25.78pp |
| stage_b | 4 | R3 | +8.72pp | +5.47pp | +14.19pp |
| stage_b | 4 | R4_action | +26.17pp | -2.60pp | +23.57pp |
| stage_b | 4 | R3_R4_action | +8.72pp | +9.11pp | +17.84pp |

上述point项从lower-success算出，相加恒等；各component的配对unknown保守上下界和bootstrap见JSON，不把分别最坏的component CI相加当精确总CI。fold1大量regret来自controller固定prior错误，fold4 R3在开发面板固定prior已很好而逐state偏离反而造成损失。静态均值NN并未找到另一个η13经验最优controller，能描述少数regime邻域稀疏，但不能将模型不对称因果归咎距离或源数量。

### Stage1：融合regime地图

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

### Stage2：覆盖×表示

完成。预定SPARSE14原Flow、DENSE另加9非留出pair融合Flow，4 LOPO；R0/R3/R4_action/R_oracle/R5。主终点必须为新testseed2–3选择success，必要门只用0–1，source normalization与原Flow P1验证NLL选步，3fits。若未执行，不存在coverage effect、R5数据、checkpoint或选择性能结果；不能借Stage0或pilot替代它们。

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


## D. Hypothesis Evaluation

| 命题 | 判定 |
| --- | --- |
| real_flow_eta_regime_variation | SUPPORTED — existing sparse TRAIN evidence; not new confirmation |
| mean_static_features_LOO_outperform_majority | REJECTED IN THIS DESCRIPTIVE TEST |
| blend_internal_regime_coverage | SUPPORTED |
| coverage_vs_representation_vs_method | INCONCLUSIVE |
| held_blend_selection_transfer | INCONCLUSIVE |

“LOO均值静态分类超过多数类”的否定只对应既定分类器和14点面板，不是对controller条件Q普遍可学习性的否定。门未通过不能用于把coverage/representation/method三种根因逐一排除。

## E. Controller–Q Relationship and Transferability

已知真实Flow改变η regime而非仅Q水平；Stage0不同best/η10/13曲线支持这一现象。本轮主析因已执行；两臂均仅3个必要controller，低于事前要求4个，主结论INCONCLUSIVE。R4_action的必要controller描述性success为DENSE79.69%、SPARSE78.30%，没有取得正式联合迁移确认。R3与特权坐标的DENSE−SPARSE分别−9.90pp、−26.91pp（同时CI均小于0），因此本设定没有支持增加融合源即可修复；这仍不能唯一归因表示或方法。R5有一个unknown探测，完整保留而未填补，整臂未执行。

融合族内插、留pair和特权已知融合坐标不能自动推广到全新真实Flow checkpoint或跨几何。R5若执行具有200step模拟行为查询部署假设，不是静态descriptor；其成功只能支持该proxy更有用，不能证明因果机制已识别。coverage对比同时改变总标签数量，明确其有限可辨识性；DENSE显著/SPARSE不显著也须检查直接DENSE−SPARSE差。

## F. Limitations and Next Steps

所有旧state为DEVELOPMENT；Stage0经验best存在同数据选择偏差、head覆盖差异、controller少数类稀疏，独立seed≠独立state。Pilot只有24state/Q4/K4，专门找regime边界，不能推断完整K16选择headroom或连续三维success basin形状。融合同时改变速度场与h0初始噪声分布，不能把结果只归因于速度场。R_oracle仍含压缩的锚点R3，而非完整程序的无损真坐标。必要controller门不足、源验证仅原Flow、NLL与选择目标可能偏移、未控制额外标签数量，均阻止唯一根因判断。统计相关、机制解释和理论猜想分别陈述。本轮已完成Stage0–2及所有适用分支，新增标签、模型和预测实际存在。必要controller仅3个，低于4个；禁止把描述性失败改写成正式方法缺陷结论。优先复用现有预测检查源验证NLL为何选中R3/oracle的很早训练步（50/100等），以及held-pair输入是否在源特征支持范围内。这些是后续待验证假设，不是本轮已确认根因；不得在已打开终点上调参后冒充新确认。R5失败的一个probe原样保存，不重跑、不填0；因此闭环行为表示优于静态表示的假设仍未检验。未来独立确认另需新state、重新预注册及预算。

## G. Reproducibility and Cost

工作根目录/home/zhihan/research/Basin_C1，Python /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python，代码版本a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e（无commit）。DATA_REUSE_INDEX.json记录历史资产hash；PREREGISTRATION*.receipt.json记录事前UTC/hash；stage_1/protocol.json含state内容、精确UID/η、checkpoint及两chart。runtime和model JAX x64必须false。

| 预注册 | UTC | SHA256 |
| --- | --- | --- |
| PREREGISTRATION_STAGE_0.md | 2026-10-12T17:03:26.275215+00:00 | 1ab2f7323ddf00cb07d25a550484a61f871af55c5e5b21940cd58fb27fe8ce9d |
| PREREGISTRATION_STAGE_1.md | 2026-10-12T17:06:56.957831+00:00 | 62b2670293247d5d85406b62b445820f779fe0adb996c20447b7197cb859c599 |
| PREREGISTRATION_STAGE_2.md | 2026-10-12T17:52:18.970700+00:00 | 138d94ce8cca223db7c9b2d64609d4f3c81ef8fbf9606d4f9e16621ad43620f5 |

98次pilot平均10.39s、median 9.37s、p90 22.84s；全Stage1初估13.30CPU小时/0.55h@24CPU，Stage2完整标签初估177.38CPU小时/7.39h@24CPU。实际全量尾部、特征、训练、队列与合并另计，不能视作任务完成时间保证。

Stage2正式成本门：复用2,304个pilot seed格（2,299 valid、5 unknown不重跑），新增59136个完整H700 episode及832个短probe，共59968、cap65,000。按全pilot平均10.14s估计166.53CPU小时，@24CPU=6.94h、@18CPU=9.25h；K16包含更多可能慢的η，仍非完成时间保证。

实际全研究新增63,744次完整闭环请求及832次短probe，总计64,576次。Stage2已合并61,440个格子标签（含2,304个pilot缓存）：有效61,009、unknown 431、缺失/隔离均0。probe有效统计831、unknown 1；R5整体未训练，保留失败与partial trace。历史TRAIN/验证/Stage0旧test分别57,344/21,504/16,384 seed格，共95,232；2,304个本轮pilot复用不计入历史独立标签数。

复现入口（既有raw永不重新运行；先audit再missing）：

```bash
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/stage0.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/blend_runtime.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/design.py create
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan --manifest diagnostics/controller_flow_blend_v1/stage_1/PLANNED_ROLLOUTS.json --output diagnostics/controller_flow_blend_v1/stage_1/CACHE_PREFLIGHT_RECHECK.json
# 已有MISSING_CASES是原冻结计划；重提作业先核查DB/raw，不重跑unknown。
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/rollout.py merge
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/rollout.py postflight
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/stage1_analyze.py
# Stage2从已保存标签与冻结逐seed预测重算；不会重训或生成rollout。
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/analyze.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/render_report.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_flow_blend_v1/final_audit.py
```

SLURM_SUBMISSIONS.json为实际作业/commands；CPU24/GPU8按需，用户额外6CPU/4GPU授权到2026-10-14 08:00+08，之后18CPU/4GPU。RESOURCE_AUTHORIZATION.json及独立systemd cutoff定时器只撤本任务r5x_额外作业，不动其他人的作业。未用GPU时不预占GPU。raw命名为精确rollout UID，不覆盖原数据。

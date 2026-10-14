# Round8：真实新controller的少样本η适应

状态 **COMPLETE**，UTC 2026-10-13T06:19:48.044745+00:00。全部证据 **DEVELOPMENT**。新增rollout/probe **0**，critic训练 **0**，共享DB连接/写入 **0**。所有采集是从既有标签池无放回抽取的模拟；纯Q不需要视频。

## 回答与结论映射

**少样本适应可行：预设regime候选集R下，E-safe在128条目标标签首次达到SPT/CT；H和冻结critic没有建立额外预算节省。** 在预注册预算网格0/4/8/16/32/64/128中，首次通过SPT/CT的是 **R + E-safe，B=128**。R + C-F2-safe也在128通过，额外预算节省为 **0条**。H及其他组合128以内未通过；不是“所有方法都失败”的分支。64和128之间未测试，因此128是网格中的首次通过值，不是任意整数预算的精确最小值。

必要目标base_u/v10_local在R + E-safe、B128分别相对η10提高 **+26.47pp / +22.38pp**，合并 **+24.42pp**，同时95%CI **[+6.91, +42.78]pp**。7目标平均提高 **+7.25pp**，同时CI **[+2.24, +12.61]pp**；200次抽样的possible/certain E3以及any-sample target E3均为0，因此CT成立。

“安全正迁移”与“接近最佳固定η”是两个门：E-safe R在B128仍距base_u/v10的discovery固定最佳η约3.80pp/3.79pp保守上界，两个必要目标的 **B_2pp均>128**。五个非必要目标在η10基线已经距对应hindsight固定η≤2pp，故 **B_2pp=0**；整体安全正迁移仍需防止适应误伤它们。

按用户第五节：没有“H/C首次SPT预算更小且配对同时CI支持”的实例，故不支持“源知识可迁移为额外样本效率”。结果部分符合“少样本可行但H/C未带来额外价值”一行；**H明显有害，不能写H≈E或统计等价**。源知识的预设四头候选菜单R本身有工程价值：E-safe在R为128，U/A为>128；这是跨策略阈值的描述性比较，未做预注册跨策略因果CI，也不能当作critic学会了未见controller的证据。A的采集由H驱动，E-A也含源采集知识。

事实：H/C没有节省SPT/CT预算，H的部分高预算选择造成灾难；C-F2-safe在B128的平均收益低于E-safe。解释：强源混合曲线与目标分布不匹配可能限制适应。猜想：更弱的形状先验、目标总体成功水平与regime分离可能更合适，本轮未调整或追加这些方法。

## 预注册、复用与标签池

预注册UTC **2026-10-13T05:55:06.921183+00:00**，SHA256 `f5987985713acf18ada1d8f2d1ef712ceaad225b79ef0751df42b1904d61df1d`；先读AGENTS.md、round8、早期草稿及round4–7证据，再冻结PREREGISTRATION。早期草稿未执行，其256预算/S/C-FT不纳入本轮。

可复用路径：`../controller_behavior_repr_v1/` 的REPORT/HANDOFF/results及零样本终点；`../controller_pathway_fix_v1/` 的REAL6 F2-R3-PCA2与R0配置；`../controller_loco_real_flow_v1/` 的StageA/B标签、协议、静态特征、冻结R0及controller_prior；`../controller_q_transfer_round3_v1/` 的P1；`../controller_flow_blend_v1/` 的Stage0曲线、融合标签及粗略目标校准表。旧标签execution语义保持有效。新证据缺口只有pool预测与重复适应统计，没有任何训练或rollout必要性。

主面板7折，每折6源；必要base_u/v10_local。目标池：TRAIN128state×8掩码head×Q4=4096格，加P1 24state×K16×Q4=1536格，合计5632；P1按controller名称重排，不沿用其原数组顺序。主评价64loco_teststate seed4–7；hindsight η仅seed0–3的observed-valid curve选出。TRAIN、P1、test的UID及physical内容两两无交集，controller checkpoint/horizon/dt/FF/success/conditioning核对通过，原完整execution payload保存于POOL_AUDIT。

| 目标 | 可采集格 | unknown | TRAIN格 | P1格 |
| --- | --- | --- | --- | --- |
| plus_w_v4_wide | 5632 | 70 | 4096 | 1536 |
| base_u | 5632 | 72 | 4096 | 1536 |
| plus_w_v3 | 5632 | 74 | 4096 | 1536 |
| plus_w_v4_train_targeted | 5632 | 85 | 4096 | 1536 |
| v10_local | 5632 | 26 | 4096 | 1536 |
| plus_w_v4 | 5632 | 69 | 4096 | 1536 |
| v9_local | 5632 | 15 | 4096 | 1536 |
| v4 | 4096 | 58 | 4096 | 0 |
| plus_w_v6 | 4096 | 50 | 4096 | 0 |
| v3 | 4096 | 82 | 4096 | 0 |
| v2 | 4096 | 61 | 4096 | 0 |
| plus_w_v5 | 4096 | 48 | 4096 | 0 |
| v7_local | 4096 | 8 | 4096 | 0 |
| v8_local | 4096 | 17 | 4096 | 0 |


StageB新增7Flow只用TRAIN4096格做目标池，其他13Flow作源，P1 24state作描述评价；source验证也用过P1，故不进入主判断。没有对应13源LOO的冻结R0/F2，StageB C明确INAPPLICABLE，只执行E/H及safe/P0。每η可用格/有效/unknown/state覆盖在POOL_AUDIT与pools/完整保留，全部方法/策略无“无法采集”事件。

200重复无放回，预算为同一长度128路径的前缀；每个目标/策略/重复的RNG固定为202610130800+target×10000+strategy×1000+repeat。所有适应方法共享标签，预算/策略/方法的配对关系保留。unknown消耗预算，后验和偏置均不按成功/失败更新；P1的Y=0只有在U=0时才是有效失败。

## 固定适应规则与冻结证据

P0固定源η10。E用每η Beta(1,1)，未采valid标签的head不参与，平局先η10再编号最小。E-safe需P(candidate>η10)≥.9。H为预注册的共轭mixture-Beta：每个源component的a=1+TRAIN成功seed数、b=1+失败数，controller先验等权；目标s/f既更新component权重的Beta-binomial边际似然，也与a/b相加。H-safe概率按同一controller混合分量中的两head联合后验计算。源每head约256条标签决定其浓度，**H并非任意强度可调的regime先验**；浓度/权重更新方式的局限保留，未根据终点弱化先验。

R取固定四候选{2,4,10,13}；这个菜单由任务/历史总体预声明，不声称每折6源都有四种最佳regime（StageA没有η4型）。A先从H posterior抽controller分量、再抽各head Beta值并选最大，在该head剩余格中均匀抽state/seed，完全没有未来标签搜索。

C-R0/C-F2用冻结3fit的成功概率q，offset=sum(y−q)/(n+8)，q'=clip(q+offset,0,1)。n只计valid目标标签，probability-space residual与round5的未收缩logit-offset不同；本轮n0=8固定。C每fit逐state选argmax并平均真实成功率，C-ctl每fit在评估state的无标签预测均值上选一个固定η。safe需3fit top1一致且平均优势≥.10，否则η10。评估初态可用于无标签平均，终点seed标签不进入这些规则。

冻结模型仅补152个poolstate预测，旧64state终点预测直接复用。R0前4state预测与旧cache max误差0；JAX x64=false，GPU通过Slurm单shard/1CPU推理；采集和统计用CPU最多14job×1CPU，资源提交前检查用户所有任务共同18CPU/3GPU上限。推理、14目标采集路径与选择先冻结ADAPTATION_FREEZE，再打开新终点统计。

## 统计与安全门

每预算统一max-T同时CI，覆盖12方法×3策略、单份P0、E1、必要合并/逐目标E2、预定H−E/C−E/C−Cctl配对以及每目标oracle差。主预算族491个效应（均含unknown上下界），没有按表现删臂；每个B各自一族，不宣称跨预算全局FWER。

5000次**精确交叉重抽**：同一64physicalstate权重跨7目标共享，每target对200次采集重复独立重抽200次，所有方法/策略/预算共享该target重抽权重；fit先平均、不作为独立样本。主种子202610130883、StageB描述202610130884。CI估计200重复平均性能，同时另报单次采集效果分位数与any-catastrophe，不是单次部署保证。

unknown按配对上下界：同η完全对消；不同η的lo=Ya−Yb−Ub、hi=Ya+Ua−Yb。E3 possible以每target/rep平均fit与state保守收益≤−10pp判断，certain用上界；平均E3和“任一次抽样”target比例都报告。SPT严格要求E1 CI下界>−2pp、possible E3平均=0、E2合并CI下界>0；CT再要求两个必要目标点收益>0。

### B_SPT / B_CT与预算不确定性

| 方法/策略 | B_SPT | B_CT | SPT bootstrap 2.5/50/97.5% | SPT右删失率 |
| --- | --- | --- | --- | --- |
| P0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / E | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / E-safe | >128 | >128 | ['>128', '>128', '>128'] | 99.46% |
| U / H | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / H-safe | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / C-R0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / C-R0-safe | >128 | >128 | ['>128', '>128', '>128'] | 99.92% |
| U / C-ctl-R0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / C-ctl-R0-safe | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| U / C-F2 | >128 | >128 | ['>128', '>128', '>128'] | 99.70% |
| U / C-F2-safe | >128 | >128 | [0, '>128', '>128'] | 72.52% |
| U / C-ctl-F2 | >128 | >128 | [0, '>128', '>128'] | 97.06% |
| U / C-ctl-F2-safe | >128 | >128 | [0, '>128', '>128'] | 97.06% |
| R / E | >128 | >128 | ['>128', '>128', '>128'] | 99.94% |
| R / E-safe | 128 | 128 | [64, 128, '>128'] | 3.32% |
| R / H | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| R / H-safe | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| R / C-R0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| R / C-R0-safe | >128 | >128 | ['>128', '>128', '>128'] | 99.90% |
| R / C-ctl-R0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| R / C-ctl-R0-safe | >128 | >128 | [128, '>128', '>128'] | 97.40% |
| R / C-F2 | >128 | >128 | ['>128', '>128', '>128'] | 99.64% |
| R / C-F2-safe | 128 | 128 | [0, 128, '>128'] | 24.38% |
| R / C-ctl-F2 | >128 | >128 | [0, '>128', '>128'] | 97.02% |
| R / C-ctl-F2-safe | >128 | >128 | [0, '>128', '>128'] | 82.70% |
| A / E | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / E-safe | >128 | >128 | ['>128', '>128', '>128'] | 99.36% |
| A / H | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / H-safe | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / C-R0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / C-R0-safe | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / C-ctl-R0 | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / C-ctl-R0-safe | >128 | >128 | ['>128', '>128', '>128'] | 100.00% |
| A / C-F2 | >128 | >128 | ['>128', '>128', '>128'] | 99.70% |
| A / C-F2-safe | >128 | >128 | [0, '>128', '>128'] | 72.54% |
| A / C-ctl-F2 | >128 | >128 | [0, '>128', '>128'] | 97.06% |
| A / C-ctl-F2-safe | >128 | >128 | [0, '>128', '>128'] | 97.06% |


预算bootstrap保留>128右删失，内部129仅编码而非实测预算；预算节省不把删失值当128。决策bootstrap按外层交叉重抽中心和重抽E3，保留原max-T CI宽度，并非嵌套再估CI。R + E-safe区间[64,>128]，R + C-F2-safe的SPT区间[0,>128]、CT区间[64,>128]反映阈值不稳定；原零样本F2在v9的损失接近−10pp硬阈值，state重抽可使E3门翻转。主点判定仍B128，不能从bootstrap偶发B0改写为零样本通过。

### 预算学习曲线（安全方法主表）

| B | 策略/方法 | 7目标success下界 | E1 vs prior同时CI | 必要E2保守收益/同时CI | E3平均possible/certain | any-sample目标possible比例 | SPT/CT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | R / E-safe | 60.21% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0.00% / 0.00% | 0.00% | False/False |
| 0 | R / H-safe | 60.21% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0.00% / 0.00% | 0.00% | False/False |
| 0 | R / C-R0-safe | 60.38% | [-0.25, +0.56]pp | +0.00pp [+0.00, +0.00]pp | 0.00% / 0.00% | 0.00% | False/False |
| 0 | R / C-F2-safe | 61.27% | [-1.17, +3.28]pp | +8.79pp [+1.12, +16.45]pp | 14.29% / 14.29% | 14.29% | False/False |
| 0 | R / C-ctl-F2-safe | 61.66% | [-1.28, +4.18]pp | +13.09pp [+4.11, +22.06]pp | 14.29% / 14.29% | 14.29% | False/False |
| 4 | R / E-safe | 60.40% | [-0.40, +0.76]pp | +1.13pp [-0.58, +2.91]pp | 0.71% / 0.71% | 85.71% | False/False |
| 4 | R / H-safe | 59.97% | [-0.61, +0.11]pp | -0.25pp [-1.15, +0.64]pp | 1.14% / 1.14% | 42.86% | False/False |
| 4 | R / C-R0-safe | 59.71% | [-1.32, +0.25]pp | +0.63pp [-0.62, +1.88]pp | 2.00% / 1.79% | 71.43% | False/False |
| 4 | R / C-F2-safe | 61.39% | [-1.28, +3.60]pp | +8.33pp [-0.79, +17.46]pp | 3.57% / 3.57% | 71.43% | False/False |
| 4 | R / C-ctl-F2-safe | 62.03% | [-1.50, +5.13]pp | +11.70pp [+0.35, +23.07]pp | 9.29% / 9.29% | 71.43% | False/False |
| 8 | R / E-safe | 60.91% | [-0.40, +1.76]pp | +3.30pp [+0.00, +6.76]pp | 1.43% / 1.43% | 71.43% | False/False |
| 8 | R / H-safe | 58.62% | [-2.73, -0.50]pp | -0.21pp [-1.62, +1.21]pp | 7.21% / 7.21% | 57.14% | False/False |
| 8 | R / C-R0-safe | 60.26% | [-1.06, +1.09]pp | +2.47pp [-0.67, +5.63]pp | 1.71% / 1.50% | 71.43% | False/False |
| 8 | R / C-F2-safe | 61.95% | [-0.70, +4.13]pp | +9.51pp [+0.80, +18.29]pp | 2.71% / 2.57% | 71.43% | False/False |
| 8 | R / C-ctl-F2-safe | 62.64% | [-0.72, +5.56]pp | +12.13pp [+1.30, +23.08]pp | 6.50% / 6.50% | 57.14% | False/False |
| 16 | R / E-safe | 62.02% | [+0.07, +3.50]pp | +6.76pp [+1.11, +12.67]pp | 1.07% / 1.07% | 85.71% | False/False |
| 16 | R / H-safe | 56.56% | [-5.74, -1.62]pp | -0.44pp [-2.08, +1.21]pp | 15.07% / 15.07% | 85.71% | False/False |
| 16 | R / C-R0-safe | 61.27% | [-0.78, +2.82]pp | +6.37pp [+0.61, +12.24]pp | 2.07% / 1.79% | 71.43% | False/False |
| 16 | R / C-F2-safe | 62.60% | [-0.24, +4.93]pp | +11.17pp [+2.40, +20.12]pp | 2.21% / 2.14% | 71.43% | False/False |
| 16 | R / C-ctl-F2-safe | 63.14% | [-0.16, +5.97]pp | +12.88pp [+2.45, +23.60]pp | 4.79% / 4.79% | 71.43% | False/False |
| 32 | R / E-safe | 63.93% | [+0.90, +6.45]pp | +12.67pp [+3.27, +22.54]pp | 0.50% / 0.50% | 57.14% | False/False |
| 32 | R / H-safe | 54.83% | [-8.30, -2.52]pp | +0.09pp [-1.10, +1.30]pp | 22.71% / 22.71% | 71.43% | False/False |
| 32 | R / C-R0-safe | 63.13% | [+0.06, +5.65]pp | +11.94pp [+2.80, +21.39]pp | 1.00% / 0.79% | 71.43% | False/False |
| 32 | R / C-F2-safe | 63.82% | [+0.41, +6.68]pp | +13.95pp [+3.65, +24.63]pp | 0.86% / 0.79% | 57.14% | False/False |
| 32 | R / C-ctl-F2-safe | 64.85% | [+0.95, +8.26]pp | +16.90pp [+4.44, +29.87]pp | 1.71% / 1.71% | 71.43% | False/False |
| 64 | R / E-safe | 65.68% | [+1.52, +9.27]pp | +18.25pp [+5.13, +31.99]pp | 0.36% / 0.36% | 28.57% | False/False |
| 64 | R / H-safe | 53.35% | [-10.57, -3.21]pp | -0.09pp [-0.74, +0.55]pp | 29.00% / 29.00% | 57.14% | False/False |
| 64 | R / C-R0-safe | 64.60% | [+0.32, +8.29]pp | +17.48pp [+4.59, +30.86]pp | 1.86% / 1.79% | 42.86% | False/False |
| 64 | R / C-F2-safe | 64.15% | [+0.44, +7.26]pp | +14.23pp [+3.04, +25.93]pp | 0.21% / 0.21% | 14.29% | False/False |
| 64 | R / C-ctl-F2-safe | 65.52% | [+1.10, +9.41]pp | +18.45pp [+4.27, +33.28]pp | 0.50% / 0.50% | 42.86% | False/False |
| 128 | R / E-safe | 67.75% | [+2.24, +12.61]pp | +24.42pp [+6.91, +42.78]pp | 0.00% / 0.00% | 0.00% | True/True |
| 128 | R / H-safe | 53.33% | [-10.83, -2.94]pp | +1.16pp [-0.69, +3.01]pp | 31.14% / 31.14% | 57.14% | False/False |
| 128 | R / C-R0-safe | 65.51% | [+0.27, +10.08]pp | +20.97pp [+5.30, +37.40]pp | 2.57% / 2.43% | 28.57% | False/False |
| 128 | R / C-F2-safe | 63.89% | [+0.05, +7.09]pp | +12.81pp [+1.28, +24.94]pp | 0.00% / 0.00% | 0.00% | True/True |
| 128 | R / C-ctl-F2-safe | 65.43% | [+0.78, +9.53]pp | +17.70pp [+2.76, +33.35]pp | 0.14% / 0.14% | 28.57% | False/False |


![Safe curves](figures/learning_curves_safe.png)

![Raw curves](figures/learning_curves_raw.png)

图按U/R/A与方法展示，success、E3平均和any-sample均将必要/非必要目标分开；全部数值及同时CI见endpoints/和CSV。

![Simultaneous effect bands](figures/R_safe_simultaneous_effects.png)

### 每目标B_2pp与最佳固定η差距

| 方法/策略 | 目标 | 必要 | B_2pp点指标 | 同时CI认证预算 |
| --- | --- | --- | --- | --- |
| P0 | plus_w_v4_wide | False | 0 | 0 |
| P0 | base_u | True | >128 | >128 |
| P0 | plus_w_v3 | False | 0 | 0 |
| P0 | plus_w_v4_train_targeted | False | 0 | 0 |
| P0 | v10_local | True | >128 | >128 |
| P0 | plus_w_v4 | False | 0 | 0 |
| P0 | v9_local | False | 0 | 0 |
| R / E-safe | plus_w_v4_wide | False | 0 | 0 |
| R / E-safe | base_u | True | >128 | >128 |
| R / E-safe | plus_w_v3 | False | 0 | 0 |
| R / E-safe | plus_w_v4_train_targeted | False | 0 | 0 |
| R / E-safe | v10_local | True | >128 | >128 |
| R / E-safe | plus_w_v4 | False | 0 | 0 |
| R / E-safe | v9_local | False | 0 | 0 |
| R / H-safe | plus_w_v4_wide | False | 0 | 0 |
| R / H-safe | base_u | True | >128 | >128 |
| R / H-safe | plus_w_v3 | False | 0 | 0 |
| R / H-safe | plus_w_v4_train_targeted | False | 0 | 0 |
| R / H-safe | v10_local | True | >128 | >128 |
| R / H-safe | plus_w_v4 | False | 0 | 0 |
| R / H-safe | v9_local | False | 0 | 0 |
| R / C-F2-safe | plus_w_v4_wide | False | 0 | 0 |
| R / C-F2-safe | base_u | True | >128 | >128 |
| R / C-F2-safe | plus_w_v3 | False | 0 | 0 |
| R / C-F2-safe | plus_w_v4_train_targeted | False | 0 | 0 |
| R / C-F2-safe | v10_local | True | >128 | >128 |
| R / C-F2-safe | plus_w_v4 | False | 0 | 0 |
| R / C-F2-safe | v9_local | False | 32 | >128 |
| R / C-ctl-F2-safe | plus_w_v4_wide | False | 0 | 0 |
| R / C-ctl-F2-safe | base_u | True | >128 | >128 |
| R / C-ctl-F2-safe | plus_w_v3 | False | 0 | 0 |
| R / C-ctl-F2-safe | plus_w_v4_train_targeted | False | 0 | 0 |
| R / C-ctl-F2-safe | v10_local | True | 0 | 0 |
| R / C-ctl-F2-safe | plus_w_v4 | False | 0 | 0 |
| R / C-ctl-F2-safe | v9_local | False | 32 | 64 |


全37组合×7目标的B_2pp、bootstrap区间、删失率在SAMPLE_EFFICIENCY.json及tables/target_B2pp.csv。point指标用200重复平均paired保守oracle差≥−2pp，认证版本再要求预算族同时CI下界≥−2pp；两者不混用。oracle由seed0–3选后在seed4–7评价，本身也可能不优于η10；oracle−prior保守分母≤0的gap-closed记NA，其他比例不clip0..1。

| B128组合 | 必要目标 | vs prior下界 | oracle gap保守上界 | 固定η差距闭合比例 |
| --- | --- | --- | --- | --- |
| R / E-safe | base_u | +26.47pp | +3.80pp | 88.00% |
| R / H-safe | base_u | +0.00pp | +31.64pp | 0.00% |
| R / C-F2-safe | base_u | +12.97pp | +17.47pp | 43.11% |
| R / C-ctl-F2-safe | base_u | +15.37pp | +15.11pp | 51.10% |
| R / E-safe | v10_local | +22.38pp | +3.79pp | 85.50% |
| R / H-safe | v10_local | +2.33pp | +23.85pp | 8.89% |
| R / C-F2-safe | v10_local | +12.66pp | +13.51pp | 48.37% |
| R / C-ctl-F2-safe | v10_local | +20.02pp | +6.15pp | 76.50% |


## 源知识节省了多少标签

| 策略 | 源方法 | vs E的SPT预算节省 | vs E的CT预算节省 | 预算节省SUPPORTED |
| --- | --- | --- | --- | --- |
| U | H | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-R0 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-F2 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-ctl-R0 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-ctl-F2 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | H-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-R0-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-F2-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-ctl-R0-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| U | C-ctl-F2-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| R | H | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| R | C-R0 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| R | C-F2 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| R | C-ctl-R0 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| R | C-ctl-F2 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| R | H-safe | <0 | <0 | False |
| R | C-R0-safe | <0 | <0 | False |
| R | C-F2-safe | 0 | 0 | False |
| R | C-ctl-R0-safe | <0 | <0 | False |
| R | C-ctl-F2-safe | <0 | <0 | False |
| A | H | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-R0 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-F2 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-ctl-R0 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-ctl-F2 | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | H-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-R0-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-F2-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-ctl-R0-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |
| A | C-ctl-F2-safe | INCONCLUSIVE: both >128 | INCONCLUSIVE: both >128 | False |


只有首次SPT/CT更早、且该预算相对E的7目标配对成功差同时CI下界>0才判样本效率SUPPORTED，本轮无实例。R C-F2-safe与E-safe都是128，节省0；H>128、E-safe为128，H的节省为<0且右删失，不能给精确负预算。两个方法都>128时节省INCONCLUSIVE。完整删失率与条件两者未删失的预算差区间保存在JSON，条件区间不能当全体预算差区间。

| R策略 B128配对比较 | 7目标差保守下界 | 同时95%CI |
| --- | --- | --- |
| R / H-safe-E-safe | -14.64pp | [-20.89, -7.87]pp |
| R / C-R0-safe-E-safe | -2.35pp | [-4.33, -0.20]pp |
| R / C-F2-safe-E-safe | -3.91pp | [-6.71, -0.97]pp |
| R / C-ctl-F2-safe-E-safe | -2.35pp | [-4.14, -0.38]pp |


R H-safe较E-safe约−14.64pp，CI[-20.89,−7.87]pp；R C-F2-safe约−3.91pp，CI[-6.71,−0.97]pp。它们在大预算下不等价于E，更没有建立额外选择优势。

## 逐state选择是否有价值

| B | 策略/同骨干比较 | 逐state−controller保守收益 | 同时95%CI | 判定 |
| --- | --- | --- | --- | --- |
| 0 | U / C-R0-C-ctl-R0 | -4.04pp | [-7.12, -0.65]pp | SUPPORTED nonpositive statewise value |
| 0 | U / C-R0-safe-C-ctl-R0-safe | +0.11pp | [-0.25, +0.56]pp | INCONCLUSIVE |
| 0 | U / C-F2-C-ctl-F2 | -2.53pp | [-4.92, +0.02]pp | INCONCLUSIVE |
| 0 | U / C-F2-safe-C-ctl-F2-safe | -0.45pp | [-2.43, +1.60]pp | INCONCLUSIVE |
| 0 | R / C-R0-C-ctl-R0 | -4.04pp | [-7.12, -0.65]pp | SUPPORTED nonpositive statewise value |
| 0 | R / C-R0-safe-C-ctl-R0-safe | +0.11pp | [-0.25, +0.56]pp | INCONCLUSIVE |
| 0 | R / C-F2-C-ctl-F2 | -2.53pp | [-4.92, +0.02]pp | INCONCLUSIVE |
| 0 | R / C-F2-safe-C-ctl-F2-safe | -0.45pp | [-2.43, +1.60]pp | INCONCLUSIVE |
| 0 | A / C-R0-C-ctl-R0 | -4.04pp | [-7.12, -0.65]pp | SUPPORTED nonpositive statewise value |
| 0 | A / C-R0-safe-C-ctl-R0-safe | +0.11pp | [-0.25, +0.56]pp | INCONCLUSIVE |
| 0 | A / C-F2-C-ctl-F2 | -2.53pp | [-4.92, +0.02]pp | INCONCLUSIVE |
| 0 | A / C-F2-safe-C-ctl-F2-safe | -0.45pp | [-2.43, +1.60]pp | INCONCLUSIVE |
| 16 | U / C-R0-C-ctl-R0 | -0.48pp | [-3.45, +2.77]pp | INCONCLUSIVE |
| 16 | U / C-R0-safe-C-ctl-R0-safe | -0.20pp | [-1.04, +0.68]pp | INCONCLUSIVE |
| 16 | U / C-F2-C-ctl-F2 | -1.47pp | [-4.08, +1.33]pp | INCONCLUSIVE |
| 16 | U / C-F2-safe-C-ctl-F2-safe | -0.88pp | [-3.03, +1.30]pp | INCONCLUSIVE |
| 16 | R / C-R0-C-ctl-R0 | -1.12pp | [-3.36, +1.34]pp | INCONCLUSIVE |
| 16 | R / C-R0-safe-C-ctl-R0-safe | +0.03pp | [-1.03, +1.16]pp | INCONCLUSIVE |
| 16 | R / C-F2-C-ctl-F2 | -1.03pp | [-3.15, +1.24]pp | INCONCLUSIVE |
| 16 | R / C-F2-safe-C-ctl-F2-safe | -0.57pp | [-2.40, +1.30]pp | INCONCLUSIVE |
| 16 | A / C-R0-C-ctl-R0 | +0.69pp | [-1.28, +2.83]pp | INCONCLUSIVE |
| 16 | A / C-R0-safe-C-ctl-R0-safe | +0.74pp | [-1.08, +2.74]pp | INCONCLUSIVE |
| 16 | A / C-F2-C-ctl-F2 | -1.00pp | [-3.11, +1.17]pp | INCONCLUSIVE |
| 16 | A / C-F2-safe-C-ctl-F2-safe | +0.12pp | [-1.31, +1.65]pp | INCONCLUSIVE |
| 64 | U / C-R0-C-ctl-R0 | -0.72pp | [-3.06, +1.84]pp | INCONCLUSIVE |
| 64 | U / C-R0-safe-C-ctl-R0-safe | -0.31pp | [-1.47, +0.92]pp | INCONCLUSIVE |
| 64 | U / C-F2-C-ctl-F2 | -1.00pp | [-3.20, +1.35]pp | INCONCLUSIVE |
| 64 | U / C-F2-safe-C-ctl-F2-safe | -0.79pp | [-2.60, +1.07]pp | INCONCLUSIVE |
| 64 | R / C-R0-C-ctl-R0 | -2.03pp | [-4.70, +0.82]pp | INCONCLUSIVE |
| 64 | R / C-R0-safe-C-ctl-R0-safe | -0.98pp | [-2.68, +0.80]pp | INCONCLUSIVE |
| 64 | R / C-F2-C-ctl-F2 | -1.55pp | [-3.82, +0.83]pp | INCONCLUSIVE |
| 64 | R / C-F2-safe-C-ctl-F2-safe | -1.42pp | [-3.46, +0.68]pp | INCONCLUSIVE |
| 64 | A / C-R0-C-ctl-R0 | +1.11pp | [-0.42, +2.72]pp | INCONCLUSIVE |
| 64 | A / C-R0-safe-C-ctl-R0-safe | +1.69pp | [-0.98, +4.57]pp | INCONCLUSIVE |
| 64 | A / C-F2-C-ctl-F2 | -0.96pp | [-3.15, +1.27]pp | INCONCLUSIVE |
| 64 | A / C-F2-safe-C-ctl-F2-safe | +0.25pp | [-1.24, +1.82]pp | INCONCLUSIVE |
| 128 | U / C-R0-C-ctl-R0 | -1.08pp | [-3.01, +1.02]pp | INCONCLUSIVE |
| 128 | U / C-R0-safe-C-ctl-R0-safe | -0.25pp | [-1.51, +1.08]pp | INCONCLUSIVE |
| 128 | U / C-F2-C-ctl-F2 | -0.97pp | [-2.96, +1.13]pp | INCONCLUSIVE |
| 128 | U / C-F2-safe-C-ctl-F2-safe | -0.82pp | [-2.59, +0.99]pp | INCONCLUSIVE |
| 128 | R / C-R0-C-ctl-R0 | -1.83pp | [-4.90, +1.45]pp | INCONCLUSIVE |
| 128 | R / C-R0-safe-C-ctl-R0-safe | -1.71pp | [-3.64, +0.35]pp | INCONCLUSIVE |
| 128 | R / C-F2-C-ctl-F2 | -1.60pp | [-4.14, +1.06]pp | INCONCLUSIVE |
| 128 | R / C-F2-safe-C-ctl-F2-safe | -1.61pp | [-3.91, +0.77]pp | INCONCLUSIVE |
| 128 | A / C-R0-C-ctl-R0 | +0.91pp | [-0.43, +2.29]pp | INCONCLUSIVE |
| 128 | A / C-R0-safe-C-ctl-R0-safe | +2.41pp | [-0.72, +5.79]pp | INCONCLUSIVE |
| 128 | A / C-F2-C-ctl-F2 | -1.32pp | [-3.67, +1.09]pp | INCONCLUSIVE |
| 128 | A / C-F2-safe-C-ctl-F2-safe | +0.18pp | [-1.42, +1.85]pp | INCONCLUSIVE |


B>0没有C−C-ctl成功率配对差的同时CI下界>0，故额外逐state成功率价值 **INCONCLUSIVE**。R B128 C-F2-safe−C-ctl-F2-safe点差−1.61pp，CI[-3.91,+0.77]pp；符合controller固定选择点均值更高的描述，但没有证明普遍无价值。另一方面C-F2-safe在R B128的E3=0，C-ctl-F2-safe仍有2/1400个target×repeat灾难（两目标各一次），因此后者未过SPT；均值更高与rare catastrophe并存，不能用一个指标覆盖另一个。

## StageB描述扩展

其他7Flow作目标、13源混合，目标pool只TRAIN，评价P1。历史源模型验证使用同一P1开发分布，本轮H源分量本身仍只由TRAIN构造。StageB的oracle在同一P1数据选出和评价，含hindsight选择偏差；仅作描述参照。所有预算的E/H及safe结果见stageB_B*.json和CSV，以下仅列128，均不进SPT/CT主结论。

| 方法/策略 | success下界 | 相对源prior保守差/同时CI | E3平均possible | any-sample目标比例 |
| --- | --- | --- | --- | --- |
| P0 | 59.82% | +0.00pp [+0.00, +0.00]pp | 0.00% | 0.00% |
| U / E | 58.33% | -1.80pp [-9.02, +5.83]pp | 14.86% | 85.71% |
| U / E-safe | 60.33% | +0.39pp [-1.71, +2.66]pp | 1.50% | 85.71% |
| U / H | 60.76% | +0.62pp [-5.68, +7.45]pp | 10.50% | 42.86% |
| U / H-safe | 62.26% | +2.14pp [-3.13, +7.89]pp | 1.21% | 28.57% |
| R / E | 60.82% | +0.70pp [-5.82, +7.65]pp | 6.14% | 42.86% |
| R / E-safe | 61.74% | +1.68pp [-2.55, +6.30]pp | 0.79% | 14.29% |
| R / H | 61.78% | +1.65pp [-3.92, +7.73]pp | 4.14% | 42.86% |
| R / H-safe | 62.30% | +2.18pp [-3.06, +7.90]pp | 1.00% | 28.57% |
| A / E | 59.24% | -0.89pp [-7.61, +6.24]pp | 9.21% | 85.71% |
| A / E-safe | 60.81% | +0.83pp [-1.87, +3.77]pp | 1.07% | 42.86% |
| A / H | 62.03% | +1.93pp [-2.94, +7.25]pp | 1.21% | 28.57% |
| A / H-safe | 62.18% | +2.08pp [-2.68, +7.28]pp | 0.50% | 28.57% |


![StageB descriptive](figures/StageB_descriptive_curves.png)

## 成本与研究路线

R5每controller32×200=6400步且无成功标签。按H700上限换算 **9.14条** 完整rollout步数；按StageA所有TRAIN请求的实际记录episode_length均值，约 **9.25–10.65条**。这是模拟步数描述，未计模型/安全求解开销，标签信息不等价；这里只引用真实TRAIN raw，不混入P1或终点来选择策略。

| 目标 | 真实TRAIN记录数 | 平均recorded steps | R5步数等价完整请求数 |
| --- | --- | --- | --- |
| plus_w_v4_wide | 4096 | 678.74 | 9.43 |
| base_u | 4096 | 682.99 | 9.37 |
| plus_w_v3 | 4096 | 620.26 | 10.32 |
| plus_w_v4_train_targeted | 4096 | 691.08 | 9.26 |
| v10_local | 4096 | 608.31 | 10.52 |
| plus_w_v4 | 4096 | 692.12 | 9.25 |
| v9_local | 4096 | 600.73 | 10.65 |


round5的粗略固定前kstate校准表对应3个必要融合controller，k1=32完整H700且需要目标标签。本轮7真实目标、200次重复、多种随机采集、unknown与严格安全门，取代其样本效率估计；不能拿round5的83.33%融合绝对成功率与本轮真实Flow直接比较。

对部署研究最小路线是 **预设少数η候选 + 少量目标标签 + 明确安全回退**。当前可审查候选为R + E-safe B128：对新真实controller随机采128条来自预设四头的标签，冻结Beta规则；未来需要新controller/新state确认。本轮支持开发面板上的互补路线，没有支持加入冻结critic或强源mixture能进一步减预算。对非必要目标，η10零标签已经够好；实际部署缺乏目标必要性真值，统一适应仍需避免误伤。

新的独立η13真实controller仍有研究价值，可用于检验源覆盖/先验misspecification，但当前证据不足以优先宣称它必然比目标适应更省：按旧128state×8head×Q4加24state×K16×Q4源标签设计，每个新controller5632完整episode；两个为11264完整episode+64短probe，Flow训练成本另计。仅标签数量已相当于88个目标各128条适应，尚未计新Flow训练。应先用已有源内资料研究弱混合/目标成功水平与regime分离等结构，再单独预注册新覆盖确认。

前瞻最小确认建议：4个全新真实Flow（至少2个候选η13型、2个其他型），冻结R E-safe与P0/R C-F2-safe对照；每target采128条来自全新采集state的四头标签，共512；另用全新64state×K16×Q4终点评价，共16384；独立24state×{η10,η13}×Q2作必要性发现共384。总计 **17280有标签H700 episode**，本轮未运行；是否额外采32短probe/controller取决于新预注册对照，不把R5作为当前必要成本。

## 假设、限制与交付审计

| 假设 | 判定 |
| --- | --- |
| H_fewshot_safe_positive_transfer | SUPPORTED DEVELOPMENT |
| H_complete_transfer_on_both_necessary_controllers | SUPPORTED DEVELOPMENT at B128 R E-safe and C-F2-safe |
| H_safe_transfer_with_B64_or_less | REJECTED under registered methods and 200 repetitions |
| H_H_prior_saves_labels | REJECTED under registered mixture-Beta implementation |
| H_critic_saves_labels | REJECTED under registered frozen-offset implementation |
| H_statewise_additional_success_value | INCONCLUSIVE: no positive simultaneous CI at B>0 |
| H_R_shortlist_acquisition_helpful | SUPPORTED DESCRIPTIVE: E-safe R reaches128; U/A >128; no preregistered cross-strategy causal CI |
| H_within_2pp_of_oracle_for_both_necessary_by128 | REJECTED for E-safe R |
| H_source_knowledge_has_no_possible_value | INCONCLUSIVE |
| H_independent_new_eta13_source_controller_value | INCONCLUSIVE |
| H_prospective_confirmation | INCONCLUSIVE |


边界为Ring、FF、H700、K16、固定几何、仅2个必要真实Flow；已打开数据全部DEVELOPMENT。采集pool为两类state分布混合且head稀疏，跨η的state组成不同；模拟无放回与真实新采集环境的差异保留。0次样本灾难只描述200重复经验结果，不证明新controller风险恒为0；预算CI估均值且硬E3阈值有不稳定性。H结论仅对应本次源count浓度的共轭mixture，C仅概率offset与冻结骨干，不能否定所有源知识或controller–Q结构可学性。StageB共享P1开发分布不作确认。

FINAL_AUDIT **PASS**：原预注册/采集冻结hash不变，1588历史输入hash不变；主/描述pool无放回、unknown计数正确，source排除目标；H后验权重独立重算max误差9.8e-13，offset独立重算max误差2.55e-08；R0/F2零预算与round7逐目标success/gain精确一致；交叉bootstrap与独立标量计算一致。没有DB连接，不作共享DB全局不变声明。所有文件只在本目录，无commit。

文件入口：PREREGISTRATION及收据；POOL_AUDIT、PROTOCOL、HISTORICAL_ASSET_INDEX；pools/、predictions/及PREDICTION_RECEIPT；simulations/的paths、计数、posterior、offsets、choices；ADAPTATION_FREEZE；endpoints/的原/带E3CI结果和bootstrap；SAMPLE_EFFICIENCY/BUDGET_BOOTSTRAP_DRAWS；figures/ PNG-PDF、tables/ CSV；results.json、REPORT、HANDOFF、FINAL_AUDIT。不依赖本对话即可复算。

冻结结果的统计复算命令（无训练、无rollout、无DB）：

```bash
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ROUND8_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
# bi=0..6对应冻结预算；通过Slurm或当前允许CPU余量执行。
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/evaluate.py 6 main
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/evaluate.py 6 stageB
# 完整预算统计已在endpoints/；汇总不会重做采集。
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/efficiency.py finalize
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/audit.py
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/figures.py
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/report.py
```

完整重算全部预算时逐bi运行evaluate.py与efficiency.py bi，再finalize；不要重新调用setup覆盖原freeze，不要运行历史rollout/probe/merger，不要把已打开终点调参后的新策略称确认。

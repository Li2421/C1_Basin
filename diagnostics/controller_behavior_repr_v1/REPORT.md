# Round7：闭环行为表示 × 锚点邻域覆盖

状态 **COMPLETE**；UTC 2026-10-13T05:43:40.033473+00:00。全部证据 **DEVELOPMENT**。新增rollout **0**，新增probe **0**，共享DB写入 **0**，DB连接 **0**。纯Q实验，无视频验收要求。

## 结论与证据边界

**表示（探测设计）仍不足：当前200步、两个η的R5没有建立真实Flow regime的可迁移编码。** 按预注册第四节映射：真实LOCO四格均未达到SPT；真实14Flow的I1主检验ρ=-0.297，95%CI [-0.854, 0.347]，下界不大于0；P_rule也未达到SPT。因此没有走“信息存在但学习通路没用上”的分支。CT亦未成立。

这里的“表示不足”是本次固定probe、表示和学习规则下的操作性结论。ρ的CI跨零不证明真实相关严格为零，也不证明160维原始轨迹没有任何非线性信息；不能把“不支持正单调信息”写成信息论意义的无信息。覆盖臂失败只否定本次锚点邻域融合增广的充分性；增加独立η13型真实controller仍为 **INCONCLUSIVE**。

事实：R5与覆盖增广在许多规则下恢复了安全回退，却没有提高两个必要真实Flow的选择收益。解释：200步进展差可能偏好短期快速运动，但未反映H700是否能完成交换。猜想：后段停滞、绕障进度或投影持续性可能更有用；这些没有在本轮作为新特征试验。

| 表示判定族 | 覆盖 | SPT通过 | 通过臂 | CT通过臂 |
| --- | --- | --- | --- | --- |
| R3 | REAL6 | False | 无 | 无 |
| R5 | REAL6 | False | 无 | 无 |
| R3 | REAL6+BLEND | False | 无 | 无 |
| R5 | REAL6+BLEND | False | 无 | 无 |


R3判定族为3条主要规则；R5判定族为PCA2/Δprog×3规则。R5-Dprog是文件名中对R5-Δprog的ASCII拼写。不得把argmax或κ=.5描述性结果升级成主结论。

## 预注册、历史复用与实现

先读取AGENTS.md及round4/5/6协议、报告、代码与冻结资产，随后在 2026-10-13T05:09:44.888555+00:00 留下PREREGISTRATION.md及sha256收据：`05314f4387a577ea32b5d5c9d4f5e2156f1bb01ec1eceea3c85d569c65029463`。新统计和重放均晚于收据。完整用户设计原文已嵌入预注册。

复用路径：`../controller_pathway_fix_v1/` 的R0/F2/F4配置、checkpoint与预测；`../controller_flow_blend_v1/` 的TRAIN/终点标签、Stage0曲线、96state静态特征及R5 raw；`../controller_loco_real_flow_v1/` 的真实7折标签、R0与128state静态特征；`../controller_q_transfer_round3_v1/` 的P1 24state验证集。execution语义未变，旧标签保持有效。缺口只有新表示/源集合的小型模型，不需要新rollout或Flow查询。3068项历史资产hash见HISTORICAL_ASSET_INDEX.json。

独立核对发现任务说明的一处历史事实差异：round6真实LOCO的R3均值实际为128 TRAIN state，融合LOPO为96；源码及round6预注册一致。本轮R3×REAL6精确复用128state旧fit；REAL6+BLEND保留真实Flow128state均值，融合取其已有96state均值；阶段1描述均取共同96state。这保留了真实Flow的表示输入，但融合特征的平均state组成不同，是覆盖比较的额外局限。

F2冻结同折、同fit seed的R0，加16/16/16 SiLU残差，末层零初始化，tanh界1；AdamW lr0.0008、weight decay0.01、clip5、batch128头，12000step，seed17/23/41，3fit平均P1源验证NLL选共同step，JAX x64=false。REAL6+BLEND的R0、prior和P1验证只由原6源真实Flow决定。新增controller均满足两锚点在源中；训练与验证无目标controller，所有变换source-only。所有新预测及模型先冻结到MODEL_FREEZE.json，再执行新终点评价。

105个主真实seed fit、30个主融合seed fit、30个插补敏感性seed fit，合计 **165**。另复用R3旧fit，输出 **82** 个臂×折预测包（每包3seed）及全部checkpoint。相同SPARSE源模型每表示只优化一次，其余折共享SHA并各存预测；不重复训练。新F4及R0、prior、P_rule参数/选择见classifiers/、CLASSIFIERS_INDEX.json。首个正式3seed GPU fit耗时46.37s，作为正式结果保留。

资源按用户实时更新为所有任务共同18CPU/3GPU shard、本轮优先。实际真实LOCO105seed fit使用Slurm GPU；融合主/敏感性60seed fit使用Slurm CPU，每job1核、最多14CPU并发，另有3个GPU job各1CPU、预留1CPU编排。CPU/GPU仍保持x64=false，backend不同不声称bitwise等价，未根据终点挑选后端。原队列20个尚未启动的融合任务改为CPU调度，未删臂。CPU日志中的CUDA插件无设备信息是插件探测警告；实际backend=cpu，全部拟合完成，非rollout失败。资源修订先于新终点评价，见RESOURCE_EXECUTION_ADDENDUM及收据、SLURM_SUBMISSIONS.json。

阶段0后先执行了一个正式F2成本测量任务，阶段1的信息量分析与其独立运行；阶段1脚本本身只读TRAIN_ONLY.npz、无标签表示与旧Stage0曲线，不读任何终点seed，不拟合Q。这项执行顺序并行不参与方法选择，完整源验证预算固定。

## 阶段0：硬门复现与R5审计

15个task，R0、F2-PCA2悲观κ1/保守、F4的所有逐目标选择和成功率/收益与round6精确一致。R0最大logit误差 0；F2最大logit误差 0.000168；logit误差不是另加的选择门。所有主要选择一致才解锁后续阶段。STAGE0_REPRODUCTION.json保留逐项收据；FINAL_AUDIT另检查155条复用R3终点行及η直方图与round6一致。

R5形状(26,160) float32；排列为controller→state0..15→η10/13→[progress,projection_fraction,minimum_pair_distance,mean_speed,stalled_steps]。831条完整raw trace独立重算与raw统计最大误差 **0.0**；与float32 NPZ最大误差 **1.18936302e-07**，转float32后完全一致。25个controller各32个完整probe，base_u_v3为31个；唯一NaN为该行105..109，即state10/η13。失败CBFSolverError发生于132步，partial只作描述，不生成任何统计特征。无缺失零填充；stalled_steps中的829个真实零来自完整trace，不是插补。

| 统计量 | 有效格数 | min | median | mean | max |
| --- | --- | --- | --- | --- | --- |
| progress | 831 | 0.143 | 0.399 | 0.435 | 0.829 |
| projection_fraction | 831 | 0.160 | 1.000 | 0.879 | 1.000 |
| minimum_pair_distance | 831 | 0.323 | 0.995 | 1.126 | 2.426 |
| mean_speed | 831 | 0.150 | 0.345 | 0.340 | 0.517 |
| stalled_steps | 831 | 0.000 | 0.000 | 0.002 | 1.000 |


16probe state与R4全部192state、R5全部128state无交集；融合TRAIN96全部属于真实TRAIN128，和真实终点64state无交集。主方案所有controller删除state10/η13五维，R5为155维；所有差值统一使用另外15state。敏感性只在阶段1/3，用折内源且不含base_u_v3的均值插补该格，PCA160维、差值16state；不使用132步partial。

## 阶段1：controller级信息与识别

真值：14真实Flow采用历史Stage0 best_eta与observed_valid_curve；融合采用96 TRAIN state的valid curve，regime遵循round6四候选顺序[10,13,2,4]。各controller曲线、K16 best与regime见TRAIN_TRUTH.json。融合K16 best落在四候选外的数量为 0。**稀疏掩码使η10与η13在不同state子集上观测，ΔS包含state组成差异。**

### I1：零参数信息检验

| 人群 | 统计量差值 | ρ | 95%CI | 退化bootstrap次数 | 性质 |
| --- | --- | --- | --- | --- | --- |
| real | progress | -0.297 | [-0.854, 0.347] | 0 | 主检验 |
| real | projection_fraction | -0.035 | [-0.665, 0.585] | 0 | 探索性 |
| real | minimum_pair_distance | -0.161 | [-0.829, 0.524] | 0 | 探索性 |
| real | mean_speed | -0.240 | [-0.847, 0.438] | 0 | 探索性 |
| real | stalled_steps | -0.379 | [-0.720, -0.243] | 1829 | 探索性 |
| blend | progress | 0.007 | [-0.650, 0.771] | 0 | 主检验 |
| blend | projection_fraction | -0.556 | [-0.774, -0.036] | 24 | 探索性 |
| blend | minimum_pair_distance | -0.497 | [-0.796, 0.073] | 0 | 探索性 |
| blend | mean_speed | 0.301 | [-0.650, 0.956] | 0 | 探索性 |
| blend | stalled_steps | NA | [NA, NA] | 5000 | 探索性 |
| combined | progress | 0.252 | [-0.261, 0.620] | 0 | 主检验 |
| combined | projection_fraction | 0.024 | [-0.405, 0.484] | 0 | 探索性 |
| combined | minimum_pair_distance | -0.024 | [-0.521, 0.501] | 0 | 探索性 |
| combined | mean_speed | 0.281 | [-0.262, 0.635] | 0 | 探索性 |
| combined | stalled_steps | -0.307 | [-0.597, -0.254] | 1797 | 探索性 |


每项5000次bootstrap：真实Flow按controller，融合按pair，合并按14个单Flow簇+4个pair簇。常量重抽样记退化，剔除并报告数量；stalled_steps几乎恒为0，该探索性相关尤其不能按831个独立观测解读。主ρ未证明正关系，也不是等价性检验。

![Δprog vs ΔS](figures/I1_delta_progress_vs_delta_success.png)

### I2：固定ridge α=1的留出ΔS预测

| 表示 | 划分 | 留出Spearman | 符号准确率 |
| --- | --- | --- | --- |
| R3 | real | 0.484 | 71.43% |
| R3 | blend | 0.399 | 50.00% |
| R3-PCA2 | real | 0.427 | 85.71% |
| R3-PCA2 | blend | 0.175 | 41.67% |
| R4_action | real | 0.304 | 50.00% |
| R4_action | blend | -0.203 | 58.33% |
| R5 | real | 0.416 | 85.71% |
| R5 | blend | -0.112 | 41.67% |
| R5-PCA2 | real | 0.185 | 85.71% |
| R5-PCA2 | blend | 0.322 | 58.33% |
| R5-Dprog | real | -0.724 | 85.71% |
| R5-Dprog | blend | -0.168 | 50.00% |
| R3-PCA2+R5-PCA2 | real | 0.310 | 85.71% |
| R3-PCA2+R5-PCA2 | blend | 0.350 | 66.67% |


真实LOO仅13源；融合LOPO为14真实锚点+9个非留出pair融合，锚点保留。14真实Flow中ΔS正号仅2个，预测全负的符号准确率本就为12/14=85.71%；高符号准确率不能替代排序或少数类识别证据。

### I3：14个Flow的LOO regime分类

| 表示 | 分类器 | 准确率 | 相对64.29%多数类差 |
| --- | --- | --- | --- |
| R3 | 1NN | 64.29% | +0.00pp |
| R3 | logistic_C1 | 64.29% | +0.00pp |
| R3-PCA2 | 1NN | 57.14% | -7.14pp |
| R3-PCA2 | logistic_C1 | 57.14% | -7.14pp |
| R4_action | 1NN | 42.86% | -21.43pp |
| R4_action | logistic_C1 | 50.00% | -14.29pp |
| R5 | 1NN | 57.14% | -7.14pp |
| R5 | logistic_C1 | 57.14% | -7.14pp |
| R5-PCA2 | 1NN | 64.29% | +0.00pp |
| R5-PCA2 | logistic_C1 | 64.29% | +0.00pp |
| R5-Dprog | 1NN | 50.00% | -14.29pp |
| R5-Dprog | logistic_C1 | 64.29% | +0.00pp |
| R3-PCA2+R5-PCA2 | 1NN | 57.14% | -7.14pp |
| R3-PCA2+R5-PCA2 | logistic_C1 | 50.00% | -14.29pp |


没有预定表示/分类器超过9/14多数类；完整混淆矩阵、逐目标预测及source-majority结果在STAGE1.json和tables/I3_main.csv。

![Confusion matrices](figures/I3_confusion_main.png)

### I4：配方与regime的描述性预测

| 表示 | d(base,v10) | d(base,v9) | d(v10,v9) | 该表示预定关系成立 |
| --- | --- | --- | --- | --- |
| R3 | 34.653 | 34.522 | 27.454 | True |
| R3-PCA2 | 29.803 | 30.822 | 5.723 | True |
| R5-PCA2 | 20.768 | 24.372 | 10.700 | False |
| R5-Dprog | 3.326 | 1.734 | 1.592 | False |


H_static_recipe = **SUPPORTED**；H_behavior_regime = **REJECTED**。这是距离关系预测，不证明训练配方是静态描述的唯一因果因素。7×7距离表与每controller完整近邻排序在tables/I4_*；只用14真实Flow拟合描述性z-score/PCA。

| 面板Flow | raw Δprog | 符号 |
| --- | --- | --- |
| plus_w_v4_wide | 0.286235 | 1 |
| base_u | 0.080659 | 1 |
| plus_w_v3 | 0.131603 | 1 |
| plus_w_v4_train_targeted | 0.141287 | 1 |
| v10_local | 0.338254 | 1 |
| plus_w_v4 | 0.217841 | 1 |
| v9_local | 0.214951 | 1 |


7个面板Flow全部Δprog>0，P_rule因此对每个目标都选择η13；短期进展的正号无法区分η10/η13最终成功regime。

![Distances](figures/I4_panel_distances.png)

## 阶段2：真实Flow LOCO主检验

终点仅64个loco_test state × seed4–7。全部主要臂联合E1 max-T，5000次物理state成簇bootstrap，各controller与3fit先等权平均；不把同state的controller/seed当独立样本。真实主要族18臂+唯一P_rule，两个coverage中完全相同的P_rule仍逐行展示，重复相同列不改变max-T。E1下界须严格大于−2pp。E2每个R3三臂/R5六臂判定族上，联合vs prior及R0两比较做max-T；SPT要求两个下界都>0且possible E3=0。argmax/κ.5只在JSON与CSV描述，不参与判定。

### 覆盖与防泄漏

| 留出Flow | 总源C数 | 新增融合C数 | 新增regime数量 |
| --- | --- | --- | --- |
| plus_w_v4_wide | 12 | 6 | {10: 2, 13: 1, 2: 0, 4: 3} |
| base_u | 12 | 6 | {10: 3, 13: 2, 2: 0, 4: 1} |
| plus_w_v3 | 18 | 12 | {10: 4, 13: 5, 2: 0, 4: 3} |
| plus_w_v4_train_targeted | 18 | 12 | {10: 4, 13: 5, 2: 0, 4: 3} |
| v10_local | 12 | 6 | {10: 1, 13: 3, 2: 0, 4: 2} |
| plus_w_v4 | 18 | 12 | {10: 4, 13: 5, 2: 0, 4: 3} |
| v9_local | 12 | 6 | {10: 2, 13: 4, 2: 0, 4: 0} |


完整新增controller名称/所属pair/regime见COVERAGE.json。base_u折仅pair2/3，v10折仅pair0/1，v4_wide折仅pair1/3，v9折仅pair0/2，其余全4pair。该增广引入6或12个附近exemplar，并增加标签量；不是独立新η13型真实Flow。

### 给定controller时的η选择（主终点）

| 覆盖 | 基线 | 成功率下界 | 成功率上界 | 必要C相对prior保守收益 |
| --- | --- | --- | --- | --- |
| REAL6 | B0 | 54.89% | 55.60% | +1.30pp |
| REAL6 | source_prior | 60.21% | 61.22% | +0.00pp |
| REAL6+BLEND | B0 | 54.89% | 55.60% | +1.30pp |
| REAL6+BLEND | source_prior | 60.21% | 61.22% | +0.00pp |


| 覆盖 | 表示 | 规则 | 成功率下界 | E1 vs R0 同时CI | E2 vs prior 同时CI | E2 vs R0 同时CI | E3 possible/certain | SPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REAL6 | R3-PCA2 | pessimistic1 | 58.31% | [+0.62, +6.37]pp | +10.16pp [+3.30, +17.01]pp | +8.85pp [+2.18, +15.53]pp | 2/7 ; 2/7 | False |
| REAL6 | R3-PCA2 | conservative | 61.27% | [+3.15, +9.90]pp | +8.79pp [+3.02, +14.56]pp | +7.49pp [+1.82, +13.15]pp | 1/7 ; 1/7 | False |
| REAL6 | R3-PCA2 | F4 | 60.21% | [+2.50, +8.50]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.27, +0.67]pp | 0/7 ; 0/7 | False |
| REAL6 | P_rule | P_rule | 46.21% | [-15.65, -2.20]pp | +28.12pp [+19.28, +37.89]pp | +26.82pp [+18.43, +36.15]pp | 5/7 ; 5/7 | False |
| REAL6 | R5-PCA2 | pessimistic1 | 57.20% | [-0.04, +4.79]pp | -0.59pp [-1.90, +0.72]pp | -1.89pp [-3.87, +0.09]pp | 0/7 ; 0/7 | False |
| REAL6 | R5-PCA2 | conservative | 60.04% | [+2.34, +8.26]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |
| REAL6 | R5-PCA2 | F4 | 60.21% | [+2.50, +8.50]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |
| REAL6 | R5-Dprog | pessimistic1 | 54.97% | [-1.62, +1.85]pp | -0.59pp [-1.90, +0.72]pp | -1.89pp [-3.87, +0.09]pp | 1/7 ; 1/7 | False |
| REAL6 | R5-Dprog | conservative | 60.32% | [+2.68, +8.54]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |
| REAL6 | R5-Dprog | F4 | 60.21% | [+2.50, +8.50]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |
| REAL6+BLEND | R3-PCA2 | pessimistic1 | 55.80% | [-1.33, +3.25]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-2.93, +0.33]pp | 1/7 ; 1/7 | False |
| REAL6+BLEND | R3-PCA2 | conservative | 60.32% | [+2.61, +8.54]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-2.93, +0.33]pp | 0/7 ; 0/7 | False |
| REAL6+BLEND | R3-PCA2 | F4 | 60.21% | [+2.50, +8.50]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-2.93, +0.33]pp | 0/7 ; 0/7 | False |
| REAL6+BLEND | P_rule | P_rule | 46.21% | [-15.65, -2.20]pp | +28.12pp [+19.28, +37.89]pp | +26.82pp [+18.43, +36.15]pp | 5/7 ; 5/7 | False |
| REAL6+BLEND | R5-PCA2 | pessimistic1 | 54.46% | [-2.59, +1.83]pp | -0.59pp [-1.90, +0.72]pp | -1.89pp [-3.87, +0.09]pp | 1/7 ; 1/7 | False |
| REAL6+BLEND | R5-PCA2 | conservative | 60.27% | [+2.53, +8.50]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |
| REAL6+BLEND | R5-PCA2 | F4 | 54.85% | [-3.14, +3.31]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 1/7 ; 1/7 | False |
| REAL6+BLEND | R5-Dprog | pessimistic1 | 54.63% | [-1.95, +1.50]pp | -0.59pp [-1.90, +0.72]pp | -1.89pp [-3.87, +0.09]pp | 1/7 ; 1/7 | False |
| REAL6+BLEND | R5-Dprog | conservative | 60.38% | [+2.72, +8.54]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |
| REAL6+BLEND | R5-Dprog | F4 | 60.21% | [+2.50, +8.50]pp | +0.00pp [+0.00, +0.00]pp | -1.30pp [-3.19, +0.59]pp | 0/7 ; 0/7 | False |


R5保守/F4多数回退η10，在base_u与v10上相对prior收益为0；safe fallback没有证明Q学会controller作用。R3×REAL6保守规则在v10上+17.58pp、base_u为0，但v9上−10.16pp，因此仍因灾难失败。κ1的possible E3是2/7，不能只列v9一个受损目标而把总灾难率写1/7。

P_rule必要controller合并收益较大，却有5/7可能灾难，整体E1相对R0的同时CI低于门；未达到安全正迁移。不能用base_u/v10收益掩盖其余目标伤害。

| 覆盖 | 表示 | 规则 | 目标 | vs prior 保守收益 | vs R0 保守收益 | η10占比 | η13占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REAL6 | R3-PCA2 | pessimistic1 | base_u | -1.17pp | -1.82pp | 98.44% | 0.00% |
| REAL6 | R3-PCA2 | conservative | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6 | R3-PCA2 | F4 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6 | P_rule | P_rule | base_u | +30.08pp | +29.43pp | 0.00% | 100.00% |
| REAL6 | R5-PCA2 | pessimistic1 | base_u | -1.17pp | -1.82pp | 98.44% | 0.00% |
| REAL6 | R5-PCA2 | conservative | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6 | R5-PCA2 | F4 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | pessimistic1 | base_u | -1.17pp | -1.82pp | 98.44% | 0.00% |
| REAL6 | R5-Dprog | conservative | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | F4 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | R3-PCA2 | pessimistic1 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | R3-PCA2 | conservative | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | R3-PCA2 | F4 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | P_rule | P_rule | base_u | +30.08pp | +29.43pp | 0.00% | 100.00% |
| REAL6+BLEND | R5-PCA2 | pessimistic1 | base_u | -1.17pp | -1.82pp | 98.44% | 0.00% |
| REAL6+BLEND | R5-PCA2 | conservative | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-PCA2 | F4 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-Dprog | pessimistic1 | base_u | -1.17pp | -1.82pp | 98.44% | 0.00% |
| REAL6+BLEND | R5-Dprog | conservative | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-Dprog | F4 | base_u | +0.00pp | -0.65pp | 100.00% | 0.00% |
| REAL6 | R3-PCA2 | pessimistic1 | v10_local | +21.48pp | +19.53pp | 9.38% | 90.62% |
| REAL6 | R3-PCA2 | conservative | v10_local | +17.58pp | +15.62pp | 29.69% | 70.31% |
| REAL6 | R3-PCA2 | F4 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | P_rule | P_rule | v10_local | +26.17pp | +24.22pp | 0.00% | 100.00% |
| REAL6 | R5-PCA2 | pessimistic1 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | R5-PCA2 | conservative | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | R5-PCA2 | F4 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | pessimistic1 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | conservative | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | F4 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6+BLEND | R3-PCA2 | pessimistic1 | v10_local | +0.00pp | -1.95pp | 98.44% | 1.56% |
| REAL6+BLEND | R3-PCA2 | conservative | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6+BLEND | R3-PCA2 | F4 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6+BLEND | P_rule | P_rule | v10_local | +26.17pp | +24.22pp | 0.00% | 100.00% |
| REAL6+BLEND | R5-PCA2 | pessimistic1 | v10_local | +0.00pp | -1.95pp | 98.44% | 1.56% |
| REAL6+BLEND | R5-PCA2 | conservative | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-PCA2 | F4 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-Dprog | pessimistic1 | v10_local | +0.00pp | -1.95pp | 98.44% | 1.56% |
| REAL6+BLEND | R5-Dprog | conservative | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-Dprog | F4 | v10_local | +0.00pp | -1.95pp | 100.00% | 0.00% |
| REAL6 | R3-PCA2 | pessimistic1 | v9_local | -13.28pp | -11.59pp | 4.69% | 87.50% |
| REAL6 | R3-PCA2 | conservative | v9_local | -10.16pp | -8.46pp | 45.31% | 53.12% |
| REAL6 | R3-PCA2 | F4 | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |
| REAL6 | P_rule | P_rule | v9_local | -16.02pp | -14.32pp | 0.00% | 100.00% |
| REAL6 | R5-PCA2 | pessimistic1 | v9_local | -1.56pp | +0.13pp | 68.75% | 26.56% |
| REAL6 | R5-PCA2 | conservative | v9_local | +0.00pp | +1.69pp | 98.44% | 1.56% |
| REAL6 | R5-PCA2 | F4 | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | pessimistic1 | v9_local | -1.56pp | +0.13pp | 79.69% | 20.31% |
| REAL6 | R5-Dprog | conservative | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |
| REAL6 | R5-Dprog | F4 | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |
| REAL6+BLEND | R3-PCA2 | pessimistic1 | v9_local | -2.73pp | -1.04pp | 68.75% | 29.69% |
| REAL6+BLEND | R3-PCA2 | conservative | v9_local | +0.00pp | +1.69pp | 98.44% | 1.56% |
| REAL6+BLEND | R3-PCA2 | F4 | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |
| REAL6+BLEND | P_rule | P_rule | v9_local | -16.02pp | -14.32pp | 0.00% | 100.00% |
| REAL6+BLEND | R5-PCA2 | pessimistic1 | v9_local | -2.73pp | -1.04pp | 68.75% | 29.69% |
| REAL6+BLEND | R5-PCA2 | conservative | v9_local | +0.00pp | +1.69pp | 98.44% | 1.56% |
| REAL6+BLEND | R5-PCA2 | F4 | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |
| REAL6+BLEND | R5-Dprog | pessimistic1 | v9_local | -2.73pp | -1.04pp | 68.75% | 29.69% |
| REAL6+BLEND | R5-Dprog | conservative | v9_local | +0.00pp | +1.69pp | 96.88% | 3.12% |
| REAL6+BLEND | R5-Dprog | F4 | v9_local | +0.00pp | +1.69pp | 100.00% | 0.00% |


所有7目标、全部16个η直方图及逐目标描述CI在real_results.json、tables/real_per_target.csv；下图覆盖全部目标。

![Factorial endpoints](figures/real_factorial_E1_E2_E3.png)

![All target choices](figures/real_eta_histograms_all_targets.png)

### Q预测（独立于选择与regime识别）

| 覆盖 | 模型 | valid-head MSE | valid-seed NLL |
| --- | --- | --- | --- |
| REAL6 | R3-PCA2 | 0.10241 | 0.43171 |
| REAL6 | B0 | 0.10446 | 0.42599 |
| REAL6 | R5-PCA2 | 0.08388 | 0.37044 |
| REAL6 | R5-Dprog | 0.11824 | 0.46582 |
| REAL6+BLEND | R3-PCA2 | 0.09218 | 0.39926 |
| REAL6+BLEND | B0 | 0.10446 | 0.42599 |
| REAL6+BLEND | R5-PCA2 | 0.09250 | 0.38772 |
| REAL6+BLEND | R5-Dprog | 0.10440 | 0.42608 |


MSE针对终点有效seed比例的Q估计，保留有限Q4标签噪声；NLL仅valid seed计入。unknown未当作失败。逐目标chosen-Q高估区间见Q_prediction字段。F4的regime概率和P_rule不是成功概率Q，不伪造Q误差。这里的Q误差、上节η选择收益、阶段1 controller级regime准确率是三个不同问题，不能相互替代。

## 阶段3：融合族LOPO次检验

4fold，SPARSE14真实Flow，DENSE加9个非留出pair融合。终点仅32state × seed2–3；必要3个controller沿用旧discovery gate，不用终点重选。R3旧模型全部复用、纳入本轮同时推断族。

| 覆盖 | 基线 | 成功率下界 | 成功率上界 |
| --- | --- | --- | --- |
| DENSE | B0 | 79.86% | 80.56% |
| DENSE | source_prior | 78.91% | 79.69% |
| SPARSE | B0 | 78.91% | 79.69% |
| SPARSE | source_prior | 78.91% | 79.69% |


| 覆盖 | 表示 | 规则 | 成功率下界 | E1 vs R0 同时CI | E2 vs prior 同时CI | E2 vs R0 同时CI | E3 possible/certain | SPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | R3-PCA2 | pessimistic1 | 81.12% | [-0.19, +2.69]pp | +1.04pp [-1.04, +3.12]pp | +0.87pp [-0.87, +2.60]pp | 0/12 ; 0/12 | False |
| DENSE | R3-PCA2 | conservative | 80.34% | [-0.81, +1.68]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.52, +0.17]pp | 0/12 ; 0/12 | False |
| DENSE | R3-PCA2 | F4 | 80.08% | [-1.56, +2.01]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.52, +0.17]pp | 0/12 ; 0/12 | False |
| DENSE | P_rule | P_rule | 84.38% | [-2.19, +10.82]pp | +13.02pp [+5.62, +20.89]pp | +12.85pp [+5.50, +20.66]pp | 1/12 ; 0/12 | False |
| DENSE | R5-PCA2 | pessimistic1 | 79.95% | [-0.72, +0.91]pp | +1.04pp [-1.19, +3.28]pp | +0.87pp [-1.00, +2.73]pp | 0/12 ; 0/12 | False |
| DENSE | R5-PCA2 | conservative | 78.91% | [-2.82, +0.90]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.55, +0.20]pp | 0/12 ; 0/12 | False |
| DENSE | R5-PCA2 | F4 | 78.91% | [-2.82, +0.90]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.55, +0.20]pp | 0/12 ; 0/12 | False |
| DENSE | R5-Dprog | pessimistic1 | 78.39% | [-3.85, +0.65]pp | -3.65pp [-7.47, +0.18]pp | -3.82pp [-7.64, +0.00]pp | 1/12 ; 1/12 | False |
| DENSE | R5-Dprog | conservative | 79.04% | [-2.55, +0.93]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.55, +0.20]pp | 0/12 ; 0/12 | False |
| DENSE | R5-Dprog | F4 | 78.91% | [-2.82, +0.90]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.55, +0.20]pp | 0/12 ; 0/12 | False |
| SPARSE | R3-PCA2 | pessimistic1 | 82.68% | [-0.96, +8.33]pp | +7.81pp [+0.52, +15.42]pp | +7.81pp [+0.52, +15.42]pp | 0/12 ; 0/12 | True |
| SPARSE | R3-PCA2 | conservative | 81.77% | [-0.66, +6.15]pp | +4.69pp [+0.12, +9.26]pp | +4.69pp [+0.12, +9.26]pp | 0/12 ; 0/12 | True |
| SPARSE | R3-PCA2 | F4 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | P_rule | P_rule | 84.38% | [-2.00, +12.50]pp | +13.02pp [+5.68, +20.83]pp | +13.02pp [+5.68, +20.83]pp | 1/12 ; 0/12 | False |
| SPARSE | R5-PCA2 | pessimistic1 | 79.43% | [-1.80, +2.93]pp | +2.08pp [-4.69, +9.27]pp | +2.08pp [-4.69, +9.27]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-PCA2 | conservative | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-PCA2 | F4 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-Dprog | pessimistic1 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-Dprog | conservative | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-Dprog | F4 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |


融合SPARSE的R3-PCA2悲观/保守仍达到SPT；R5两种表示在两个coverage均未达到SPT。这保留了round6“融合族有静态表示正迁移，真实Flow没有”的差异；不能据此证明R5优于R3。融合controller由已知锚点插值、相关性更强，且TRAIN regime可能随λ变为η4，不是纯二类连续η10↔η13过渡；它不等价于新真实checkpoint。

![Blend factorial](figures/blend_factorial_E1_E2_E3.png)

### 插补敏感性（不进入主判定）

阶段1进展差主检验的插补结果如下（同一预定5000次成簇bootstrap）：

| 人群 | 插补ρ | 插补95%CI |
| --- | --- | --- |
| real | -0.323 | [-0.868, 0.341] |
| blend | -0.007 | [-0.664, 0.771] |
| combined | 0.253 | [-0.257, 0.628] |


| 表示 | 分类器 | 插补LOO准确率 |
| --- | --- | --- |
| R5 | 1NN | 57.14% |
| R5 | logistic_C1 | 57.14% |
| R5-PCA2 | 1NN | 64.29% |
| R5-PCA2 | logistic_C1 | 64.29% |
| R5-Dprog | 1NN | 64.29% |
| R5-Dprog | logistic_C1 | 64.29% |
| R3-PCA2+R5-PCA2 | 1NN | 57.14% |
| R3-PCA2+R5-PCA2 | logistic_C1 | 50.00% |


| 覆盖 | 表示 | 规则 | 成功率下界 | E1 vs R0 同时CI | E2 vs prior 同时CI | E2 vs R0 同时CI | E3 possible/certain | SPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | R5-PCA2 | pessimistic1 | 79.95% | [-0.68, +0.87]pp | +1.04pp [-1.40, +3.49]pp | +0.87pp [-1.17, +2.91]pp | 0/12 ; 0/12 | False |
| DENSE | R5-PCA2 | conservative | 78.91% | [-2.72, +0.80]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.58, +0.23]pp | 0/12 ; 0/12 | False |
| DENSE | R5-PCA2 | F4 | 78.91% | [-2.72, +0.80]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.58, +0.23]pp | 0/12 ; 0/12 | False |
| DENSE | P_rule | P_rule | 84.38% | [-1.86, +10.48]pp | +13.02pp [+5.62, +20.89]pp | +12.85pp [+5.50, +20.66]pp | 1/12 ; 0/12 | False |
| DENSE | R5-Dprog | pessimistic1 | 76.56% | [-5.69, -1.10]pp | -12.50pp [-19.18, -5.82]pp | -12.67pp [-19.46, -5.89]pp | 1/12 ; 1/12 | False |
| DENSE | R5-Dprog | conservative | 76.82% | [-5.46, -0.62]pp | -8.85pp [-14.79, -2.91]pp | -9.03pp [-15.10, -2.95]pp | 1/12 ; 1/12 | False |
| DENSE | R5-Dprog | F4 | 78.91% | [-2.72, +0.80]pp | +0.00pp [+0.00, +0.00]pp | -0.17pp [-0.58, +0.23]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-PCA2 | pessimistic1 | 79.56% | [-1.51, +2.91]pp | +2.60pp [-4.17, +9.78]pp | +2.60pp [-4.17, +9.78]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-PCA2 | conservative | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-PCA2 | F4 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | P_rule | P_rule | 84.38% | [-1.63, +12.12]pp | +13.02pp [+5.68, +20.83]pp | +13.02pp [+5.68, +20.83]pp | 1/12 ; 0/12 | False |
| SPARSE | R5-Dprog | pessimistic1 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-Dprog | conservative | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |
| SPARSE | R5-Dprog | F4 | 78.91% | [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | +0.00pp [+0.00, +0.00]pp | 0/12 ; 0/12 | False |


插补后R5仍无SPT；DENSE Δprog的保守/悲观收益还可能更差，说明缺失格处理不能笼统称完全无影响。该方案同时恢复所有完整controller的第16个state差值，绝非只改变base_u_v3一行。阶段1敏感性I1–I4完整保存在STAGE1.json的imputed分支，CSV与混淆图一并交付；不按敏感性终点改主方案。

## 假设判定

| 假设 | 判定 |
| --- | --- |
| H_R5_REAL6_sufficient | REJECTED |
| H_anchor_neighborhood_coverage_sufficient_with_R3 | REJECTED |
| H_R5_plus_anchor_coverage_sufficient | REJECTED |
| H_positive_monotone_delta_progress_information_real | INCONCLUSIVE |
| H_P_rule_safe_positive_transfer | REJECTED |
| H_static_recipe | SUPPORTED |
| H_behavior_regime | REJECTED |
| H_independent_eta13_real_controller_coverage | INCONCLUSIVE |
| H_learning_pathway_is_the_remaining_bottleneck | INCONCLUSIVE |
| H_any_R5_statistic_or_nonlinear_model_has_no_information | INCONCLUSIVE |
| H_real_safe_positive_transfer_under_prescribed_family | REJECTED |
| H_real_complete_transfer | REJECTED |
| H_prospective_confirmation | INCONCLUSIVE |


REJECTED表示“预定方案在当前DEVELOPMENT面板不充分”；不等于证明该研究方向不可能。H_positive_monotone_delta_progress_information_real为INCONCLUSIVE而不是证明零相关，主映射中的“无信息”只指未满足预注册ρCI下界>0的信息门。学习通路仍可能有限，但本轮未识别为剩余唯一瓶颈。

## 成本对照与下一步最小设计

R5原始部署成本每controller **32×200=6,400模拟步**，无成功标签，但需要闭环模拟器；本轮全部复用，不新增。round5 k=1校准每controller **32个完整H700 episode**（8个已掩码η头×Q4），需要成功/unknown标签，最大预算32×700=22,400模拟步，真实episode可能早停；不能只按同为32次就称计算成本相等。本轮主缺失方案只用31格/15个成对state，但既有数据采集成本仍按原32次计。

以下直接引用round5 REPORT.md的“目标少样本校准参照”表。该表是 **LOPO的3个必要融合controller等权、32个test state、终点seed2–3** 的绝对成功率下界，不是7个真实Flow均值，也不是相对R0收益。fixed列为用预定前k TRAIN state标签选出的固定η；critic+offset列为同一原critic加目标标签logit校准。目标校准标签不含终点seed。

| 覆盖 | 同表源prior基线 | 同表R0基线 | 同表最近源基线 | discovery固定目标η参照 |
| --- | --- | --- | --- | --- |
| SPARSE | 78.65% | 78.65% | 85.94% | 92.71% |
| DENSE | 78.65% | 78.82% | 76.56% | 92.71% |



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

下一步先做 **0 rollout** 的源数据设计：从已保存831条完整raw轨迹定义后段goal进展、投影持续性、相对运动或绕障进度等候选，先预注册并仅在源TRAIN内选择；不把这里的200步低速统计升级为正式deadlock标签。Ring旧评价器没有runtime deadlock事件，timeout须单列。若完整200步轨迹仍不足，再单独授权并预注册更长horizon的probe。

最小独立前瞻计划（当前尚无通过SPT的R5候选，须先在源内冻结候选后才执行）：4个新controller，优先不同训练seed的真实Ring Flow，覆盖至少2个候选η13型和2个η10型；新锚点组合融合可作工程先导，但不能替代独立真实controller结论。为每个新controller采集32次短probe，共 **128次200步无标签probe=25,600步**。用独立24state×η{10,13}×Q2作必要性发现，共 **384个完整episode**；终点另用全新64state×K16×Q4，共 **16,384个完整episode**。总计 **16,768个有标签完整H700 episode +128个短probe=16,896次请求**，完整episode上限11,737,600步。发现state、终点state、probe state三者与所有历史state去重；同一冻结规则评价全部controller，不按终点筛选。

若测试更长probe（例如现有26controller×16state×2η×H700），另需 **832次无标签H700请求，582,400步**；这是未来新协议成本，不在本轮运行，不把旧200步标签/partial假作完整700步轨迹。先复用现有raw的0-rollout方案可避免不必要采集。

独立覆盖假设的后续成本也保留：每新增真实controller，128 TRAIN state×8掩码头×Q4=4,096完整episode，P1兼容验证24state×K16×Q4=1,536，另32短probe；若新增2个独立η13真实controller，源标签预算为 **11,264完整episode+64短probe**，训练Flow本身的计算成本另计。这是独立覆盖的试验计划，当前结果不支持把它写成已解决方案。若之后建立了probe信息但选择仍失败，可另预注册以probe预测regime先验、再接源固定η曲线的结构化模型；对当前已有标签/轨迹的开发成本为0新rollout。

边界：所有数据早已打开，均DEVELOPMENT；固定Ring、FF、H700、K16和几何，真实必要controller仅base_u与v10_local两个。source-only bootstrap没有把controller总体不确定性估进去；14Flow/4pair信息检验样本少。融合增广还混杂额外标签量、源采样分布与静态均值state组成。不能推广到四场景、N=100、任意真实Flow或连续η basin；安全正迁移不等价于严格死锁解决。

## 审计、文件与复算

FINAL_AUDIT状态为PASS_WITH_DOCUMENTED_RESOURCE_UPDATE：除用户任务文档资源段的外部更新外，其余3067项历史资产hash不变。原失败审计保留FINAL_AUDIT_INITIAL.json，PROMPT_EXTERNAL_CHANGE.diff记录唯一差异。原预注册hash、全部模型冻结hash均未变化；155条R3复用结果/直方图与round6逐值一致；unknown相同η精确抵消、不同η保守区间通过独立检查。没有连接共享DB，因此只声明本任务DB写入0，不宣称共享DB全局字节不变。所有新增文件只在本目录，没有git commit。

核心文件：PREREGISTRATION.md与receipt；STAGE0.json / STAGE0_REPRODUCTION.json / R5_AUDIT.json；STAGE1.json / TRAIN_TRUTH.json；TASKS.json / FIT_TASKS.json / COVERAGE.json；fits/、shared_sparse_fits/、classifiers/及CHECKPOINT_INDEX.json；MODEL_FREEZE.json；real_results.json、blend_results.json、blend_imputed_results.json及state_vectors；results.json；figures/的PNG/PDF、tables/的CSV；REPORT.md、HANDOFF.md。

复算入口（不重训、不读写DB、不生成rollout）：

```bash
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/controller_behavior_repr_v1/evaluate.py real
$PY -B diagnostics/controller_behavior_repr_v1/evaluate.py blend
$PY -B diagnostics/controller_behavior_repr_v1/evaluate.py blend imputed
$PY -B diagnostics/controller_behavior_repr_v1/verify.py
$PY -B diagnostics/controller_behavior_repr_v1/figures.py info
$PY -B diagnostics/controller_behavior_repr_v1/figures.py real
$PY -B diagnostics/controller_behavior_repr_v1/figures.py blend
$PY -B diagnostics/controller_behavior_repr_v1/figures.py blend imputed
$PY -B diagnostics/controller_behavior_repr_v1/report.py
```

只有确实丢失拟合产物时才按FIT_TASKS.json补对应critic，不盲目重训；既有rollout/probe永远不重跑。不得调用历史probe.py merge/worker或rollout merger。资源上限与其他任务共享，重新提交前须检查占用。

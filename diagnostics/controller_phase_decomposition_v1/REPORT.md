H_compress **SUPPORTED（DEVELOPMENT）**：P1、P2 均满足预注册阈值。分解形式下，现有 R5 描述对前期效应 E 展现较强的留一预测能力（Spearman 0.842，符号准确率13/14），是可学习性的局部正面证据；后期效应 L 的预测能力尚未确立，不能据此宣布完整 controller–Q 关系已可学习。两个描述的1-NN均未降低动作选择遗憾。

所有解释与结论均为 **DEVELOPMENT**。本次只分析已有结果，未新增 rollout、未写共享 DB，未改 round10 或历史资产。用户明确允许以数值完成启动本次纯分析；round10 的 NUMERICAL_COMPLETE_VIDEO_INCOMPLETE 状态保留。本分析完成不代表 round10 视频验收完成。

## 冻结定义、证据与复用

设计者 addendum 的 SHA256 为 `2ffa9e43d2dd94a1726fa5f5e0bb0b52d59b93af8bb88cc2214926770baacd9e`，receipt UTC 为2026-10-13T09:18:52+00:00，早于 round10 结果开放2026-10-13T09:59:37.263707+00:00。执行代理此前参与 round10，已接触过其结果；自己的预注册如实披露这点，不能将本执行登记称为代理盲态预注册。定义与预测来自独立设计者的冻结文件，本次未按结果改变。

复用 round10 LABELS、18688条原始记录、64新 state、seed0–3、14真实 Flow、73个可用controller–arm组合，unknown131条全部保留。独立逐条核对 success、unknown 和原始记录哈希，与冻结 endpoint 收据完全一致。Freg仅三controller可用，其他位置不进入oracle。复用 round7 FEATURES：R3用共同96-state均值（沿用stage1），R5统一删105–109列保留155维；没有新短探测、Flow训练或checkpoint推理。新增证据仅为这些冻结结果的分析和源controller上的ridge拟合。

E=S500−F10，L=S500−F13，N=F13−F10。各臂点成功率按unknown=0取保守下界，并同时报告unknown=1上界。两个下界成功率之差仍只是规定的点估计，并非该差值的数学下界；差值界另按A_lower−B_upper、A_upper−B_lower计算。E−L=N是恒等式，不是检验结果。

CI为64个state成簇bootstrap5000次，每簇保留4个seed，各controller/arm共用抽样，seed202610131001。下表是点估计的percentile95% CI；完整unknown界、CI包络、S300/S400敏感性在E_L_N.csv与results.json。所有数值单位为pp。

## 事实：P1、P2与逐controller效应

P1通过：N<0的12个controller中，9个E≥10pp，超过至少一半要求；全部14个中10个同时E、L≥10pp，超过至少5个要求。P2通过：E>0为11/14，L>0为14/14；N符号为负12、正2、零0。unknown的所有允许赋值不改变E、L正号数量（分别11与14）。

|controller|E [95% CI]|L [95% CI]|N [95% CI]|
|---|---:|---:|---:|
|base_u_plus_w_v4_wide|32.81 [23.05, 42.97]|72.66 [66.41, 78.52]|-39.84 [-50.01, -29.30]|
|base_u|61.33 [50.78, 71.09]|34.77 [26.17, 43.75]|26.56 [12.50, 39.45]|
|base_u_v4|48.83 [36.72, 60.94]|60.55 [53.91, 67.19]|-11.72 [-22.66, -0.78]|
|base_u_plus_w_v6|53.91 [44.53, 62.89]|84.38 [78.91, 89.45]|-30.47 [-41.80, -19.53]|
|base_u_plus_w_v3|12.11 [5.08, 19.92]|34.38 [27.34, 41.41]|-22.27 [-31.65, -12.89]|
|base_u_plus_w_v4_train_targeted|38.67 [26.95, 50.39]|71.48 [63.67, 78.91]|-32.81 [-44.14, -21.88]|
|base_u_v3|16.02 [7.81, 24.61]|28.12 [19.92, 36.72]|-12.11 [-24.61, -0.39]|
|base_u_v2|49.61 [39.84, 59.38]|63.28 [55.86, 70.70]|-13.67 [-26.56, -1.56]|
|base_u_v10_local|23.83 [15.23, 32.42]|1.17 [0.00, 2.73]|22.66 [14.06, 31.25]|
|base_u_plus_w_v5|62.50 [51.95, 72.27]|80.47 [73.83, 86.33]|-17.97 [-28.52, -7.81]|
|base_u_v7_local|0.00 [-3.12, 3.52]|58.59 [50.39, 66.02]|-58.59 [-66.80, -49.61]|
|base_u_plus_w_v4|51.95 [39.84, 63.67]|89.45 [85.55, 92.97]|-37.50 [-49.22, -25.78]|
|base_u_v8_local|-0.39 [-1.17, 0.00]|65.23 [58.20, 72.27]|-65.62 [-72.66, -58.59]|
|base_u_v9_local|-5.08 [-8.59, -1.95]|17.97 [11.33, 25.39]|-23.05 [-31.64, -15.62]|

敏感性分析（DEVELOPMENT；不替代S500主判定）：

|切换臂|平均E pp|平均L pp|E>0|L>0|N<0中E≥10pp|E、L均≥10pp|
|---|---:|---:|---:|---:|---:|---:|
|S300|28.74|51.34|12/14|14/14|9/12|10/14|
|S400|30.92|53.52|12/14|14/14|9/12|10/14|
|S500|31.86|54.46|11/14|14/14|9/12|10/14|

解释（DEVELOPMENT）：多数controller的两个阶段效应都为正且较大，而固定η净偏好的符号来自二者差值，符合预注册的H_compress映射。此结论支持该14-controller测试床上的机制解释；没有因果识别到‘所有过往零样本失败的唯一根因’，也不扩展到其他场景或任意controller。S500形成的有限差分亦不能严格拆成完全独立、无交互的局部时间段因果效应。

## 事实：P3扩展动作空间

固定集每折源均值选择F10，平均遗憾32.90pp、最大62.50pp。扩展集每折选择S500，平均遗憾1.03pp、最大5.86pp；平均遗憾仅为固定集的3.14%，满足≤1/3条件，但最大遗憾超过5pp。因此 **P3未通过（DEVELOPMENT）**，不得因平均改善而改判。

unknown兼容的平均遗憾范围为固定[32.14,33.54]pp、扩展[0.89,1.17]pp；扩展最大遗憾范围[5.08,7.42]pp。这些是允许赋值的边界，不是采样CI。

最大的点遗憾来自base_u_v3：S500成功211/256，Freg成功226/256，差15/256=5.859375pp；两臂unknown分别5、4，使该controller遗憾界为3.91–7.42pp。base_u_v9_local的S500与F10无unknown，差13/256=5.078125pp，仍确定超过5pp；所以即便v3的unknown全部有利，P3的最大遗憾门也不能通过。其余非零遗憾为v7_local 2.734375pp、v10_local与v8_local各0.390625pp。

解释（DEVELOPMENT）：扩展动作空间大幅改善平均零样本选择，但预注册的统一最坏情况保证未成立。

## 事实：P4分解关系与描述选动作

源13controller拟合等权z-score与PCA2，目标仅投影；ridge α=1、不惩罚截距、无调参。每折只用源标签选动作；目标标签仅用于评估。符号按负/零/正精确相等计分。

|描述|效应|Spearman|符号准确率|源均值符号准确率|RMSE pp|源均值RMSE pp|
|---|---|---:|---:|---:|---:|---:|
|R3-PCA2|E|0.376|78.57%|78.57%|20.33|24.88|
|R3-PCA2|L|0.090|100.00%|100.00%|24.46|27.79|
|R3-PCA2|N|0.371|85.71%|85.71%|23.77|26.73|
|R5-PCA2|E|0.842|92.86%|78.57%|8.59|24.88|
|R5-PCA2|L|0.209|100.00%|100.00%|24.24|27.79|
|R5-PCA2|N|0.301|85.71%|85.71%|23.16|26.73|

R5的E预测明显强于其N预测：ρ分别0.842与0.301，E的符号13/14高于源均值11/14，RMSE由24.88pp降至8.59pp。这是与设计者预期相反的重要局部正面发现。L的ρ仅0.209（R3为0.090）；L全14个为正，100%符号准确率与无描述源均值相同，不能用它证明描述学到了L的大小。N的符号12/14亦未优于源均值。源均值留一Spearman为−1是移除当前目标后均值随目标真值机械反向变化的性质，不是独立预测模型效果；该参考主要用于RMSE与符号基线。

P4没有预注册方向性预测，也没有定义‘可预测’显著性阈值。因此报告这些观察，不补造正式统计门：E显示可学习信号，L尚缺充分证据，完整分解controller–Q可学习性尚未确立。样本仅14个controller，未进行新的独立controller验证；不能把L的弱相关解释成数学上不可学习。

两种描述的1-NN平均遗憾均2.65pp、最大24.22pp，劣于源平均选择的1.03pp与5.86pp；没有进一步降低遗憾。最近源的动作选择在若干fold上失败，完整邻居、选择与regret见results.json和FOLD_PREPROCESSING.json。

解释（DEVELOPMENT）：下一轮可将E的结构化预测作为候选，但不能声称完整critic或基于描述的动作选择已验证成功；L与选择最坏情况仍需解决。

## 事实：例外与发生步数

不满足E、L同时≥10pp的四个例外为v10_local、v7_local、v8_local、v9_local；前三者主要表现为F10/F13已经高成功或E很小，v9表现为晚切换后E为负。下表给出这些controller及P3关键例外v3各臂实际终止类型和发生步数。SUCCESS列用首个成功步，TIMEOUT列用实际终止步；这不是死锁检测时间。若有数值unknown则取记录中的error.step，不将其当失败或死锁。每条失败的state、seed、UID与精确步数见EXCEPTION_FAILURE_RECORDS.csv；逐arm完整步数数组保留在results.json中的例外摘要（v3详情见CSV）。

|controller|臂|实际类型|数量|步数 min / median / max|
|---|---|---|---:|---|
|base_u_v3|F10|success|170|289 / 417.5 / 698|
|base_u_v3|F10|timeout|78|700 / 700 / 700|
|base_u_v3|F10|numerical_failure|8|48 / 97 / 354|
|base_u_v3|F13|success|139|265 / 457 / 695|
|base_u_v3|F13|timeout|111|700 / 700 / 700|
|base_u_v3|F13|numerical_failure|6|86 / 108.5 / 597|
|base_u_v3|Freg|success|226|268 / 369.5 / 697|
|base_u_v3|Freg|timeout|26|700 / 700 / 700|
|base_u_v3|Freg|numerical_failure|4|71 / 170 / 293|
|base_u_v3|S300|success|213|265 / 366 / 693|
|base_u_v3|S300|numerical_failure|5|86 / 103 / 182|
|base_u_v3|S300|timeout|38|700 / 700 / 700|
|base_u_v3|S400|success|211|265 / 425 / 681|
|base_u_v3|S400|numerical_failure|6|86 / 108.5 / 499|
|base_u_v3|S400|timeout|39|700 / 700 / 700|
|base_u_v3|S500|success|211|265 / 513 / 696|
|base_u_v3|S500|numerical_failure|5|86 / 103 / 182|
|base_u_v3|S500|timeout|40|700 / 700 / 700|
|base_u_v10_local|F10|success|183|378 / 552 / 698|
|base_u_v10_local|F10|timeout|73|700 / 700 / 700|
|base_u_v10_local|F13|success|241|276 / 317 / 613|
|base_u_v10_local|F13|timeout|15|700 / 700 / 700|
|base_u_v10_local|S300|success|245|276 / 307 / 517|
|base_u_v10_local|S300|timeout|11|700 / 700 / 700|
|base_u_v10_local|S400|success|245|276 / 317 / 486|
|base_u_v10_local|S400|timeout|11|700 / 700 / 700|
|base_u_v10_local|S500|success|244|276 / 317 / 688|
|base_u_v10_local|S500|timeout|12|700 / 700 / 700|
|base_u_v7_local|F10|success|249|282 / 398 / 692|
|base_u_v7_local|F10|timeout|7|700 / 700 / 700|
|base_u_v7_local|F13|success|99|266 / 415 / 699|
|base_u_v7_local|F13|timeout|157|700 / 700 / 700|
|base_u_v7_local|S300|success|255|266 / 330 / 694|
|base_u_v7_local|S300|timeout|1|700 / 700 / 700|
|base_u_v7_local|S400|success|256|266 / 421 / 694|
|base_u_v7_local|S500|success|249|266 / 519 / 694|
|base_u_v7_local|S500|timeout|7|700 / 700 / 700|
|base_u_v8_local|F10|success|256|278 / 330 / 654|
|base_u_v8_local|F13|success|88|287 / 466.5 / 699|
|base_u_v8_local|F13|timeout|168|700 / 700 / 700|
|base_u_v8_local|Freg|success|256|290 / 367.5 / 684|
|base_u_v8_local|S300|success|256|287 / 317 / 510|
|base_u_v8_local|S400|success|256|287 / 414 / 507|
|base_u_v8_local|S500|success|255|287 / 514 / 608|
|base_u_v8_local|S500|timeout|1|700 / 700 / 700|
|base_u_v9_local|F10|success|247|299 / 387 / 634|
|base_u_v9_local|F10|timeout|9|700 / 700 / 700|
|base_u_v9_local|F13|success|188|266 / 334.5 / 687|
|base_u_v9_local|F13|timeout|68|700 / 700 / 700|
|base_u_v9_local|S300|success|242|266 / 312 / 674|
|base_u_v9_local|S300|timeout|14|700 / 700 / 700|
|base_u_v9_local|S400|success|241|266 / 367 / 693|
|base_u_v9_local|S400|timeout|15|700 / 700 / 700|
|base_u_v9_local|S500|success|234|266 / 364 / 669|
|base_u_v9_local|S500|timeout|22|700 / 700 / 700|

这些记录没有合格的DEADLOCK事件证据。TIMEOUT只能称超时，不能冒充deadlock；success或运动渲染也不新增任何安全证据。

## 产物与复现

代码：analyze.py、verify_and_report.py。数值：results.json、E_L_N_MAIN.csv、E_L_N.csv、ARM_RATES.csv、RIDGE_PREDICTIONS.csv、P3_*_folds.csv、FOLD_PREPROCESSING.json。图：E_L_scatter.png/pdf，点编号见SCATTER_CONTROLLER_INDEX.csv。审核：ANALYSIS_AUDIT.json、RAW_REPRODUCTION_AUDIT.json、HISTORICAL_ASSET_INDEX.json及预注册/授权收据。

复现：使用指定venv Python以-B运行analyze.py，再运行verify_and_report.py。脚本限制文件写入本目录，并将BLAS线程限制为1，无GPU、rollout或DB依赖。

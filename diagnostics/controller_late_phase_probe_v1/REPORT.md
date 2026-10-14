L可学假设 **REJECTED（DEVELOPMENT）**；controller信息带来可迁移选择收益的假设 **SUPPORTED（新controller检验，非独立确认）**。后期探测R6-Δ虽对L达到Spearman 0.789，但RMSE比源均值基线为0.772，未过≤0.7门；组合描述亦为0.709。融合族G平均遗憾0.13pp、最大1.56pp，源平均选择为0.20pp、1.56pp；G_E判定 **SUPPORTED**。

## 事实：预注册、冻结与标签访问

预注册文件PREREGISTRATION.md与UTC/SHA收据先于本轮探测及分析。AGENTS.md、shared_rollout_db/README.md已读；历史controller/state/checkpoint/probe代码先检查并复用。旧R5只有200步，不能提供400步前缀及两个后期200步分支，所以只补此缺口。执行代理曾参与历史轮次，预注册明确披露已有历史结果知识；本轮阶段A没有重新读任何成功标签。

R6冻结UTC：2026-10-13T10:44:20.629541+00:00；模型与全部融合动作冻结UTC：2026-10-13T10:44:21.003154+00:00；新的融合endpoint开放UTC：2026-10-13T11:07:56.143024+00:00。阶段B/C标签仅在R6冻结之后读取；融合任务标签只在14个真实controller训练出的模型及12融合选择冻结之后打开。阶段A的文件读取屏障阻止旧results/labels/raw结果访问，DB仅查询UNLABELED_SHORT_PROBE source元数据。阶段A不消费env.step返回的reward/done/info，不调用summary；几何到达机器人数变化是预注册的无标签统计量，不是整局success。

26controller×16probe state，η13前缀400步，从同一状态复制两个分支（η13与η10各200步），未来seed0，分支共用绝对step400–599 RNG。关闭成功/碰撞提前终止，物理、安全投影、controller和700步horizon未改。7个统计量定义及顺序见预注册和probe_protocol.json。final_stalled_robot_count按最后执行速度<0.01，只是运动学描述，不能作为deadlock判据。

416组、832分支，其中完整828、unknown 4，实际模拟331601步，未重跑unknown，未填零。wide与v3各有15个完整配对state，其余24controller为16个。所有分支起点与第400步前缀位置精确一致，复算7项统计量最大误差0。R6 raw224D保留NaN；Delta对完整配对state均值。PCA缺失格只用source-controller列均值填补，缺失来源与折内预处理保存在STAGE_B.json；不能把插补称为真实观测。

无标签probe仅登记source_file、UNLABELED_SHORT_PROBE与独立UID，rollout.success行数为0。完整评估经过标准DB preflight、worker局部cache_writer journals、单merger、postflight。32例门全部通过，与历史结果的终止类型、episode长度、first-action float64字节精确一致。S300/S5003072请求均genuinely missing才执行；固定F10/F133072记录直接复用。

## 事实：阶段B的L可学门

14真实Flow LOCO，ridge α=1，不惩罚截距、无调参；z-score/PCA只拟合源13controller。判定要求至少一种R6系列描述同时Lρ≥0.6且RMSE/源均值RMSE≤0.7。符号准确率将0作为第三符号；L全14真值均为正，100%符号准确率与源均值相同，不足以验证L的大小预测。

|描述|效应|Spearman|符号准确率|RMSE pp|基线RMSE pp|RMSE比|
|---|---|---:|---:|---:|---:|---:|
|R5-PCA2|E|0.842|92.86%|8.59|24.88|0.345|
|R5-PCA2|L|0.209|100.00%|24.24|27.79|0.872|
|R5-PCA2|N|0.301|85.71%|23.16|26.73|0.866|
|R6-Delta|E|0.446|64.29%|25.49|24.88|1.025|
|R6-Delta|L|0.789|100.00%|21.46|27.79|0.772|
|R6-Delta|N|-0.226|85.71%|42.33|26.73|1.583|
|R6-PCA2|E|0.411|78.57%|20.53|24.88|0.825|
|R6-PCA2|L|0.121|100.00%|26.86|27.79|0.967|
|R6-PCA2|N|0.262|85.71%|28.83|26.73|1.079|
|R5-PCA2+R6-Delta|E|0.868|85.71%|8.99|24.88|0.361|
|R5-PCA2+R6-Delta|L|0.745|100.00%|19.70|27.79|0.709|
|R5-PCA2+R6-Delta|N|0.534|85.71%|22.10|26.73|0.827|

解释（DEVELOPMENT）：R6-Δ与组合描述提高了L的排序相关，但误差改善不足以通过冻结门。组合描述0.708855接近0.7仍不得放宽阈值。这里的REJECTED指预注册可学习性标准未满足，不证明任何可能表示或学习器都学不到L。R5对E的ρ0.841758、RMSE8.594656pp精确复现上一轮，是标签与controller对齐的额外检查。

## 事实：冻结规则与真实LOCO

G使用R5-PCA2的Ê与固定主描述R6-Δ的L̂：Ê<5pp选F10；否则L̂<5pp选F13；否则S500。G_E只使用第一个条件。动作集F10/F13/S300/S500，源均值tie按菜单顺序。真实LOCO1NN用R5-PCA2⊕R6-Δ欧氏距离，选最近源在菜单中的最佳动作；源预处理、源标签选择动作，目标标签仅用于评估。所有方法oracle均含目标可用Freg及S400等全部6臂。

|真实14Flow方法|平均遗憾 pp|最大遗憾 pp|
|---|---:|---:|
|G|0.64|5.86|
|G_E|0.64|5.86|
|source_average|1.03|5.86|
|nearest_neighbor|2.62|24.22|

逐controller选择、Ê、L̂及邻居在SELECTOR_MODELS.json，成功率与regret及unknown界见REAL_ACTIONS_REGRETS.csv。真实组只报告DEVELOPMENT，不是融合族主门的替代。

## 事实：融合族主检验

12个融合controller，32个旧test state×seed0–3。新S300/S500完整rollout3072条，其中数值unknown 12；固定F10/F13复用3072条，其中unknown 16。unknown既不重跑，也不当观测失败；点成功率取unknown=0下界，上界允许unknown=1。目标oracle只取F10/F13/S300/S500四臂。

|方法|平均遗憾 pp|最大遗憾 pp|相对源平均成功率收益 pp|点门|unknown判定|
|---|---:|---:|---:|---|---|
|G|0.13|1.56|0.07|PASS|SUPPORTED|
|G_E|0.13|1.56|0.07|PASS|SUPPORTED|
|source_average|0.20|1.56|0.00|FAIL|REJECTED|

主门是G平均遗憾严格低于源均值且最大≤5pp。源平均的‘transfer_status’只作统一函数的比较参考，它与自身没有严格收益；它不是一个额外研究假设。数值未知范围及5000次state-cluster bootstrap CI均在results.json，真实seed202610131101、融合seed202610131102，保留4个未来seed成簇；CI不替换点门。相同动作比较严格相消，未知success不能制造相同动作间收益。

|融合controller|Ê pp|L̂ pp|G / G_E / 源均值|G遗憾 pp|源均值遗憾 pp|
|---|---:|---:|---|---:|---:|
|blend_pair0_lambda25|51.18|57.02|S500 / S500 / S500|0.00|0.00|
|blend_pair0_lambda50|44.79|61.28|S500 / S500 / S500|0.00|0.00|
|blend_pair0_lambda75|41.41|66.64|S500 / S500 / S500|0.00|0.00|
|blend_pair1_lambda25|38.34|40.72|S500 / S500 / S500|0.00|0.00|
|blend_pair1_lambda50|27.96|40.68|S500 / S500 / S500|0.00|0.00|
|blend_pair1_lambda75|10.48|34.38|S500 / S500 / S500|0.00|0.00|
|blend_pair2_lambda25|35.25|35.78|S500 / S500 / S500|1.56|1.56|
|blend_pair2_lambda50|36.43|62.26|S500 / S500 / S500|0.00|0.00|
|blend_pair2_lambda75|32.31|181.74|S500 / S500 / S500|0.00|0.00|
|blend_pair3_lambda25|8.64|19.60|S500 / S500 / S500|0.00|0.00|
|blend_pair3_lambda50|-1.37|25.76|F10 / F10 / S500|0.00|0.00|
|blend_pair3_lambda75|-4.36|21.69|F10 / F10 / S500|0.00|0.78|

融合族收益幅度很小：平均成功率仅比源平均高0.07pp，对应12×32×4=1536个配对样本中净增加1次成功，来自blend_pair3_lambda75（F10为128/128，S500为127/128）。其余11controller没有点收益。收益的state-cluster95%CI为[0.00,0.20]pp，含0；SUPPORTED是预注册点门与unknown保守性判定通过，不是显著正收益或广泛迁移能力的统计证明。

真实14Flow中最大遗憾仍为5.859375pp，v3的η2型损失尚未解决。融合族的锚点不包含v3，不能用融合族通过门来宣布v3问题消除。

模型未加未预注册的截断；blend_pair2_lambda75的L̂约181.74pp是模型在描述空间外推的输出，超出实际效应范围，不是真实成功率增益。G只是按固定5pp阈值使用该值。全部融合controller的G与G_E动作一致，因此本组无法归因出额外后期L̂的选择收益。

解释与冻结结论映射：Transferable component is early E; late component either covered by schedules or requires another representation.

融合族是新的controller上的检验，来自已打开过的真实锚点，旧test states与F10/F13也已存在；不能称为新真实controller的独立确认。只在这些controller/状态/seed/菜单范围内解释，Q4不是B15认证。TIMEOUT仅叫超时，不冒充deadlock。

## 审核与交付

阶段C是纯Q函数数据生成/评估，按AGENTS.md明确例外不运行视频入口。该决定在预注册、任何新结果前固定。所有新文件均在本目录；无修改/删除历史资产、无git commit、无controller/物理/安全投影/成功判定修改。DB写入限本轮新增身份与证据及SQLite标准WAL/SHM生命周期，worker的journals在本目录。

入口：probe.py create → probe.py worker → audit_probe.py → probe.py merge（冻结R6）→ models.py（冻结选择）→ design.py create → run.py gate → gate.py → merge.py → run.py main → finalize.py。Slurm launcher记录任务与17worker资源策略；finalize.py只在完整raw存在后合并、评价、报告与审核，失败不生成COMPLETE收据。

关键产物：R6_FEATURES.npz、R6_AUDIT.json、R6_FROZEN.receipt.json、PROBE_REPRODUCTION_AUDIT.json、STAGE_B.json、STAGE_B_PREDICTIONS.csv、SELECTOR_MODELS.json、SELECTOR_FROZEN.receipt.json、*_CACHE_PREFLIGHT.json、main_POSTFLIGHT.json、REAL_ACTIONS_REGRETS.csv、BLEND_ACTIONS_REGRETS.csv、BLEND_ARM_RATES.csv、BLEND_LABELS.npz、results.json、两组PNG/PDF图、FINAL_AUDIT.json、COMPLETE.receipt.json。

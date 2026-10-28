# 正式论文基线与内部消融

五篇指定论文对应六项正式baseline。以下复用上一轮同一15类有效场景、每类4个留出初态，共60个物理任务的开发集选参后冻结端口结果；不重新生成rollout。Wang两版本、Chen选中参数与历史profile相同，直接使用原记录。

|正式论文方法端口|成功|成功样本平均任务时间(s)|碰撞|超时|unknown|
|---|---:|---:|---:|---:|---:|
|Wang T-RO2017 Consistent Perturbation|4/60|11.1125|10|9|37|
|Wang T-RO2017完整Deadlock Resolution|4/60|11.1125|10|9|37|
|Chen RA-L2023 Dynamic Priority + Right-Hand Rule + Safe Corridor|9/60|9.8667|5|45|1|
|Zhang ICRA2025 Adaptive CLF–CBF Deadlock Avoidance|0/60|无成功样本|11|11|38|
|Zhang T-RO2025 Deadlock-Aware Control|12/60|13.0667|0|39|9|
|Lee IROS2025 Merry-Go-Round|12/60|15.3958|2|46|0|

原始统计：[上一轮PAPER_RESULTS.json](../scene_types_basis_dimension_v1/PAPER_RESULTS.json)；精确开发菜单和冻结选择：[PAPER_PARAMETERS_FROZEN.json](../scene_types_basis_dimension_v1/PAPER_PARAMETERS_FROZEN.json)；源码和作者仓库pin：[PAPER_PORT_INPUTS.json](../scene_types_basis_dimension_v1/PAPER_PORT_INPUTS.json)；异常根因：[PAPER_FAILURE_ANALYSIS.md](../scene_types_basis_dimension_v1/PAPER_FAILURE_ANALYSIS.md)。历史profile完整保留，不能只报调参较好的profile而丢弃旧证据。

这些是共同SI物理承载下的完整论文机制端口，各保留自己的controller/安全方法。C1两个冻结nominal使用外部可见图路线引导，论文端口不接收该引导；论文端口未完成同等全队列单机器人能力门，且涉及动力学映射和较多求解/检测器异常。因此本表反映本适配协议的实际完成情况，不足以排名原论文方法，也不能把unknown都称为死锁消解失败。timeout单列，不直接等于deadlock。

以下仅是内部比较，分别列两种nominal，分母都为相同60个物理任务：

|内部方法|centralized_sampling|gradient_mpc|
|---|---:|---:|
|原B3固定参数|16/60|27/60|
|G/R/T1/C2容纳族的固定参数（实际非零G/R/T1）|22/60|25/60|
|原B3 Oracle@16|28/60|34/60|
|G/R/T1/C2 Oracle@16|37/60|40/60|

Oracle逐案例利用最终结果选择η，是有限候选池存在性筛选；论文profile为固定参数，不能把它们直接当同等部署能力。每个方法的成功条件耗时对应不同成功子集；配对共同成功时间需另报。本轮Oracle@64最终分析会同时生成逐nominal、共同成功配对耗时和全队列截断时间，不将它们混为一列。

截至64点扩展之前，原四列失败43例中，其他旧线性池已解决11例；补查开发成功配置56次，再新增1例：[G,R,T3,P2]，疏散类型9、centralized_sampling、初态6，固定系数为G=.875、R=.375、T3=-.9778857994824648、P2=.22705783881247044，23.85s完成。该点属于另一四列组合，不能记入原四列的77/120。逐候选raw/trace与官方journal已合并并读回。

用户后续明确要求原16点失败案例扩到64点；该阶段按独立冻结协议执行，结果完成前不填表，不修改上述历史计数。

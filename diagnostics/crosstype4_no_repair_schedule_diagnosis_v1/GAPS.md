# 缺口与解释边界

全部为开发证据；没有未见controller类型的确认性结论。

按prompt跳过其他8个学习型训练类型的新状态评估；AXIS_TABLE对应类型保留N/A。

跨类型3能力门中未提供数值安全投影成功率的实例：base_u_plus_w_v4_wide_macflow, base_u_macflow, base_u_v4_macflow, base_u_plus_w_v6_macflow, base_u_plus_w_v3_macflow, base_u_plus_w_v4_train_targeted_macflow, base_u_v3_macflow, base_u_v2_macflow, base_u_v10_local_macflow, base_u_plus_w_v5_macflow, base_u_v7_local_macflow, base_u_plus_w_v4_macflow, base_u_v8_local_macflow, base_u_v9_local_macflow, DM0, DM1, DM2, DM3, DM4, DM5, centralized_sampling0, centralized_sampling1, centralized_sampling2, centralized_sampling3, centralized_sampling4, centralized_sampling5, cross_entropy0, cross_entropy1, cross_entropy2, cross_entropy3, cross_entropy4, cross_entropy5, prioritized_astar0, prioritized_astar1, prioritized_astar2, prioritized_astar3, prioritized_astar4, prioritized_astar5, distributed_cross_entropy0, distributed_cross_entropy1, distributed_cross_entropy2, distributed_cross_entropy3, distributed_cross_entropy4, distributed_cross_entropy5, centralized_mppi0, centralized_mppi1, centralized_mppi2, centralized_mppi3, centralized_mppi4, centralized_mppi5, ring_ippo_conservative0, ring_ippo_conservative1, ring_ippo_conservative2, ring_ippo_conservative3, ring_ippo_conservative4, ring_ippo_conservative5。不补跑能力门。

机制特征完全复用；历史5个和跨类型3新增5个不完整向量保留NaN，只在源类型内填补，共10个。

历史DB中无法恢复全部物理坐标的state无法绝对排除物理重合；UID/content全部去重，可恢复坐标也全部去重，见 NEW_STATE_DEDUP.json。

数值失败按unknown处理，不自动重跑。未知标签的配对上下界保留相同UID抵消；较好人定时间表在每次bootstrap重选。

A的对角元为选中单η，没有实际切换，代数分解复用同UID使切换效应为零；另列非对角条件结果与支持量。

工程修正与冻结前版本修订详见 ITERATION_LOG.md；真实重跑计数见 EXECUTION_COUNTS.json。

完整性审计未通过，必须先评估 INTEGRITY_COMPARISON.json 的差异，本轮结果不得当作审核通过交付。

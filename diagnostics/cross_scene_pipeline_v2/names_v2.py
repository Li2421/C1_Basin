METHOD_NAMES={
'v2':'学习选 η v2',
'v2_49':'v2·只在 49 个旧候选里选',
'v2_49_no_fallback':'v2·不退回',
'v2_49_clip':'v2·截断代替退回',
'v2_49_raw_description':'v2·原始场景描述',
'v2_49_no_prior':'v2·无先验',
'v2_49_per_cell':'v2·按状态和控制器选',
'v2_49_per_controller':'v2·按场景和控制器选',
'v2_49_lambda1':'v2·修正不收缩',
'generator_only':'只用生成器',
'global_fixed':'所有场景一组 η（训练场景里最好）',
'eta_A':'固定 η-A',
'knn':'按场景近邻选 η',
'eta13':'固定 η13',
'safety':'只加安全投影',
'scene_xstate':'每个场景一组 η（本场景其他状态选，参照）',
'v1_pipeline':'学习选 η v1（按状态和控制器选）'}
SCENE_NAMES={'lit_narrow_passage':'窄通道','lit_crowded_random':'拥挤随机','lit_circle_five':'圆形五机','lit_swap8_obstacles':'八机静态障碍',
'lit_swap8_free':'八机无障碍','lit_swap8_moving':'八机移动障碍','xsc_swap_n4':'对穿 4 机','xsc_swap_n6':'对穿 6 机','xsc_swap_n10':'对穿 10 机',
'xsc_swap_n12':'对穿 12 机','xsc_swap_n8_dense':'对穿 8 机加密','xsc_gap_n2':'缺口 2 机','xsc_gap_n4':'缺口 4 机','xsc_gap_n6':'缺口 6 机','xsc_wide_n4':'宽口 4 机','xsc_double_n4':'双瓶颈','xsc_offset_n4':'错位口','xsc_three_doors_n4':'三门 4 机'}
CONTROLLER_NAMES={'centralized_mppi':'集中式模型预测路径积分控制','centralized_sampling':'集中式随机采样模型预测控制','conflict_based_search':'冲突搜索多机规划','cross_entropy':'交叉熵方法模型预测控制','distributed_cross_entropy':'分布式交叉熵方法模型预测控制','distributed_shooting':'分布式模型预测控制','gradient_mpc':'梯度下降模型预测控制','prioritized_astar':'优先级时空A*规划'}

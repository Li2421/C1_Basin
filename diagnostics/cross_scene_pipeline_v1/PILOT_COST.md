# 计时试跑

11个场景各2状态×2生成器提案；使用该场景已有合格冲突搜索控制器，若无则取既定顺序首个实例。正式缺失请求预计 9.87 CPU小时，按当前23个worker约 0.43 小时。

未测类型的成本用已有相同控制器执行语义的旧计时作预测，不替代新η结果；规划尾部可能使均值估计失准。

| 场景/控制器类型 | 单局秒数（4局） | 平均秒 | 最长秒 |
|---|---|---:|---:|
| lit_narrow_passage|conflict_based_search | 0.040, 0.046, 0.036, 0.039 | 0.040 | 0.046 |
| lit_crowded_random|conflict_based_search | 0.637, 0.853, 0.300, 0.301 | 0.523 | 0.853 |
| lit_circle_five|conflict_based_search | 22.468, 13.174, 64.783, 47.817 | 37.061 | 64.783 |
| lit_swap8_obstacles|conflict_based_search | 3.020, 3.129, 2.505, 2.439 | 2.773 | 3.129 |
| xsc_swap_n4|conflict_based_search | 0.498, 0.519, 0.499, 0.561 | 0.519 | 0.561 |
| xsc_swap_n12|conflict_based_search | 5.022, 4.680, 4.382, 3.715 | 4.450 | 5.022 |
| xsc_swap_n8_dense|conflict_based_search | 1.487, 1.526, 1.391, 1.429 | 1.458 | 1.526 |
| xsc_gap_n4|conflict_based_search | 81.642, 91.080, 47.203, 51.263 | 67.797 | 91.080 |
| xsc_gap_n6|conflict_based_search | 158.621, 164.256, 75.577, 75.644 | 118.524 | 164.256 |
| xsc_double_n4|conflict_based_search | 103.667, 101.428, 40.933, 42.337 | 72.091 | 103.667 |
| xsc_offset_n4|conflict_based_search | 73.015, 100.941, 53.817, 56.078 | 70.963 | 100.941 |

| 正式场景/类型 | 预计缺失局数 | CPU小时 | 成本依据 |
|---|---:|---:|---|
| lit_narrow_passage|centralized_mppi | 46 | 0.001 | historical compatible controller timing |
| lit_narrow_passage|centralized_sampling | 46 | 0.001 | historical compatible controller timing |
| lit_narrow_passage|conflict_based_search | 44 | 0.000 | pilot |
| lit_narrow_passage|cross_entropy | 46 | 0.001 | historical compatible controller timing |
| lit_narrow_passage|distributed_cross_entropy | 46 | 0.001 | historical compatible controller timing |
| lit_narrow_passage|distributed_shooting | 46 | 0.001 | historical compatible controller timing |
| lit_narrow_passage|gradient_mpc | 46 | 0.006 | historical compatible controller timing |
| lit_narrow_passage|prioritized_astar | 46 | 0.000 | historical compatible controller timing |
| lit_crowded_random|centralized_mppi | 46 | 0.004 | historical compatible controller timing |
| lit_crowded_random|centralized_sampling | 51 | 0.004 | historical compatible controller timing |
| lit_crowded_random|conflict_based_search | 45 | 0.007 | pilot |
| lit_crowded_random|cross_entropy | 44 | 0.005 | historical compatible controller timing |
| lit_crowded_random|distributed_cross_entropy | 52 | 0.012 | historical compatible controller timing |
| lit_crowded_random|distributed_shooting | 51 | 0.006 | historical compatible controller timing |
| lit_crowded_random|gradient_mpc | 47 | 0.005 | historical compatible controller timing |
| lit_crowded_random|prioritized_astar | 48 | 0.005 | historical compatible controller timing |
| lit_circle_five|centralized_sampling | 44 | 0.091 | historical compatible controller timing |
| lit_circle_five|conflict_based_search | 42 | 0.432 | pilot |
| lit_circle_five|cross_entropy | 44 | 0.030 | historical compatible controller timing |
| lit_circle_five|distributed_cross_entropy | 44 | 0.040 | historical compatible controller timing |
| lit_circle_five|distributed_shooting | 44 | 0.032 | historical compatible controller timing |
| lit_circle_five|gradient_mpc | 44 | 0.032 | historical compatible controller timing |
| lit_circle_five|prioritized_astar | 44 | 0.066 | historical compatible controller timing |
| lit_swap8_obstacles|centralized_sampling | 35 | 0.038 | historical compatible controller timing |
| lit_swap8_obstacles|conflict_based_search | 34 | 0.026 | pilot |
| lit_swap8_obstacles|cross_entropy | 50 | 0.102 | historical compatible controller timing |
| lit_swap8_obstacles|distributed_cross_entropy | 34 | 0.067 | historical compatible controller timing |
| lit_swap8_obstacles|distributed_shooting | 35 | 0.044 | historical compatible controller timing |
| lit_swap8_obstacles|gradient_mpc | 51 | 0.048 | historical compatible controller timing |
| lit_swap8_obstacles|prioritized_astar | 47 | 0.045 | historical compatible controller timing |
| xsc_swap_n4|centralized_sampling | 12 | 0.002 | historical compatible controller timing |
| xsc_swap_n4|conflict_based_search | 10 | 0.001 | pilot |
| xsc_swap_n4|cross_entropy | 12 | 0.002 | historical compatible controller timing |
| xsc_swap_n4|distributed_cross_entropy | 13 | 0.003 | historical compatible controller timing |
| xsc_swap_n4|distributed_shooting | 12 | 0.002 | historical compatible controller timing |
| xsc_swap_n4|gradient_mpc | 17 | 0.003 | historical compatible controller timing |
| xsc_swap_n4|prioritized_astar | 13 | 0.002 | historical compatible controller timing |
| xsc_swap_n12|centralized_sampling | 12 | 0.013 | historical compatible controller timing |
| xsc_swap_n12|conflict_based_search | 10 | 0.012 | pilot |
| xsc_swap_n12|cross_entropy | 12 | 0.020 | historical compatible controller timing |
| xsc_swap_n12|distributed_cross_entropy | 12 | 0.033 | historical compatible controller timing |
| xsc_swap_n12|distributed_shooting | 12 | 0.017 | historical compatible controller timing |
| xsc_swap_n12|gradient_mpc | 12 | 0.015 | historical compatible controller timing |
| xsc_swap_n12|prioritized_astar | 12 | 0.014 | historical compatible controller timing |
| xsc_swap_n8_dense|centralized_sampling | 12 | 0.005 | historical compatible controller timing |
| xsc_swap_n8_dense|conflict_based_search | 10 | 0.004 | pilot |
| xsc_swap_n8_dense|cross_entropy | 12 | 0.007 | historical compatible controller timing |
| xsc_swap_n8_dense|distributed_cross_entropy | 12 | 0.010 | historical compatible controller timing |
| xsc_swap_n8_dense|distributed_shooting | 12 | 0.007 | historical compatible controller timing |
| xsc_swap_n8_dense|gradient_mpc | 12 | 0.005 | historical compatible controller timing |
| xsc_swap_n8_dense|prioritized_astar | 12 | 0.005 | historical compatible controller timing |
| xsc_gap_n4|conflict_based_search | 15 | 0.282 | pilot |
| xsc_gap_n4|prioritized_astar | 19 | 2.835 | historical compatible controller timing |
| xsc_gap_n6|conflict_based_search | 14 | 0.461 | pilot |
| xsc_gap_n6|prioritized_astar | 17 | 4.161 | historical compatible controller timing |
| xsc_double_n4|conflict_based_search | 16 | 0.320 | pilot |
| xsc_double_n4|prioritized_astar | 20 | 0.078 | historical compatible controller timing |
| xsc_offset_n4|conflict_based_search | 14 | 0.276 | pilot |
| xsc_offset_n4|prioritized_astar | 19 | 0.123 | historical compatible controller timing |

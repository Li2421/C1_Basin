# 计时试跑
新场景主标签估计 239.5 CPU小时；按当前可用17并发约 14.1 小时。仅基于2状态×2候选实测，不含能力门、旧场景补缺和外推，超时及规划尾部会增大误差。撤销及重提的未完成尝试开销另计；实际并发随他人CPU/内存需求调整。
| 场景/控制器类型 | 局数 | 平均秒 | 最长秒 | 主标签CPU小时估计 |
|---|---:|---:|---:|---:|
| xsc_swap_n4|centralized_sampling | 4 | 0.55 | 0.67 | 0.09 |
| xsc_swap_n4|conflict_based_search | 4 | 0.64 | 0.80 | 0.10 |
| xsc_swap_n4|cross_entropy | 4 | 0.57 | 0.77 | 0.09 |
| xsc_swap_n4|distributed_cross_entropy | 4 | 0.83 | 1.10 | 0.13 |
| xsc_swap_n4|distributed_shooting | 4 | 0.69 | 0.84 | 0.11 |
| xsc_swap_n4|gradient_mpc | 4 | 0.60 | 0.91 | 0.10 |
| xsc_swap_n4|prioritized_astar | 4 | 0.50 | 0.63 | 0.08 |
| xsc_swap_n6|centralized_sampling | 4 | 1.04 | 1.38 | 0.17 |
| xsc_swap_n6|conflict_based_search | 4 | 0.97 | 1.15 | 0.16 |
| xsc_swap_n6|cross_entropy | 4 | 1.16 | 1.56 | 0.19 |
| xsc_swap_n6|distributed_cross_entropy | 4 | 1.89 | 2.47 | 0.31 |
| xsc_swap_n6|distributed_shooting | 4 | 1.44 | 1.92 | 0.24 |
| xsc_swap_n6|gradient_mpc | 4 | 1.00 | 1.36 | 0.16 |
| xsc_swap_n6|prioritized_astar | 4 | 0.82 | 0.96 | 0.13 |
| xsc_swap_n10|centralized_sampling | 4 | 2.62 | 3.14 | 0.43 |
| xsc_swap_n10|conflict_based_search | 4 | 3.87 | 4.51 | 0.63 |
| xsc_swap_n10|cross_entropy | 4 | 3.01 | 3.65 | 0.49 |
| xsc_swap_n10|distributed_cross_entropy | 4 | 6.76 | 7.20 | 1.10 |
| xsc_swap_n10|distributed_shooting | 4 | 3.61 | 4.12 | 0.59 |
| xsc_swap_n10|gradient_mpc | 4 | 3.03 | 4.09 | 0.49 |
| xsc_swap_n10|prioritized_astar | 4 | 1.95 | 2.69 | 0.32 |
| xsc_swap_n12|centralized_sampling | 4 | 3.88 | 4.50 | 0.63 |
| xsc_swap_n12|conflict_based_search | 4 | 6.29 | 6.99 | 1.03 |
| xsc_swap_n12|cross_entropy | 4 | 6.02 | 6.93 | 0.98 |
| xsc_swap_n12|distributed_cross_entropy | 4 | 9.95 | 10.60 | 1.63 |
| xsc_swap_n12|distributed_shooting | 4 | 5.22 | 5.82 | 0.85 |
| xsc_swap_n12|gradient_mpc | 4 | 4.45 | 5.24 | 0.73 |
| xsc_swap_n12|prioritized_astar | 4 | 4.19 | 5.26 | 0.68 |
| xsc_swap_n8_dense|centralized_sampling | 4 | 1.59 | 1.86 | 0.26 |
| xsc_swap_n8_dense|conflict_based_search | 4 | 1.34 | 1.53 | 0.22 |
| xsc_swap_n8_dense|cross_entropy | 4 | 1.97 | 2.22 | 0.32 |
| xsc_swap_n8_dense|distributed_cross_entropy | 4 | 2.88 | 3.38 | 0.47 |
| xsc_swap_n8_dense|distributed_shooting | 4 | 2.13 | 2.35 | 0.35 |
| xsc_swap_n8_dense|gradient_mpc | 4 | 1.59 | 1.78 | 0.26 |
| xsc_swap_n8_dense|prioritized_astar | 4 | 1.50 | 1.76 | 0.24 |
| xsc_gap_n2|conflict_based_search | 4 | 38.39 | 112.90 | 4.18 |
| xsc_gap_n2|prioritized_astar | 4 | 5.93 | 7.33 | 0.65 |
| xsc_gap_n4|conflict_based_search | 4 | 95.35 | 158.04 | 10.38 |
| xsc_gap_n4|prioritized_astar | 4 | 537.20 | 1388.77 | 58.50 |
| xsc_gap_n6|conflict_based_search | 4 | 186.51 | 291.09 | 20.31 |
| xsc_gap_n6|prioritized_astar | 4 | 881.06 | 3340.74 | 95.94 |
| xsc_wide_n4|conflict_based_search | 4 | 37.95 | 77.38 | 4.13 |
| xsc_wide_n4|prioritized_astar | 4 | 10.25 | 13.14 | 1.12 |
| xsc_double_n4|conflict_based_search | 4 | 88.94 | 137.21 | 9.68 |
| xsc_double_n4|prioritized_astar | 4 | 14.09 | 14.48 | 1.53 |
| xsc_offset_n4|conflict_based_search | 4 | 104.21 | 121.97 | 11.35 |
| xsc_offset_n4|prioritized_astar | 4 | 23.25 | 35.75 | 2.53 |
| xsc_three_doors_n4|conflict_based_search | 4 | 26.53 | 54.42 | 2.89 |
| xsc_three_doors_n4|prioritized_astar | 4 | 14.28 | 22.04 | 1.55 |

## 主标签整体预算（提交时估计）

文献补缺8192条，利用同执行语义的历史耗时估计约0.9 CPU小时；新场景全表估计中扣除已完成并复用的pilot 2.39 CPU小时，尚需约237.1 CPU小时。主标签整体从提交起的预算约238.0 CPU小时，按参考17并发约14.0小时。

这是时长外推，不是保证完成时限，也不是硬上下界；文献新eta的规划耗时可能不同。未含外推、审计和之前撤销的开销，并发会随他人实际资源需求调整。没有为估时补跑或训练。详见TOTAL_MAIN_COST_ESTIMATE.json。

| 文献场景/类型 | 历史耗时样本 | 均值秒 | 补缺请求 | CPU小时估计 |
|---|---:|---:|---:|---:|
| lit_narrow_passage|centralized_mppi | 256 | 0.106 | 512 | 0.02 |
| lit_narrow_passage|centralized_sampling | 256 | 0.100 | 512 | 0.01 |
| lit_narrow_passage|conflict_based_search | 256 | 0.107 | 512 | 0.02 |
| lit_narrow_passage|cross_entropy | 256 | 0.146 | 512 | 0.02 |
| lit_narrow_passage|distributed_cross_entropy | 256 | 0.178 | 512 | 0.03 |
| lit_narrow_passage|distributed_shooting | 256 | 0.130 | 512 | 0.02 |
| lit_narrow_passage|gradient_mpc | 256 | 0.114 | 512 | 0.02 |
| lit_narrow_passage|prioritized_astar | 256 | 0.093 | 512 | 0.01 |
| lit_crowded_random|centralized_mppi | 256 | 0.502 | 512 | 0.07 |
| lit_crowded_random|centralized_sampling | 251 | 0.567 | 512 | 0.08 |
| lit_crowded_random|conflict_based_search | 256 | 0.605 | 512 | 0.09 |
| lit_crowded_random|cross_entropy | 256 | 0.888 | 512 | 0.13 |
| lit_crowded_random|distributed_cross_entropy | 256 | 0.928 | 512 | 0.13 |
| lit_crowded_random|distributed_shooting | 256 | 0.794 | 512 | 0.11 |
| lit_crowded_random|gradient_mpc | 254 | 0.456 | 512 | 0.06 |
| lit_crowded_random|prioritized_astar | 256 | 0.485 | 512 | 0.07 |

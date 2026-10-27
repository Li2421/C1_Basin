# 12 个测试场景的视频（2026-10-27）

每个场景一个子目录；每个 controller 两段视频，起始状态相同：
- `<controller>__safety_failure.mp4`：只加安全投影（η = 0），失败。
- `<controller>__correction_success.mp4`：加修正，成功。修正优先用流水线（generator 16 个提案 + critic）选的 η；流水线在任何一个安全投影失败的状态上都没成功时，用最好固定 η sE22；再不行用 η13。

所有轨迹都重新跑过，结果与原记录一致（170/170）。

## 清单
| 场景 | controller | 状态 | 只加安全投影的结果 | 修正用的 η |
|---|---|---:|---|---|
| xsc_hs_plus_w020_te | centralized_sampling | 8 | timeout | pipeline |
| xsc_hs_plus_w020_te | distributed_shooting | 8 | timeout | pipeline |
| xsc_hs_plus_w020_te | centralized_mppi | 9 | timeout | best_fixed_eta |
| xsc_hs_plus_w020_te | distributed_cross_entropy | 8 | timeout | pipeline |
| xsc_hs_plus_w020_te | conflict_based_search | 8 | timeout | pipeline |
| xsc_hs_plus_w020_te | prioritized_astar | 15 | timeout | pipeline |
| xsc_hs_plus_w030_te | centralized_sampling | 8 | timeout | pipeline |
| xsc_hs_plus_w030_te | distributed_shooting | 9 | timeout | pipeline |
| xsc_hs_plus_w030_te | centralized_mppi | 8 | timeout | best_fixed_eta |
| xsc_hs_plus_w030_te | distributed_cross_entropy | 12 | timeout | pipeline |
| xsc_hs_plus_w030_te | conflict_based_search | 8 | timeout | pipeline |
| xsc_hs_plus_w030_te | prioritized_astar | 12 | timeout | pipeline |
| xsc_hs_plus_w038_te | centralized_sampling | 9 | timeout | pipeline |
| xsc_hs_plus_w038_te | distributed_shooting | 9 | timeout | pipeline |
| xsc_hs_plus_w038_te | centralized_mppi | 8 | timeout | best_fixed_eta |
| xsc_hs_plus_w038_te | distributed_cross_entropy | 18 | timeout | pipeline |
| xsc_hs_plus_w038_te | conflict_based_search | 8 | timeout | pipeline |
| xsc_hs_plus_w038_te | prioritized_astar | 17 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | centralized_sampling | 11 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | gradient_mpc | 8 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | distributed_shooting | 11 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | centralized_mppi | 8 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | cross_entropy | 8 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | conflict_based_search | 8 | timeout | pipeline |
| xsc_hs_box_n09_r013_f25_te | prioritized_astar | 9 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | centralized_sampling | 9 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | gradient_mpc | 8 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | distributed_shooting | 12 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | centralized_mppi | 9 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | cross_entropy | 8 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | distributed_cross_entropy | 13 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | conflict_based_search | 11 | timeout | pipeline |
| xsc_hs_box_n10_r013_f26_te | prioritized_astar | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | centralized_sampling | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | gradient_mpc | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | distributed_shooting | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | centralized_mppi | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | cross_entropy | 9 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | distributed_cross_entropy | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | conflict_based_search | 8 | timeout | pipeline |
| xsc_hs_box_n11_r014_f27_te | prioritized_astar | 8 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | centralized_sampling | 8 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | gradient_mpc | 9 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | distributed_shooting | 8 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | centralized_mppi | 20 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | distributed_cross_entropy | 8 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | conflict_based_search | 8 | timeout | pipeline |
| hs_obst_n08_R142_r050_te | prioritized_astar | 21 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | centralized_sampling | 8 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | gradient_mpc | 11 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | distributed_shooting | 8 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | centralized_mppi | 9 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | distributed_cross_entropy | 8 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | conflict_based_search | 8 | timeout | pipeline |
| hs_obst_n08_R148_r052_te | prioritized_astar | 11 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | centralized_sampling | 9 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | gradient_mpc | 8 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | distributed_shooting | 9 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | centralized_mppi | 8 | timeout | eta13 |
| hs_obst_n08_R138_r048_te | cross_entropy | 12 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | distributed_cross_entropy | 10 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | conflict_based_search | 9 | timeout | pipeline |
| hs_obst_n08_R138_r048_te | prioritized_astar | 8 | timeout | pipeline |
| xsc_hs_swap_n06_te | centralized_sampling | 16 | timeout | pipeline |
| xsc_hs_swap_n06_te | gradient_mpc | 14 | timeout | pipeline |
| xsc_hs_swap_n06_te | distributed_shooting | 16 | timeout | pipeline |
| xsc_hs_swap_n06_te | centralized_mppi | 11 | timeout | eta13 |
| xsc_hs_swap_n06_te | distributed_cross_entropy | 9 | timeout | pipeline |
| xsc_hs_swap_n06_te | conflict_based_search | 10 | timeout | pipeline |
| xsc_hs_swap_n06_te | prioritized_astar | 21 | timeout | pipeline |
| xsc_hs_swap_n08_te | centralized_sampling | 8 | timeout | pipeline |
| xsc_hs_swap_n08_te | gradient_mpc | 8 | timeout | pipeline |
| xsc_hs_swap_n08_te | distributed_shooting | 8 | timeout | pipeline |
| xsc_hs_swap_n08_te | centralized_mppi | 17 | timeout | eta13 |
| xsc_hs_swap_n08_te | cross_entropy | 8 | timeout | pipeline |
| xsc_hs_swap_n08_te | distributed_cross_entropy | 8 | timeout | pipeline |
| xsc_hs_swap_n08_te | conflict_based_search | 9 | timeout | pipeline |
| xsc_hs_swap_n08_te | prioritized_astar | 18 | timeout | pipeline |
| xsc_hs_swap_n10_te | centralized_sampling | 8 | timeout | pipeline |
| xsc_hs_swap_n10_te | gradient_mpc | 13 | timeout | pipeline |
| xsc_hs_swap_n10_te | distributed_shooting | 8 | timeout | pipeline |
| xsc_hs_swap_n10_te | centralized_mppi | 17 | timeout | eta13 |
| xsc_hs_swap_n10_te | cross_entropy | 10 | timeout | pipeline |
| xsc_hs_swap_n10_te | distributed_cross_entropy | 10 | timeout | pipeline |
| xsc_hs_swap_n10_te | conflict_based_search | 15 | timeout | pipeline |

## 没有视频的组合
| 场景 | controller | 只加安全投影的失败局数 | 原因 |
|---|---|---:|---|
| xsc_hs_plus_w020_te | gradient_mpc | 0 | 只加安全投影从不失败 |
| xsc_hs_plus_w020_te | cross_entropy | 0 | 只加安全投影从不失败 |
| xsc_hs_plus_w030_te | gradient_mpc | 0 | 只加安全投影从不失败 |
| xsc_hs_plus_w030_te | cross_entropy | 0 | 只加安全投影从不失败 |
| xsc_hs_plus_w038_te | gradient_mpc | 0 | 只加安全投影从不失败 |
| xsc_hs_plus_w038_te | cross_entropy | 0 | 只加安全投影从不失败 |
| xsc_hs_box_n09_r013_f25_te | distributed_cross_entropy | 1 | 失败的那个状态上所有修正也失败 |
| hs_obst_n08_R142_r050_te | cross_entropy | 0 | 只加安全投影从不失败 |
| hs_obst_n08_R148_r052_te | cross_entropy | 0 | 只加安全投影从不失败 |
| xsc_hs_swap_n06_te | cross_entropy | 0 | 只加安全投影从不失败 |
| xsc_hs_swap_n10_te | prioritized_astar | 0 | 只加安全投影从不失败 |

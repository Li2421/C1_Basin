# 每个场景一对视频（12 对）

每对两段从同一状态开始：只加安全投影（η = 0）失败；加流水线选的 η（generator 16 个提案 + critic）成功。
每个场景选的是通过能力检查的 7 个 controller 中，流水线比只加安全投影提升最多的那个。

| 场景 | controller | 状态 | 只加安全投影结果 | 流水线 η | 该场景该 controller：流水线 / 只加安全投影 |
|---|---|---:|---|---|---|
| xsc_hs_plus_w020_te | conflict_based_search | 8 | timeout | (0.12, 0.34, 0.55) | 87.5 / 0.0 |
| xsc_hs_plus_w030_te | conflict_based_search | 8 | timeout | (0.11, 0.37, 0.57) | 100.0 / 18.8 |
| xsc_hs_plus_w038_te | conflict_based_search | 8 | timeout | (0.11, 0.31, 0.56) | 100.0 / 6.2 |
| xsc_hs_box_n09_r013_f25_te | conflict_based_search | 8 | timeout | (1.14, -0.10, 1.38) | 100.0 / 50.0 |
| xsc_hs_box_n10_r013_f26_te | centralized_sampling | 9 | timeout | (1.22, -0.06, 1.42) | 68.8 / 25.0 |
| xsc_hs_box_n11_r014_f27_te | conflict_based_search | 8 | timeout | (1.04, 0.98, 1.41) | 75.0 / 18.8 |
| hs_obst_n08_R142_r050_te | conflict_based_search | 8 | timeout | (0.53, -0.12, 1.42) | 93.8 / 18.8 |
| hs_obst_n08_R148_r052_te | conflict_based_search | 8 | timeout | (0.55, -0.15, 1.41) | 93.8 / 6.2 |
| hs_obst_n08_R138_r048_te | centralized_sampling | 9 | timeout | (0.52, -0.12, 1.41) | 87.5 / 25.0 |
| xsc_hs_swap_n06_te | distributed_cross_entropy | 9 | timeout | (0.22, 0.22, 1.05) | 62.5 / 31.2 |
| xsc_hs_swap_n08_te | distributed_cross_entropy | 8 | timeout | (0.41, 1.25, 1.03) | 81.2 / 25.0 |
| xsc_hs_swap_n10_te | distributed_cross_entropy | 10 | timeout | (0.45, 1.27, 1.14) | 50.0 / 12.5 |

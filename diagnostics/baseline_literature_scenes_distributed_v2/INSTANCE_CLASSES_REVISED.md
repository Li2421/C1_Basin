# 修订实例归类（原轨迹重算）

原COMPLETED_*与原controller级口径保持不变。这里按用户修正指令，用失败局中未到达机器人·局的互斥分类多数重新判断；只从去除分析中去掉controller级失败，不把安全层导致的全局失败排除出死锁证据。主值W=5s、τ=.05；上界≤5%为候选。旁证列同实例另一安全层成功率。

| 场景 | 层 | 合格数 | controller级失败 | state相关失败 | 安全层全局失败 | 墙前陷阱 | 有4a′且可作证据 |
|---|---|---:|---:|---:|---:|---:|---:|
| 窄通道场景 | central | 44 | 0 | 3 | 0 | 0 | 0 |
| 窄通道场景 | distributed | 44 | 2 | 0 | 42 | 0 | 42 |
| 二维拥挤随机场景 | central | 44 | 0 | 4 | 0 | 0 | 0 |
| 二维拥挤随机场景 | distributed | 44 | 0 | 41 | 0 | 0 | 40 |
| 圆形围场五机随机场景 | central | 45 | 3 | 19 | 0 | 0 | 12 |
| 圆形围场五机随机场景 | distributed | 45 | 3 | 42 | 0 | 0 | 42 |
| 门口场景 | central | 45 | 5 | 0 | 0 | 0 | 0 |
| 门口场景 | distributed | 45 | 5 | 0 | 0 | 0 | 1 |

## 低成功上界实例及旁证

| 场景 | 层 | controller | 修订类 | 本层成功上下界 | 另一层成功上下界 | 未到达机器人·局分类 |
|---|---|---|---|---|---|---|
| 圆形围场五机随机场景 | central | centralized_mppi2 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 152, 'robot_filter_equilibrium_deadlock': 5} |
| 圆形围场五机随机场景 | central | centralized_mppi3 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 149, 'robot_filter_equilibrium_deadlock': 10} |
| 圆形围场五机随机场景 | central | centralized_mppi5 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 148, 'robot_filter_equilibrium_deadlock': 8} |
| 圆形围场五机随机场景 | distributed | centralized_mppi2 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 146, 'robot_filter_equilibrium_deadlock': 7, 'blocked_by_arrived': 2} |
| 圆形围场五机随机场景 | distributed | centralized_mppi3 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 136, 'robot_filter_equilibrium_deadlock': 20} |
| 圆形围场五机随机场景 | distributed | centralized_mppi5 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 143, 'robot_filter_equilibrium_deadlock': 13} |
| 门口场景 | central | centralized_mppi2 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 103, 'slow_progress': 152} |
| 门口场景 | central | centralized_mppi3 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 156, 'slow_progress': 100} |
| 门口场景 | central | centralized_mppi5 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 220, 'slow_progress': 27} |
| 门口场景 | central | centralized_sampling4 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'zero_nominal': 114, 'blocked_by_arrived': 14} |
| 门口场景 | central | centralized_sampling5 | controller级失败 | [0.03125, 0.03125] | [0.015625, 0.015625] | {'zero_nominal': 129, 'blocked_by_arrived': 3} |
| 门口场景 | distributed | centralized_mppi2 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 102, 'slow_progress': 153} |
| 门口场景 | distributed | centralized_mppi3 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 156, 'slow_progress': 100} |
| 门口场景 | distributed | centralized_mppi5 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'other_stationary': 220, 'slow_progress': 27} |
| 门口场景 | distributed | centralized_sampling4 | controller级失败 | [0.0, 0.0] | [0.0, 0.0] | {'zero_nominal': 110, 'blocked_by_arrived': 18} |
| 门口场景 | distributed | centralized_sampling5 | controller级失败 | [0.015625, 0.015625] | [0.03125, 0.03125] | {'zero_nominal': 113, 'blocked_by_arrived': 13} |
| 窄通道场景 | distributed | DM0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | DM1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | DM2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | DM3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_mppi0 | controller级失败 | [0.0390625, 0.0390625] | [0.703125, 0.703125] | {'blocked_by_arrived': 108, 'other_stationary': 1, 'slow_progress': 13, 'oscillation': 1} |
| 窄通道场景 | distributed | centralized_mppi1 | 安全层导致的全局失败 | [0.0, 0.0] | [0.203125, 0.203125] | {'blocked_by_arrived': 63, 'robot_filter_equilibrium_deadlock': 65} |
| 窄通道场景 | distributed | centralized_mppi2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_mppi3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_mppi4 | controller级失败 | [0.0, 0.0] | [0.8828125, 0.8828125] | {'blocked_by_arrived': 124, 'robot_filter_equilibrium_deadlock': 2, 'slow_progress': 2} |
| 窄通道场景 | distributed | centralized_mppi5 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_sampling0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_sampling1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_sampling2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | centralized_sampling3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | conflict_based_search0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | conflict_based_search1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 124, 'blocked_by_arrived': 4} |
| 窄通道场景 | distributed | conflict_based_search2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | conflict_based_search3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | conflict_based_search4 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 122, 'oscillation': 4, 'blocked_by_arrived': 2} |
| 窄通道场景 | distributed | conflict_based_search5 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 124, 'oscillation': 2, 'blocked_by_arrived': 2} |
| 窄通道场景 | distributed | cross_entropy0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | cross_entropy1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | cross_entropy2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | cross_entropy3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | cross_entropy4 | 安全层导致的全局失败 | [0.015625, 0.015625] | [1.0, 1.0] | {'slow_progress': 2, 'blocked_by_arrived': 45, 'robot_filter_equilibrium_deadlock': 78, 'oscillation': 1} |
| 窄通道场景 | distributed | cross_entropy5 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | distributed_cross_entropy0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | distributed_cross_entropy1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | distributed_cross_entropy2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | distributed_cross_entropy3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | distributed_cross_entropy4 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | distributed_cross_entropy5 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | gradient_mpc0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | gradient_mpc1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | gradient_mpc2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | gradient_mpc3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | gradient_mpc4 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | gradient_mpc5 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | prioritized_astar0 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | prioritized_astar1 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | prioritized_astar2 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | prioritized_astar3 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | prioritized_astar4 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |
| 窄通道场景 | distributed | prioritized_astar5 | 安全层导致的全局失败 | [0.0, 0.0] | [1.0, 1.0] | {'robot_filter_equilibrium_deadlock': 128} |

## 九格敏感性改变的实例

- ('lit_narrow_passage', 'distributed', 'centralized_mppi1')：[(2.5, 0.02, 'controller级失败'), (2.5, 0.05, '安全层导致的全局失败'), (2.5, 0.1, '安全层导致的全局失败'), (5.0, 0.02, 'controller级失败'), (5.0, 0.05, '安全层导致的全局失败'), (5.0, 0.1, '安全层导致的全局失败'), (10.0, 0.02, 'controller级失败'), (10.0, 0.05, 'controller级失败'), (10.0, 0.1, '安全层导致的全局失败')]

完整384个场景/层/实例、九格计数与排除标记见JSON。只读取已有raw与DIAGNOSTICS，没有新rollout或训练；timeout本身不作为死锁。

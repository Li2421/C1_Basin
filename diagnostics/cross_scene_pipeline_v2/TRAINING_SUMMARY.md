# 训练与选择记录

764次未共享排程中的263次内部拟合与其他外层折具有完全相同的有序源集合、数据、先验、标准化、网络、采样和种子，因此共享检查点。实际291次内部拟合和210次外层三种子拟合，共501次。固定初始化key=0，种子17/23/41控制各场景独立采样流，均沿用指定实现。

| 外层折与留出场景 | 模型 | 所选种子 | 所选步 | 内部验证损失 |
|---|---|---:|---:|---:|
| loso：窄通道（loso__lit_narrow_passage） | 学习选 η v2的先验加有界修正critic | 23 | 400 | 0.454719782 |
| loso：窄通道（loso__lit_narrow_passage） | v2·原始场景描述的critic | 23 | 600 | 0.446120918 |
| loso：窄通道（loso__lit_narrow_passage） | v2·无先验的critic | 23 | 400 | 0.463081956 |
| loso：窄通道（loso__lit_narrow_passage） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 29.6097393 |
| loso：拥挤随机（loso__lit_crowded_random） | 学习选 η v2的先验加有界修正critic | 23 | 400 | 0.461745411 |
| loso：拥挤随机（loso__lit_crowded_random） | v2·原始场景描述的critic | 23 | 600 | 0.457416177 |
| loso：拥挤随机（loso__lit_crowded_random） | v2·无先验的critic | 17 | 400 | 0.4518736 |
| loso：拥挤随机（loso__lit_crowded_random） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 29.8556805 |
| loso：圆形五机（loso__lit_circle_five） | 学习选 η v2的先验加有界修正critic | 23 | 500 | 0.408778429 |
| loso：圆形五机（loso__lit_circle_five） | v2·原始场景描述的critic | 41 | 700 | 0.441607773 |
| loso：圆形五机（loso__lit_circle_five） | v2·无先验的critic | 17 | 400 | 0.416703463 |
| loso：圆形五机（loso__lit_circle_five） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 28.7877731 |
| loso：八机静态障碍（loso__lit_swap8_obstacles） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.429577231 |
| loso：八机静态障碍（loso__lit_swap8_obstacles） | v2·原始场景描述的critic | 23 | 600 | 0.434546769 |
| loso：八机静态障碍（loso__lit_swap8_obstacles） | v2·无先验的critic | 17 | 400 | 0.437498331 |
| loso：八机静态障碍（loso__lit_swap8_obstacles） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 29.5061283 |
| loso：对穿 4 机（loso__xsc_swap_n4） | 学习选 η v2的先验加有界修正critic | 23 | 600 | 0.432221293 |
| loso：对穿 4 机（loso__xsc_swap_n4） | v2·原始场景描述的critic | 17 | 800 | 0.454557568 |
| loso：对穿 4 机（loso__xsc_swap_n4） | v2·无先验的critic | 17 | 300 | 0.431001037 |
| loso：对穿 4 机（loso__xsc_swap_n4） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 28.632864 |
| loso：对穿 12 机（loso__xsc_swap_n12） | 学习选 η v2的先验加有界修正critic | 23 | 500 | 0.462888122 |
| loso：对穿 12 机（loso__xsc_swap_n12） | v2·原始场景描述的critic | 17 | 400 | 0.502760768 |
| loso：对穿 12 机（loso__xsc_swap_n12） | v2·无先验的critic | 17 | 400 | 0.458971858 |
| loso：对穿 12 机（loso__xsc_swap_n12） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 30.1542969 |
| loso：对穿 8 机加密（loso__xsc_swap_n8_dense） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.468215525 |
| loso：对穿 8 机加密（loso__xsc_swap_n8_dense） | v2·原始场景描述的critic | 23 | 600 | 0.481748879 |
| loso：对穿 8 机加密（loso__xsc_swap_n8_dense） | v2·无先验的critic | 17 | 400 | 0.475295722 |
| loso：对穿 8 机加密（loso__xsc_swap_n8_dense） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 29.9502335 |
| loso：缺口 4 机（loso__xsc_gap_n4） | 学习选 η v2的先验加有界修正critic | 41 | 400 | 0.417178631 |
| loso：缺口 4 机（loso__xsc_gap_n4） | v2·原始场景描述的critic | 23 | 600 | 0.421296 |
| loso：缺口 4 机（loso__xsc_gap_n4） | v2·无先验的critic | 23 | 600 | 0.410571873 |
| loso：缺口 4 机（loso__xsc_gap_n4） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 28.7962341 |
| loso：缺口 6 机（loso__xsc_gap_n6） | 学习选 η v2的先验加有界修正critic | 23 | 500 | 0.46232608 |
| loso：缺口 6 机（loso__xsc_gap_n6） | v2·原始场景描述的critic | 23 | 600 | 0.466648579 |
| loso：缺口 6 机（loso__xsc_gap_n6） | v2·无先验的critic | 17 | 400 | 0.462954581 |
| loso：缺口 6 机（loso__xsc_gap_n6） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 30.6403599 |
| loso：双瓶颈（loso__xsc_double_n4） | 学习选 η v2的先验加有界修正critic | 23 | 400 | 0.425242841 |
| loso：双瓶颈（loso__xsc_double_n4） | v2·原始场景描述的critic | 23 | 600 | 0.455842823 |
| loso：双瓶颈（loso__xsc_double_n4） | v2·无先验的critic | 23 | 400 | 0.417578995 |
| loso：双瓶颈（loso__xsc_double_n4） | 学习选 η v2的场景级提案生成器 | 17 | 200 | 16.0821724 |
| loso：错位口（loso__xsc_offset_n4） | 学习选 η v2的先验加有界修正critic | 23 | 400 | 0.470447183 |
| loso：错位口（loso__xsc_offset_n4） | v2·原始场景描述的critic | 23 | 400 | 0.488398254 |
| loso：错位口（loso__xsc_offset_n4） | v2·无先验的critic | 17 | 400 | 0.461509556 |
| loso：错位口（loso__xsc_offset_n4） | 学习选 η v2的场景级提案生成器 | 23 | 800 | 22.5506382 |
| lofo：缺口 4 机、缺口 6 机、双瓶颈、错位口（lofo__bottleneck） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.4444879 |
| lofo：缺口 4 机、缺口 6 机、双瓶颈、错位口（lofo__bottleneck） | v2·原始场景描述的critic | 23 | 200 | 0.487568647 |
| lofo：缺口 4 机、缺口 6 机、双瓶颈、错位口（lofo__bottleneck） | v2·无先验的critic | 23 | 200 | 0.449146003 |
| lofo：缺口 4 机、缺口 6 机、双瓶颈、错位口（lofo__bottleneck） | 学习选 η v2的场景级提案生成器 | 17 | 400 | 8.5120821 |
| lofo：窄通道、拥挤随机、圆形五机、八机静态障碍（lofo__literature） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.405504525 |
| lofo：窄通道、拥挤随机、圆形五机、八机静态障碍（lofo__literature） | v2·原始场景描述的critic | 23 | 200 | 0.413096398 |
| lofo：窄通道、拥挤随机、圆形五机、八机静态障碍（lofo__literature） | v2·无先验的critic | 23 | 200 | 0.452591121 |
| lofo：窄通道、拥挤随机、圆形五机、八机静态障碍（lofo__literature） | 学习选 η v2的场景级提案生成器 | 41 | 300 | 37.9130859 |
| full18：窄通道（full18__lit_narrow_passage） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.432930678 |
| full18：拥挤随机（full18__lit_crowded_random） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.432175964 |
| full18：圆形五机（full18__lit_circle_five） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.433209658 |
| full18：八机无障碍（full18__lit_swap8_free） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.444488019 |
| full18：八机静态障碍（full18__lit_swap8_obstacles） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.425291449 |
| full18：八机移动障碍（full18__lit_swap8_moving） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.432193935 |
| full18：对穿 4 机（full18__xsc_swap_n4） | 学习选 η v2的先验加有界修正critic | 41 | 600 | 0.426969588 |
| full18：对穿 6 机（full18__xsc_swap_n6） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.430500686 |
| full18：对穿 10 机（full18__xsc_swap_n10） | 学习选 η v2的先验加有界修正critic | 17 | 200 | 0.443901837 |
| full18：对穿 12 机（full18__xsc_swap_n12） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.445039034 |
| full18：对穿 8 机加密（full18__xsc_swap_n8_dense） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.454698563 |
| full18：缺口 2 机（full18__xsc_gap_n2） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.443089575 |
| full18：缺口 4 机（full18__xsc_gap_n4） | 学习选 η v2的先验加有界修正critic | 23 | 200 | 0.438416094 |
| full18：缺口 6 机（full18__xsc_gap_n6） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.435942054 |
| full18：宽口 4 机（full18__xsc_wide_n4） | 学习选 η v2的先验加有界修正critic | 23 | 600 | 0.443778127 |
| full18：双瓶颈（full18__xsc_double_n4） | 学习选 η v2的先验加有界修正critic | 17 | 200 | 0.428249389 |
| full18：错位口（full18__xsc_offset_n4） | 学习选 η v2的先验加有界修正critic | 17 | 400 | 0.437524229 |
| full18：三门 4 机（full18__xsc_three_doors_n4） | 学习选 η v2的先验加有界修正critic | 41 | 600 | 0.424281836 |

每个种子的完整验证曲线、耗时、数据哈希与检查点见各节点目录；内部共享的全部节点与来源见TRAINING_AUDIT.json、INNER_NODES.json及各折LAMBDA_SELECTION.json。

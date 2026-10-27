# 训练完成记录

11个去重场景留一场景11折、去重池内留一族2折、全18场景留一场景18折，共31折；70个模型配置各3个种子，共210次训练。各折数据文件只含源场景标签；内部验证状态序号除4余3。

原训练函数的固定初始化key=0保留；种子17、23、41控制各场景独立采样流。按源内验证损失选择种子，并列选较小种子；检查点改进阈值1e-5、优化器、步数上限和早停沿用指定源码。

| 折与留出场景 | 模型 | 选中种子 | 最佳步 | 验证损失 | 三种子任务墙钟秒数合计 |
|---|---|---:|---:|---:|---:|
| 留一场景：窄通道（lit_narrow_passage） | 完整状态与控制器critic（full） | 23 | 700 | 0.51687086 | 26.713 |
| 留一场景：窄通道（lit_narrow_passage） | 不看控制器的critic（no_controller） | 17 | 400 | 0.53918946 | 33.326 |
| 留一场景：窄通道（lit_narrow_passage） | 只看场景且保留控制器的critic（scene_only） | 23 | 800 | 0.5057174 | 33.232 |
| 留一场景：窄通道（lit_narrow_passage） | 状态条件修正生成器（generator） | 23 | 800 | 29.479021 | 12.986 |
| 留一场景：拥挤随机（lit_crowded_random） | 完整状态与控制器critic（full） | 41 | 500 | 0.48152313 | 22.493 |
| 留一场景：拥挤随机（lit_crowded_random） | 不看控制器的critic（no_controller） | 41 | 500 | 0.50520122 | 24.501 |
| 留一场景：拥挤随机（lit_crowded_random） | 只看场景且保留控制器的critic（scene_only） | 17 | 1300 | 0.45755333 | 37.281 |
| 留一场景：拥挤随机（lit_crowded_random） | 状态条件修正生成器（generator） | 23 | 1000 | 29.908125 | 13.662 |
| 留一场景：圆形五机（lit_circle_five） | 完整状态与控制器critic（full） | 41 | 500 | 0.44680998 | 22.454 |
| 留一场景：圆形五机（lit_circle_five） | 不看控制器的critic（no_controller） | 41 | 500 | 0.46644872 | 22.600 |
| 留一场景：圆形五机（lit_circle_five） | 只看场景且保留控制器的critic（scene_only） | 23 | 800 | 0.44169748 | 24.000 |
| 留一场景：圆形五机（lit_circle_five） | 状态条件修正生成器（generator） | 41 | 300 | 28.659456 | 12.784 |
| 留一场景：八机静态障碍（lit_swap8_obstacles） | 完整状态与控制器critic（full） | 41 | 500 | 0.44594637 | 37.320 |
| 留一场景：八机静态障碍（lit_swap8_obstacles） | 不看控制器的critic（no_controller） | 41 | 500 | 0.46927261 | 29.945 |
| 留一场景：八机静态障碍（lit_swap8_obstacles） | 只看场景且保留控制器的critic（scene_only） | 23 | 800 | 0.45283926 | 28.600 |
| 留一场景：八机静态障碍（lit_swap8_obstacles） | 状态条件修正生成器（generator） | 41 | 300 | 29.683521 | 15.501 |
| 留一场景：对穿4机（xsc_swap_n4） | 完整状态与控制器critic（full） | 41 | 500 | 0.44286543 | 49.318 |
| 留一场景：对穿4机（xsc_swap_n4） | 不看控制器的critic（no_controller） | 41 | 500 | 0.48514557 | 39.024 |
| 留一场景：对穿4机（xsc_swap_n4） | 只看场景且保留控制器的critic（scene_only） | 41 | 2100 | 0.42754596 | 31.202 |
| 留一场景：对穿4机（xsc_swap_n4） | 状态条件修正生成器（generator） | 41 | 800 | 28.987982 | 21.775 |
| 留一场景：对穿12机（xsc_swap_n12） | 完整状态与控制器critic（full） | 41 | 700 | 0.49277407 | 32.069 |
| 留一场景：对穿12机（xsc_swap_n12） | 不看控制器的critic（no_controller） | 41 | 500 | 0.50569677 | 42.822 |
| 留一场景：对穿12机（xsc_swap_n12） | 只看场景且保留控制器的critic（scene_only） | 17 | 1300 | 0.46915081 | 52.350 |
| 留一场景：对穿12机（xsc_swap_n12） | 状态条件修正生成器（generator） | 41 | 300 | 30.390238 | 16.498 |
| 留一场景：对穿8机加密（xsc_swap_n8_dense） | 完整状态与控制器critic（full） | 41 | 500 | 0.4939543 | 26.963 |
| 留一场景：对穿8机加密（xsc_swap_n8_dense） | 不看控制器的critic（no_controller） | 41 | 500 | 0.51307851 | 24.078 |
| 留一场景：对穿8机加密（xsc_swap_n8_dense） | 只看场景且保留控制器的critic（scene_only） | 23 | 800 | 0.48872727 | 32.115 |
| 留一场景：对穿8机加密（xsc_swap_n8_dense） | 状态条件修正生成器（generator） | 41 | 300 | 30.021185 | 19.244 |
| 留一场景：缺口4机（xsc_gap_n4） | 完整状态与控制器critic（full） | 41 | 700 | 0.4736945 | 24.412 |
| 留一场景：缺口4机（xsc_gap_n4） | 不看控制器的critic（no_controller） | 23 | 800 | 0.5018245 | 23.635 |
| 留一场景：缺口4机（xsc_gap_n4） | 只看场景且保留控制器的critic（scene_only） | 23 | 1900 | 0.44973278 | 45.993 |
| 留一场景：缺口4机（xsc_gap_n4） | 状态条件修正生成器（generator） | 23 | 1000 | 28.549168 | 22.745 |
| 留一场景：缺口6机（xsc_gap_n6） | 完整状态与控制器critic（full） | 41 | 500 | 0.49395931 | 42.227 |
| 留一场景：缺口6机（xsc_gap_n6） | 不看控制器的critic（no_controller） | 41 | 500 | 0.51803702 | 39.613 |
| 留一场景：缺口6机（xsc_gap_n6） | 只看场景且保留控制器的critic（scene_only） | 23 | 2100 | 0.46478662 | 61.925 |
| 留一场景：缺口6机（xsc_gap_n6） | 状态条件修正生成器（generator） | 41 | 300 | 30.811867 | 11.474 |
| 留一场景：双瓶颈（xsc_double_n4） | 完整状态与控制器critic（full） | 41 | 700 | 0.45798951 | 26.579 |
| 留一场景：双瓶颈（xsc_double_n4） | 不看控制器的critic（no_controller） | 41 | 500 | 0.50211191 | 20.873 |
| 留一场景：双瓶颈（xsc_double_n4） | 只看场景且保留控制器的critic（scene_only） | 23 | 1900 | 0.46815109 | 30.239 |
| 留一场景：双瓶颈（xsc_double_n4） | 状态条件修正生成器（generator） | 23 | 700 | 16.00919 | 16.902 |
| 留一场景：错位口（xsc_offset_n4） | 完整状态与控制器critic（full） | 41 | 500 | 0.4961997 | 47.042 |
| 留一场景：错位口（xsc_offset_n4） | 不看控制器的critic（no_controller） | 41 | 500 | 0.5237124 | 34.314 |
| 留一场景：错位口（xsc_offset_n4） | 只看场景且保留控制器的critic（scene_only） | 23 | 1900 | 0.47401816 | 31.586 |
| 留一场景：错位口（xsc_offset_n4） | 状态条件修正生成器（generator） | 41 | 800 | 22.577141 | 18.387 |
| 去重池内留一族：缺口4机（xsc_gap_n4）、缺口6机（xsc_gap_n6）、双瓶颈（xsc_double_n4）、错位口（xsc_offset_n4） | 完整状态与控制器critic（full） | 17 | 400 | 0.46711209 | 21.428 |
| 去重池内留一族：缺口4机（xsc_gap_n4）、缺口6机（xsc_gap_n6）、双瓶颈（xsc_double_n4）、错位口（xsc_offset_n4） | 不看控制器的critic（no_controller） | 17 | 400 | 0.49983919 | 23.446 |
| 去重池内留一族：缺口4机（xsc_gap_n4）、缺口6机（xsc_gap_n6）、双瓶颈（xsc_double_n4）、错位口（xsc_offset_n4） | 只看场景且保留控制器的critic（scene_only） | 17 | 1400 | 0.45511538 | 25.922 |
| 去重池内留一族：缺口4机（xsc_gap_n4）、缺口6机（xsc_gap_n6）、双瓶颈（xsc_double_n4）、错位口（xsc_offset_n4） | 状态条件修正生成器（generator） | 41 | 600 | 8.604702 | 10.531 |
| 去重池内留一族：窄通道（lit_narrow_passage）、拥挤随机（lit_crowded_random）、圆形五机（lit_circle_five）、八机静态障碍（lit_swap8_obstacles） | 完整状态与控制器critic（full） | 41 | 500 | 0.4759275 | 19.937 |
| 去重池内留一族：窄通道（lit_narrow_passage）、拥挤随机（lit_crowded_random）、圆形五机（lit_circle_five）、八机静态障碍（lit_swap8_obstacles） | 不看控制器的critic（no_controller） | 41 | 500 | 0.48562536 | 21.610 |
| 去重池内留一族：窄通道（lit_narrow_passage）、拥挤随机（lit_crowded_random）、圆形五机（lit_circle_five）、八机静态障碍（lit_swap8_obstacles） | 只看场景且保留控制器的critic（scene_only） | 41 | 1200 | 0.47956347 | 21.726 |
| 去重池内留一族：窄通道（lit_narrow_passage）、拥挤随机（lit_crowded_random）、圆形五机（lit_circle_five）、八机静态障碍（lit_swap8_obstacles） | 状态条件修正生成器（generator） | 23 | 800 | 38.060032 | 8.976 |
| 全18场景留一场景：窄通道（lit_narrow_passage） | 完整状态与控制器critic（full） | 17 | 600 | 0.47280777 | 60.662 |
| 全18场景留一场景：拥挤随机（lit_crowded_random） | 完整状态与控制器critic（full） | 17 | 900 | 0.45650208 | 84.309 |
| 全18场景留一场景：圆形五机（lit_circle_five） | 完整状态与控制器critic（full） | 17 | 900 | 0.43321466 | 64.166 |
| 全18场景留一场景：八机无障碍（lit_swap8_free） | 完整状态与控制器critic（full） | 41 | 500 | 0.47029096 | 79.809 |
| 全18场景留一场景：八机静态障碍（lit_swap8_obstacles） | 完整状态与控制器critic（full） | 41 | 500 | 0.45004234 | 61.900 |
| 全18场景留一场景：八机移动障碍（lit_swap8_moving） | 完整状态与控制器critic（full） | 41 | 500 | 0.45844233 | 75.442 |
| 全18场景留一场景：对穿4机（xsc_swap_n4） | 完整状态与控制器critic（full） | 17 | 900 | 0.44216764 | 80.741 |
| 全18场景留一场景：对穿6机（xsc_swap_n6） | 完整状态与控制器critic（full） | 41 | 500 | 0.46375486 | 75.300 |
| 全18场景留一场景：对穿10机（xsc_swap_n10） | 完整状态与控制器critic（full） | 41 | 500 | 0.47398573 | 80.385 |
| 全18场景留一场景：对穿12机（xsc_swap_n12） | 完整状态与控制器critic（full） | 41 | 500 | 0.47389138 | 63.050 |
| 全18场景留一场景：对穿8机加密（xsc_swap_n8_dense） | 完整状态与控制器critic（full） | 17 | 900 | 0.47725382 | 69.032 |
| 全18场景留一场景：缺口2机（xsc_gap_n2） | 完整状态与控制器critic（full） | 41 | 500 | 0.4918327 | 75.481 |
| 全18场景留一场景：缺口4机（xsc_gap_n4） | 完整状态与控制器critic（full） | 41 | 500 | 0.45847273 | 49.615 |
| 全18场景留一场景：缺口6机（xsc_gap_n6） | 完整状态与控制器critic（full） | 41 | 500 | 0.46187431 | 72.109 |
| 全18场景留一场景：宽门4机（xsc_wide_n4） | 完整状态与控制器critic（full） | 41 | 500 | 0.47709703 | 65.250 |
| 全18场景留一场景：双瓶颈（xsc_double_n4） | 完整状态与控制器critic（full） | 41 | 500 | 0.46061066 | 73.280 |
| 全18场景留一场景：错位口（xsc_offset_n4） | 完整状态与控制器critic（full） | 41 | 500 | 0.46807817 | 62.693 |
| 全18场景留一场景：三门4机（xsc_three_doors_n4） | 完整状态与控制器critic（full） | 17 | 900 | 0.45198491 | 68.560 |

每个种子的完整验证曲线与选中步保存在folds/<折>/<模型>/seed<种子>/history.json与FIT.receipt.json；选种子依据保存在SELECTED.json。TRAINING_AUDIT.json逐项复核早停、检查点字节哈希、选种子、源内标准化及留出隔离。

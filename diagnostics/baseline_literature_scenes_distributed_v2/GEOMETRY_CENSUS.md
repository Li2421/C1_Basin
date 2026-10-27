仅从保存的仿真轨迹重算。以下计数为描述性候选计数，不是类型等权发生率。约束法向、条件下界、名义速度与安全速度均取实际机器人二维块；非零安全速度、非零条件下界会使理想齐次锥解释不精确。逃逸距离为给定局部锥几何的描述量，不证明物理可解、控制器可达或长期死锁。

| 场景 | 安全层 | 候选轨迹 | 候选 agent-step | 最大安全速率 | 最小条件下界 | 最大条件下界 | 最小法向角° | 最大法向角° | 最小逃逸下确界 | 最大逃逸下确界 | 逃逸计算分类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 圆形围场五机随机场景 | 集中式联合投影 | 182 | 34300 | 0.373651 | -3.04466 | 1.74443 | 0.000612757 | 180 | 0 | 0.369164 | {"cone_is_full_space":7240,"empty_cone":19,"nominal_already_outside_cone_numerical_membership":7018,"origin_incident_supporting_facet":19111,"rank_deficient_cone_zero_infimum":912} |
| 圆形围场五机随机场景 | 分布式逐机器人投影 | 1497 | 262600 | 0.272738 | -1.84585 | 2.65807e-10 | 4.69475 | 180 | 0 | 0.372473 | {"cone_is_full_space":600,"empty_cone":17,"nominal_already_outside_cone_numerical_membership":22881,"origin_incident_supporting_facet":235047,"rank_deficient_cone_zero_infimum":4055} |
| 二维拥挤随机场景 | 集中式联合投影 | 8 | 340 | 0.310014 | -0.378146 | -0.0430223 | 87.879 | 159.472 | 0.224644 | 0.348139 | {"origin_incident_supporting_facet":340} |
| 二维拥挤随机场景 | 分布式逐机器人投影 | 7018 | 737358 | 0.90634 | -0.990641 | 1.33509e-10 | 0.00378645 | 180 | 0 | 0.999992 | {"cone_is_full_space":35366,"empty_cone":222,"nominal_already_outside_cone_numerical_membership":44649,"origin_incident_supporting_facet":655763,"rank_deficient_cone_zero_infimum":1358} |
| 门口场景 | 集中式联合投影 | 256 | 19050 | 0.134572 | -0.0221771 | -2.07499e-15 | 140.272 | 179.987 | 0 | 0.086957 | {"nominal_already_outside_cone_numerical_membership":494,"origin_incident_supporting_facet":18556} |
| 门口场景 | 分布式逐机器人投影 | 260 | 19250 | 0.134572 | -0.0221771 | -2.00534e-15 | 90.9742 | 179.987 | 0 | 0.294827 | {"nominal_already_outside_cone_numerical_membership":494,"origin_incident_supporting_facet":18756} |
| 窄通道场景 | 分布式逐机器人投影 | 7861 | 196525 | 0.877513 | -0.0706524 | -1.22263e-14 | 9.00663 | 179.995 | 0 | 0.999005 | {"empty_cone":41,"nominal_already_outside_cone_numerical_membership":350,"origin_incident_supporting_facet":196134} |

完整逐轨迹条目见 GEOMETRY_CENSUS.json，原始局部 A、条件下界与名义输入见 DIAGNOSTICS。数值缺口保持显式记录，不生成新的控制轨迹。

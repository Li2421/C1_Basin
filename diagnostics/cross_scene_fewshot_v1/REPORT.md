# Round23 cross-scene few-shot

主菜单的安全方法 S 在 k∈{1,2,4,8,16} 中没有达到全部30单元可靠性门，故 k*>16、所需标签数>64，精确值未识别；F1 REJECTED，F2 REJECTED（“所有单元/预算的 p90≤10pp”门未通过），F3 INCONCLUSIVE。这是已有前瞻标签的 DEVELOPMENT 模拟，不是新的实地部署确认。

F2 超阈值项 38；最难单元 ['R13_ring_exchange / flow_isotropic', 'R13_ring_exchange / flow_coupled', 'R13_toy_give_way / flow_isotropic', 'R13_toy_give_way / flow_anisotropic', 'R20_toy_044 / flow_anisotropic', 'R20_toy_044 / flow_coupled', 'R20_toy_050 / flow_isotropic', 'R20_toy_050 / flow_anisotropic', 'R20_toy_056 / flow_anisotropic', 'R20_toy_056 / flow_coupled']。
S−D0 在 k=16 的平均遗憾差 -5.243pp，单元配对95%CI [-9.191, -1.968]；k*不存在时仅作描述。

全部单位按 state_uid 排序前20校准/后20评估，无交叉；校准不采 unknown，评估同时保存上下界。次菜单、E与SE结果和逐单位概率见 results.json；两种菜单各有30面板PNG/PDF。

相对 round8 的128条标签，仅能比较研究情境和效应量：这里场景族之间差异大，但所有30单元的统一可靠性门可能受低效应量单元限制，不能从平均表现推出统一预算。

事实：F1/F2/F3 与逐单位指标见结果。解释：Beta安全门保留D0可减少低信息切换，但D0本身并非所有场景中的安全选择。建议更保守策略需要新预注册，本轮未实施。

主/次菜单方法的统一 k*：

| menu | E | S | SE |
| --- | --- | --- | --- |
| M | >16 | >16 | >16 |
| all7 | >16 | >16 | >16 |

主菜单各单元 k=16 的安全方法：

| unit | mean regret pp | p90 pp | P(regret≤3pp) | MC Wilson95 |
| --- | ---: | ---: | ---: | --- |
| R13_double_bottleneck / flow_isotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R13_double_bottleneck / flow_anisotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R13_double_bottleneck / flow_coupled | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R13_four_way_intersection / flow_isotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R13_four_way_intersection / flow_anisotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R13_four_way_intersection / flow_coupled | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R13_ring_exchange / flow_isotropic | 6.25 | 6.25 | 0.000 | [0.0, 0.004] |
| R13_ring_exchange / flow_anisotropic | 0.02 | 0.00 | 0.997 | [0.991, 0.999] |
| R13_ring_exchange / flow_coupled | 4.93 | 5.00 | 0.017 | [0.011, 0.027] |
| R13_toy_give_way / flow_isotropic | 5.61 | 26.25 | 0.707 | [0.678, 0.734] |
| R13_toy_give_way / flow_anisotropic | 27.19 | 46.25 | 0.000 | [0.0, 0.004] |
| R13_toy_give_way / flow_coupled | 0.01 | 0.00 | 0.998 | [0.993, 0.999] |
| R20_db_060 / flow_isotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_060 / flow_anisotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_060 / flow_coupled | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_075 / flow_isotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_075 / flow_anisotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_075 / flow_coupled | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_090 / flow_isotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_090 / flow_anisotropic | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_db_090 / flow_coupled | 0.00 | 0.00 | 1.000 | [0.996, 1.0] |
| R20_toy_044 / flow_isotropic | 0.98 | 2.50 | 0.998 | [0.993, 0.999] |
| R20_toy_044 / flow_anisotropic | 2.73 | 25.00 | 0.891 | [0.87, 0.909] |
| R20_toy_044 / flow_coupled | 8.20 | 10.00 | 0.185 | [0.162, 0.21] |
| R20_toy_050 / flow_isotropic | 8.70 | 20.00 | 0.565 | [0.534, 0.595] |
| R20_toy_050 / flow_anisotropic | 7.00 | 7.50 | 0.066 | [0.052, 0.083] |
| R20_toy_050 / flow_coupled | 0.12 | 0.00 | 0.978 | [0.967, 0.985] |
| R20_toy_056 / flow_isotropic | 1.66 | 0.00 | 0.949 | [0.934, 0.961] |
| R20_toy_056 / flow_anisotropic | 8.53 | 12.50 | 0.318 | [0.29, 0.348] |
| R20_toy_056 / flow_coupled | 4.54 | 12.50 | 0.638 | [0.608, 0.667] |

解释：DB等大效应单元较容易，但统一门还要求对小效应或相近臂的Toy/Ring单元可靠选对；在这些单元，保守后验门仍经常保留D0。F2尤其包含k=1，默认臂在部分单元本身损失很大，故仅“少切换”不能保证低遗憾。k*>16是本轮网格的下界，不能推出某个未测试预算必定成功。SE为次要对照；候选池耗尽时残余预算抽未耗尽的已淘汰候选，最终选择仍限幸存者，这个边界实现细节不影响主S检验。30单元嵌套在10场景/变体中，规定的单元bootstrap不能解释为30个独立拓扑的证据。

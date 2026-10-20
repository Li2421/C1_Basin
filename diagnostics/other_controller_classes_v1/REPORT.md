# Round27 other controller classes

O1 **REJECTED**，O2 **REJECTED**，O3 **REJECTED**。本轮不能将Flow上的时间表优势推广为跨类别的一般结论。Flow训练的A3跨类零样本优势未获支持。

能力门：M 6、P 6个实例保留，无替换、无性能调参；4个原Flow TT参照。主评估6144条，unknown 11保留未重跑。

**预注册偏差披露：** 原pilot/eval seed区间误重叠6个state，去重门在任何评估执行前失败。已保留失败审计，并在补充预注册中以pilot_start+16修正评估起点，重新通过去重；这是对原“重复即停、不重抽”条款的明确修订，不能称原计划无偏差通过。controller参数、能力门、阈值及臂均未变。

| class | best schedule−fixed pp | 95%CI pp | dominance | O2 | A3−A0 regret pp | O4−S500 regret pp |
| --- | ---: | --- | --- | --- | ---: | ---: |
| M | -1.82 | [-3.65, -0.26] | 2/6 | 2/6 | 0.00 | -1.82 |
| P | 0.00 | [0.0, 0.0] | 6/6 | 6/6 | 0.00 | 0.00 |
| NEW | -0.91 | [-2.08, 0.0] | 8/12 | 8/12 | 0.00 | -0.91 |
| Flow | 37.50 | [19.14, 61.33] | 4/4 | 4/4 | 3.91 | NA |

O1–O3最终判定包含unknown兼容范围；逐类/合并CI、点估计与边界见results.json。每类少于4实例时应记未定，本轮两类均6实例。O2按任务定义检验前向S500对两个端点的支配；未运行反方向，不能宣称直接证明双向排序共享。

O4为类内controller留一的DEVELOPMENT描述（仅对新M/P类，Flow的O4为NA）：5个同类source训练，R16选定step和原A3超参数固定，目标controller不用于训练、step或预处理；每折3fit，所有预测/checkpoint在o4目录。它不替代跨类零样本确认；若各controller都选同一臂，不能仅凭优于S500便声称学到了controller条件关系。

Flow参照与历史FF对照见FLOW_FF_TT_COMPARISON.json；既报各自菜单，又报共同F10/F13/S300/S500菜单。checkpoint相同但state与Q不同，不能把差值解释为同state配对的修正位置因果效应。

已知失败与终止步见FAILURE_RECORDS；unknown含首个数值错误步，未记作已知失败。timeout不直接称deadlock。完整位置/动作/连续运动swept clearance证据保存在main/traces。

去重：所有13209个DB state核对UID/content；物理三元组缺失/不匹配4机器人形状而无法额外核对者6465，不能声称跨表示物理去重全覆盖。

TT一致性门16案例：本轮执行与独立历史rollout_tt参照的首动作、native终止/长度和全轨迹逐字节一致。原生v10入口及配置schema构造问题在pilot成功标签前修复并留收据；控制器参数不变。共享DB只执行新身份登记、preflight/journal/单进程merge/postflight；probe仅登记source_file，无伪造task标签。纯Q按预注册免视频，未改历史代码/资产，未commit。

## Additional delivery context

同一批TT评估state下，4个Flow参照仍有明显时间表收益（最佳时间表−最佳固定 +37.50pp，CI [19.14,61.33]），而MPC类为−1.82pp，势场类持平。因此，本轮跨类负结果不能仅归因于把FF修正位置换成TT，更符合这些nominal controller行为不同；结论限这两个实现类别与所测参数。势场F13已达到100%，其持平有天花板限制，不能据此证明所有势场controller都不存在阶段效应。

| class | F10 % | F13 % | F2 % | S200 % | S300 % | S500 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M | 26.04 | 91.93 | 7.81 | 66.67 | 80.47 | 90.10 |
| P | 7.81 | 100.00 | 37.50 | 88.54 | 95.83 | 100.00 |
| Flow | 14.84 | 59.38 | 9.38 | 85.16 | 92.58 | 96.88 |

MPC和势场的冻结A3、A0均对全部6实例选择S500，故O3比较严格相等、CI为[0,0]，并非仅仅统计功效不足。类内O4在MPC上对6实例都选F13，改善1.82pp；这体现同类标签识别了共享较优臂，不能单凭该结果认定学到了细致的controller条件关系。势场O4在F13/S500间选择，两者均满分。P类为确定性nominal，Q2是两次真实执行，不把两个seed视作独立未来噪声的额外证据。

| class | known failure types | terminal step range | unknown | first error step range |
| --- | --- | --- | ---: | --- |
| M | {'timeout': 905} | [700, 700] | 5 | [303, 504] |
| P | {'timeout': 650} | [700, 700] | 4 | [200, 484] |
| Flow | {'timeout': 617} | [700, 700] | 2 | [232, 627] |

更宽的历史兼容性审计见HISTORICAL_TT_COMPATIBILITY_AUDIT.json：匹配checkpoint并检查结构化chain/placement及state/eta/seed，没有找到可直接用于这些gate案例的旧TT记录，故使用独立原TT入口重复一致性门。不能仅因新metadata UID不同便判旧标签失效。

实现修复前的controller/base protocol快照、原pilot manifest/preflight重建均保留，PREFLIGHT_V1_RECONSTRUCTION_AUDIT逐hash匹配原收据；当前运行使用V2冻结身份。状态区间修订为本轮已披露的预注册偏差。所有模型/探测冻结、修正后state去重和preflight均在主评估前完成。

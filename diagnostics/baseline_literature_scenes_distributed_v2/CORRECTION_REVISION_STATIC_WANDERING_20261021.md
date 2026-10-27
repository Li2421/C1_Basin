# 原语义目标附近徘徊与接触开关静态检查

只读取已有保存证据的终止构型，没有调用控制器、投影或物理，没有新rollout。超时且存在未到达机器人、全部未到达距目标≤3×原到达容差即为徘徊；统计开关关闭的未到达机器人比例，仅描述，不据此调Rc。每个rollout UID只计一次；确定性两seed是不同UID，保留各自权重。能力门、训练菜单与主线/评估分开报告，JSON给出全部。

| 场景 | 层 | 方法 | 合格 | 局数 | 徘徊局 | 比例 | 徘徊未到达机器人开关关闭比例 |
|---|---|---|---|---:|---:|---:|---:|
| 圆形围场五机随机场景 | central | fixed_eta10 | True | 1440 | 0 | 0.0000% | N/A |
| 圆形围场五机随机场景 | central | fixed_eta13 | True | 1440 | 6 | 0.4167% | 70.0000% |
| 圆形围场五机随机场景 | central | human_progress | True | 1440 | 0 | 0.0000% | N/A |
| 圆形围场五机随机场景 | central | safety | False | 96 | 0 | 0.0000% | N/A |
| 圆形围场五机随机场景 | central | safety | True | 1440 | 3 | 0.2083% | 100.0000% |
| 圆形围场五机随机场景 | distributed | fixed_eta10 | True | 1440 | 0 | 0.0000% | N/A |
| 圆形围场五机随机场景 | distributed | fixed_eta13 | True | 1440 | 8 | 0.5556% | 33.3333% |
| 圆形围场五机随机场景 | distributed | human_progress | True | 1440 | 3 | 0.2083% | 33.3333% |
| 圆形围场五机随机场景 | distributed | safety | False | 128 | 0 | 0.0000% | N/A |
| 圆形围场五机随机场景 | distributed | safety | True | 1440 | 4 | 0.2778% | 75.0000% |
| 二维拥挤随机场景 | central | fixed_eta10 | True | 9856 | 2761 | 28.0134% | 100.0000% |
| 二维拥挤随机场景 | central | fixed_eta13 | True | 9856 | 7289 | 73.9550% | 99.9805% |
| 二维拥挤随机场景 | central | human_progress | True | 9856 | 3675 | 37.2869% | 100.0000% |
| 二维拥挤随机场景 | central | safety | False | 896 | 296 | 33.0357% | 95.6208% |
| 二维拥挤随机场景 | central | safety | True | 9856 | 69 | 0.7001% | 100.0000% |
| 二维拥挤随机场景 | distributed | fixed_eta10 | True | 9856 | 2304 | 23.3766% | 100.0000% |
| 二维拥挤随机场景 | distributed | fixed_eta13 | True | 9856 | 7145 | 72.4939% | 100.0000% |
| 二维拥挤随机场景 | distributed | human_progress | True | 9856 | 3663 | 37.1652% | 100.0000% |
| 二维拥挤随机场景 | distributed | learned_scene | False | 240 | 0 | 0.0000% | N/A |
| 二维拥挤随机场景 | distributed | learned_scene | True | 2592 | 67 | 2.5849% | 100.0000% |
| 二维拥挤随机场景 | distributed | safety | False | 1120 | 194 | 17.3214% | 99.4059% |
| 二维拥挤随机场景 | distributed | safety | True | 9856 | 176 | 1.7857% | 100.0000% |
| 门口场景 | central | fixed_eta10 | True | 5760 | 0 | 0.0000% | N/A |
| 门口场景 | central | fixed_eta13 | True | 5760 | 2 | 0.0347% | 100.0000% |
| 门口场景 | central | human_progress | True | 5760 | 0 | 0.0000% | N/A |
| 门口场景 | central | learned_scene | False | 120 | 0 | 0.0000% | N/A |
| 门口场景 | central | learned_scene | True | 2808 | 0 | 0.0000% | N/A |
| 门口场景 | central | safety | False | 384 | 0 | 0.0000% | N/A |
| 门口场景 | central | safety | True | 5760 | 450 | 7.8125% | 99.6909% |
| 门口场景 | distributed | fixed_eta10 | True | 5760 | 0 | 0.0000% | N/A |
| 门口场景 | distributed | fixed_eta13 | True | 5760 | 2 | 0.0347% | 100.0000% |
| 门口场景 | distributed | human_progress | True | 5760 | 0 | 0.0000% | N/A |
| 门口场景 | distributed | learned_scene | False | 120 | 0 | 0.0000% | N/A |
| 门口场景 | distributed | learned_scene | True | 2808 | 0 | 0.0000% | N/A |
| 门口场景 | distributed | safety | False | 512 | 0 | 0.0000% | N/A |
| 门口场景 | distributed | safety | True | 5760 | 452 | 7.8472% | 98.1162% |
| 窄通道场景 | central | fixed_eta10 | True | 5632 | 384 | 6.8182% | 100.0000% |
| 窄通道场景 | central | fixed_eta13 | True | 5632 | 3387 | 60.1385% | 100.0000% |
| 窄通道场景 | central | human_progress | True | 5632 | 2220 | 39.4176% | 100.0000% |
| 窄通道场景 | central | learned_scene | False | 256 | 0 | 0.0000% | N/A |
| 窄通道场景 | central | learned_scene | True | 2624 | 0 | 0.0000% | N/A |
| 窄通道场景 | central | safety | False | 512 | 96 | 18.7500% | 100.0000% |
| 窄通道场景 | central | safety | True | 5632 | 134 | 2.3793% | 100.0000% |
| 窄通道场景 | distributed | fixed_eta10 | True | 5632 | 322 | 5.7173% | 100.0000% |
| 窄通道场景 | distributed | fixed_eta13 | True | 5632 | 3330 | 59.1264% | 100.0000% |
| 窄通道场景 | distributed | human_progress | True | 5632 | 2359 | 41.8857% | 100.0000% |
| 窄通道场景 | distributed | safety | False | 640 | 0 | 0.0000% | N/A |
| 窄通道场景 | distributed | safety | True | 5632 | 3 | 0.0533% | 100.0000% |

全部原结果不修改。此次源清单与终态检查在单元门收据中封存；最终报告将同时报告原/修订语义与两个阶段的数据。

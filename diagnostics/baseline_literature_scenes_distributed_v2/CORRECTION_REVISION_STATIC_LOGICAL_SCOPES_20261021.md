# 原主线范围的徘徊统计

原全UID静态审计包含能力门、白盒背景与标签等全部保存数据；本表按原baseline/fixed精确逻辑清单重聚合，避免将白盒背景并入黑盒baseline。两个配对seed各自保留。原静态审计不改。

| 场景 | 层 | 方法 | 合格性 | 局数 | 徘徊比例 | 开关关闭的未到达机器人比例 |
|---|---|---|---|---:|---:|---:|
| 窄通道场景 | central | safety | qualified | 5632 | 2.3793% | 100.0000% |
| 窄通道场景 | central | safety | unqualified | 512 | 18.7500% | 100.0000% |
| 窄通道场景 | central | fixed_eta10 | qualified | 5632 | 6.8182% | 100.0000% |
| 窄通道场景 | central | fixed_eta13 | qualified | 5632 | 60.1385% | 100.0000% |
| 窄通道场景 | central | human_progress | qualified | 5632 | 39.4176% | 100.0000% |
| 二维拥挤随机场景 | central | safety | qualified | 9856 | 0.7001% | 100.0000% |
| 二维拥挤随机场景 | central | safety | unqualified | 896 | 33.0357% | 95.6208% |
| 二维拥挤随机场景 | central | fixed_eta10 | qualified | 9856 | 28.0134% | 100.0000% |
| 二维拥挤随机场景 | central | fixed_eta13 | qualified | 9856 | 73.9550% | 99.9805% |
| 二维拥挤随机场景 | central | human_progress | qualified | 9856 | 37.2869% | 100.0000% |
| 圆形围场五机随机场景 | central | safety | qualified | 1440 | 0.2083% | 100.0000% |
| 圆形围场五机随机场景 | central | safety | unqualified | 96 | 0.0000% | N/A |
| 圆形围场五机随机场景 | central | fixed_eta10 | qualified | 1440 | 0.0000% | N/A |
| 圆形围场五机随机场景 | central | fixed_eta13 | qualified | 1440 | 0.4167% | 70.0000% |
| 圆形围场五机随机场景 | central | human_progress | qualified | 1440 | 0.0000% | N/A |
| 门口场景 | central | safety | qualified | 5760 | 7.8125% | 99.6909% |
| 门口场景 | central | safety | unqualified | 384 | 0.0000% | N/A |
| 门口场景 | central | fixed_eta10 | qualified | 5760 | 0.0000% | N/A |
| 门口场景 | central | fixed_eta13 | qualified | 5760 | 0.0347% | 100.0000% |
| 门口场景 | central | human_progress | qualified | 5760 | 0.0000% | N/A |
| 窄通道场景 | distributed | safety | qualified | 5632 | 0.0533% | 100.0000% |
| 窄通道场景 | distributed | safety | unqualified | 512 | 0.0000% | N/A |
| 窄通道场景 | distributed | fixed_eta10 | qualified | 5632 | 5.7173% | 100.0000% |
| 窄通道场景 | distributed | fixed_eta13 | qualified | 5632 | 59.1264% | 100.0000% |
| 窄通道场景 | distributed | human_progress | qualified | 5632 | 41.8857% | 100.0000% |
| 二维拥挤随机场景 | distributed | safety | qualified | 9856 | 1.7857% | 100.0000% |
| 二维拥挤随机场景 | distributed | safety | unqualified | 896 | 21.6518% | 99.4059% |
| 二维拥挤随机场景 | distributed | fixed_eta10 | qualified | 9856 | 23.3766% | 100.0000% |
| 二维拥挤随机场景 | distributed | fixed_eta13 | qualified | 9856 | 72.4939% | 100.0000% |
| 二维拥挤随机场景 | distributed | human_progress | qualified | 9856 | 37.1652% | 100.0000% |
| 圆形围场五机随机场景 | distributed | safety | qualified | 1440 | 0.2778% | 75.0000% |
| 圆形围场五机随机场景 | distributed | safety | unqualified | 96 | 0.0000% | N/A |
| 圆形围场五机随机场景 | distributed | fixed_eta10 | qualified | 1440 | 0.0000% | N/A |
| 圆形围场五机随机场景 | distributed | fixed_eta13 | qualified | 1440 | 0.5556% | 33.3333% |
| 圆形围场五机随机场景 | distributed | human_progress | qualified | 1440 | 0.2083% | 33.3333% |
| 门口场景 | distributed | safety | qualified | 5760 | 7.8472% | 98.1162% |
| 门口场景 | distributed | safety | unqualified | 384 | 0.0000% | N/A |
| 门口场景 | distributed | fixed_eta10 | qualified | 5760 | 0.0000% | N/A |
| 门口场景 | distributed | fixed_eta13 | qualified | 5760 | 0.0347% | 100.0000% |
| 门口场景 | distributed | human_progress | qualified | 5760 | 0.0000% | N/A |

保存轨迹机制核查：门口controller级失败实例。固定η10在两种语义下与同state/seed的失败基线配对；按封存顺序每实例取2局。近目标窗口及静止阈值在JSON明确列示，零有效帧照实保留。数值为第二次投影前的带权分量沿目标方向的和，实际朝目标动作另外给出；不作分量消融的因果声称。修正推动controller自身停住的实例，不计解开死锁。

|层|controller|state/seed|语义|近目标且名义静止帧|Bgoal方向和|Bperp方向和|Brel方向和|实际朝目标动作和|Brel为零帧|成功|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| central | centralized_sampling4 | (0, 0) | original | 11 | 1.85438331 | 0.0 | -0.45761317 | 1.3494372 | 0 | 1 |
| central | centralized_sampling4 | (0, 0) | revised | 7 | 1.07807414 | 0.0 | 0.0 | 0.85727918 | 7 | 1 |
| central | centralized_sampling4 | (0, 1) | original | 16 | 2.56752943 | 0.0 | -0.70780877 | 1.66439868 | 0 | 1 |
| central | centralized_sampling4 | (0, 1) | revised | 10 | 1.7297094 | 0.0 | 0.0 | 1.37789747 | 10 | 1 |
| central | centralized_sampling5 | (0, 0) | original | 10 | 1.33646894 | 0.0 | -0.43592932 | 0.88285089 | 0 | 1 |
| central | centralized_sampling5 | (0, 0) | revised | 10 | 1.69347534 | 0.0 | 0.0 | 1.35127933 | 10 | 1 |
| central | centralized_sampling5 | (0, 1) | original | 14 | 2.02565662 | 0.0 | -0.62077141 | 1.31468181 | 0 | 1 |
| central | centralized_sampling5 | (0, 1) | revised | 5 | 0.68552252 | 0.0 | 0.0 | 0.54034951 | 5 | 1 |
| central | centralized_mppi2 | (0, 0) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi2 | (0, 0) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi2 | (0, 1) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi2 | (0, 1) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi3 | (0, 0) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi3 | (0, 0) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi3 | (0, 1) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi3 | (0, 1) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi5 | (0, 0) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi5 | (0, 0) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi5 | (0, 1) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| central | centralized_mppi5 | (0, 1) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_sampling4 | (0, 0) | original | 12 | 2.11441062 | 0.0 | -0.54136509 | 1.49407091 | 0 | 1 |
| distributed | centralized_sampling4 | (0, 0) | revised | 10 | 1.63163672 | 0.0 | 0.0 | 1.16089653 | 10 | 1 |
| distributed | centralized_sampling4 | (0, 1) | original | 12 | 2.07730854 | 0.0 | -0.53792561 | 1.4697238 | 0 | 1 |
| distributed | centralized_sampling4 | (0, 1) | revised | 10 | 1.61323754 | 0.0 | 0.0 | 1.1415078 | 10 | 1 |
| distributed | centralized_sampling5 | (0, 0) | original | 10 | 1.64375249 | 0.0 | -0.45876593 | 1.14822457 | 0 | 1 |
| distributed | centralized_sampling5 | (0, 0) | revised | 10 | 1.73905988 | 0.0 | 0.0 | 1.21542201 | 10 | 1 |
| distributed | centralized_sampling5 | (0, 1) | original | 11 | 1.82857464 | 0.0 | -0.50304173 | 1.28477752 | 0 | 1 |
| distributed | centralized_sampling5 | (0, 1) | revised | 9 | 1.38627894 | 0.0 | 0.0 | 0.99066125 | 9 | 1 |
| distributed | centralized_mppi2 | (0, 0) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi2 | (0, 0) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi2 | (0, 1) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi2 | (0, 1) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi3 | (0, 0) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi3 | (0, 0) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi3 | (0, 1) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi3 | (0, 1) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi5 | (0, 0) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi5 | (0, 0) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi5 | (0, 1) | original | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| distributed | centralized_mppi5 | (0, 1) | revised | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |

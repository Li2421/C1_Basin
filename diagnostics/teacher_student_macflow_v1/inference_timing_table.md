# 单核推理耗时

JIT预热后、固定一个CPU核、FP32、准备好输入。单位ms。学生为单机器人单次网络推理，教师为联合10步Euler及联合噪声；QP为4个机器人的单次投影总计，不包含观测构建和完整闭环runner开销。

| Controller | Student / robot | 4×student (sequential estimate) | Teacher joint 10-step | Four local QPs |
|---|---:|---:|---:|---:|
| base_u_plus_w_v4_wide_macflow | 0.0597 | 0.2387 | 0.3368 | 0.1568 |
| base_u_macflow | 0.0498 | 0.1990 | 0.0674 | 0.1564 |
| base_u_v4_macflow | 0.0505 | 0.2019 | 0.0684 | 0.1562 |
| base_u_plus_w_v6_macflow | 0.0506 | 0.2023 | 0.0674 | 0.1567 |
| base_u_plus_w_v3_macflow | 0.0461 | 0.1843 | 0.0684 | 0.1594 |
| base_u_plus_w_v4_train_targeted_macflow | 0.0501 | 0.2004 | 0.0683 | 0.1556 |
| base_u_v3_macflow | 0.0501 | 0.2005 | 0.0684 | 0.1556 |
| base_u_v2_macflow | 0.0501 | 0.2005 | 0.0680 | 0.1547 |
| base_u_v10_local_macflow | 0.0540 | 0.2160 | 0.0678 | 0.1561 |
| base_u_plus_w_v5_macflow | 0.0579 | 0.2315 | 0.0683 | 0.1552 |
| base_u_v7_local_macflow | 0.0507 | 0.2027 | 0.0688 | 0.1544 |
| base_u_plus_w_v4_macflow | 0.0533 | 0.2131 | 0.0679 | 0.1560 |
| base_u_v8_local_macflow | 0.0511 | 0.2044 | 0.0684 | 0.1559 |
| base_u_v9_local_macflow | 0.0510 | 0.2038 | 0.0687 | 0.1565 |

4×学生列是顺序执行估计，不是一次批处理测量。每机器人延迟较低不等于在同一CPU上全队更快。复制的研究runner每步调用4次nominal，每次学生nominal计算4行，存在重复推理；本表不将其隐藏为端到端加速。完整单机器人和教师测量均值/中位数/p95及重复数见INFERENCE_TIMING.json。

专家引用round28：32/32成功，平均0.075s/episode规划；这是需要全部目标的特权参照，不与ms/step做速度比。

# Ring v4 wide：开发集局部动作审计

范围：只读取 `base_u_v4_dataset` 的 25 条开发 nominal 初始状态，并使用
`base_u_plus_w_v4_wide_macflow/best.pkl` 在每三个有效 policy 状态做一次
中央专家 re-query。没有读取或评估 test。

## 结果

该 checkpoint 的开发 rollout 为 24 个碰撞、1 个超时。与同一物理状态的
首个专家局部动作相比：

| 段 | agent 样本 | policy / expert 径向均值 | policy / expert 切向均值 | 局部 RMSE |
|---|---:|---:|---:|---:|
| 入口径向 | 6,918 | -0.0246 / -0.4567 | -0.0663 / -0.0009 | 0.4030 |
| 环行弧段 | 425 | -0.0046 / -0.0092 | -0.1012 / -0.0803 | 0.1767 |
| 出口径向 | 4,189 | -0.0237 / +0.4544 | -0.1245 / -0.0022 | 0.4530 |
| goal/等待 | 28 | +0.0120 / -0.4600 | +0.1120 / 0.0000 | 0.3687 |

入口和出口的主要残差是**径向速度塌缩/反向**，不是弧段左右方向的小误差。
该模型在入口几乎不向内、在出口几乎不向外；随后会累积至障碍或外界。

## 速度观测、归一化、采样检查

将同一位置/目标的 observation velocity 全部置零、并保持完全相同的 flow
随机键，动作变化范数为入口 0.283、弧段 0.374、出口 0.375。这说明该模型
确实强烈使用 velocity，但不是 velocity 特征的数值 OOD：velocity feature
绝对 z-score 的 p99 为 1.844、最大为 2.522，`|z|>3` 的比例为零。

内部 flow 采样有 7.38% 的原始 joint agent 动作速度略超过 0.52；采集器已
在 host 端消除 float32 的 `0.52000004` 数值超限。分量内部 clip 命中仅
`4.3e-5`，因此不是组件裁剪或归一化爆炸造成的主故障。

## 发现并修复的专家局部标签问题

从出口径向或已到目标的恢复状态重新查询时，旧专家会把 agent 再送回
`route_radius` 后才向目标走；到目标的 agent 甚至会重新入环。这会产生
反向出口/goal 标签。现已修复：

- individual goal error 在 tolerance 内：保持静止；
- 当前—目标角度差不超过 0.14 rad：在安全外环直接连到 goal。

该修复仅改变中央专家 recovery 数据，未改变环境、观察、MACFlow 架构或
部署策略。回归：Ring 单元测试 5 passed；30 个 development 初始状态的
CW/CCW expert pilot 仍为 60/60 成功、零碰撞。

JSON 原始聚合：`diagnostics/ring_exchange_stage1/v4_wide_dev_local_action_audit/report.json`。

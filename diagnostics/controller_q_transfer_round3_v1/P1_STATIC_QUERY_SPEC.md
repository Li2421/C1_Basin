# P1 静态查询具体化

在读取 P1 新 rollout outcome 前，冻结 R3/R4_action 查询。每个 state 与 controller 的 R3 调用历史 `controller_context` 七姿态及 `field_context` 七姿态×三伪时间×五动作，再用项目原 `compact_field` 得到每agent64维；flatten后用于critic。

R4_action用该state的固定 seed0/t0 JAX key，对 η=0 的投影FF执行动作查询，及三个η维度中央差分ε=0.001。当前state之外另取一个短时 **open-loop probe**：四agent位置沿各自当前位置→目标的单位方向前移0.1，初速度和目标不变；不调用env.step，也不读取未来rollout。两姿态各得到8维动作+8×3 Jacobian，共64维。查询失败必须保留为缺失，不允许按outcome删样本/Controller。所有动作是完整FF+硬投影的静态查询，其局部Jacobian不是闭环Q的精确充分统计量。

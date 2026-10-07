# Ring hard-safety 碰撞审计

## 结论

旧 B0 的 14/960 次碰撞全部是外圆边界碰撞：中心障碍 0、agent-agent 0。根因分类为 **S5（实现适配遗漏）**。

Ring 的物理环境同时包含中心禁入圆和外圆边界，但旧 `polygonal_obstacle_snapshot` 只将中心障碍的 48 条边送入通用 wall-CBF，完全没有编码外边界。14 个案例中投影器都满足了它实际收到的中心障碍/两两 agent 约束；这不能保护未编码的外圆边界。

## 逐例复现

`ring_safety_collision_traces.json` 保存全部 14 个案例的：碰撞前状态、Flow action、投影后 action、求解器状态、约束残差、下一状态、endpoint/swept 碰撞检查以及修复后的完全相同输入重放结果。

- 历史碰撞：14
- outer-boundary：14
- central-obstacle：0
- agent-agent：0
- 旧实现所有已编码约束均满足：是
- 修复后同输入重放碰撞：0/14

## 最小通用修复

在相同 wall-CBF snapshot 中加入一个保守的、内接于物理外圆的 48 边多边形；中心障碍仍使用原来的 48 边外接近似。未修改容差、CBF 方程、碰撞定义或个例参数。

新增回归测试验证 snapshot 同时包含中心障碍与外边界。该修复改变 Ring safety artifact 的 hash，因此本轮 Ring fresh test 使用新 hash，不复用旧 safety rollout；Double/Four-Way safety 实现不变。

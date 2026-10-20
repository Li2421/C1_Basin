# 可选：完整仿真视频交付规则

**默认不要求视频。** 只有当用户或本轮 prompt 明确要求视频时，才按本文件执行。
在 prompt 中启用的写法（复制这一行即可）：

> 本轮需要视频：按 `docs/video_delivery_optional.md` 交付。

启用后，视频是该 batch 交付的一部分：视频缺失，batch 交付即未完成。

## 1. 适用范围与数量

适用于 `toy_giveway`、`double_bottleneck`、`four_way_intersection`、`ring_exchange`，以及参与 batch 的 `bottleneck_family` 母版。
每个场景至少交付 3 个完整仿真视频：
- 两个不同的死锁解决案例：一个标为 `deadlock_show`，另一个标为 `deadlock_resolution`；
- 一个独立的 `safety_success` 案例。

用户另行指定数量时，以用户要求为准。

## 2. 内容要求

- 必须从真实 rollout 的初始状态播放到终止状态，逐步保留整个过程；不能用静态轨迹图、最后几帧、局部恢复片段、合成轨迹或重复视频充数。
- 死锁解决案例必须显示同初态、同 seed 的死锁基线与成功干预轨迹；标明 controller/checkpoint、seed、终止原因及死锁判据。timeout 不能直接冒充 deadlock。
- safety 成功案例必须全程无碰撞并成功到达目标；保留实际 simulator 的连续运动碰撞检查结果，不能仅凭渲染图或采样位置间距判断安全。
- 图像/视频生成只读取仿真证据，不改控制器、物理、事件或成功标准。
- `bottleneck_family` 母版的视频要明确 N 与 geometry；几何预览图和单元测试用轨迹不算正式视频。

## 3. 生成入口与验收

- 用 `docs/batch_videos.md` 中的统一入口生成 MP4 和可追溯的 manifest。batch 最终报告之前必须运行该入口并成功。
- 没有合格案例时报告缺口，不得伪造成功，也不能默默跳过场景。
- 新增或修改 batch launcher 时，把视频生成命令接入最后的成功路径，并用非零退出码阻止缺视频的 batch 被标成完成。

## 4. 已知缺口

`ring_exchange` 与 `four_way_intersection` 目前没有正式的死锁判据，统一入口会拒绝这两个场景的死锁角色，并返回非零。
启用视频时，这两个场景的死锁角色只能如实报告缺口；可以另附同初态、同 seed 的“停滞或超时 vs 成功”对比视频，但必须明确标注不是 deadlock。

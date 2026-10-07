# 全仓库要求

## 四场景大 batch 的强制视频交付

适用于 `toy_giveway`、`double_bottleneck`、`four_way_intersection`、`ring_exchange`。
每次大型训练/评估/跨场景实验 batch 都必须交付每场景至少 3 个完整仿真视频：
两个不同的死锁解决案例（其中一个标为 `deadlock_show`），以及一个独立的
`safety_success` 案例。另一个死锁案例标为 `deadlock_resolution`。
这是对当前数量描述的保守执行；用户后续明确数量时以用户要求为准。

- 必须从真实 rollout 的初始状态播放到终止状态，逐步保留整个过程；不能用
  静态轨迹图、最后几帧、局部恢复片段、合成轨迹或重复视频充数。
- 死锁解决案例必须显示同初态、同 seed 的死锁基线与成功干预轨迹；标明
  controller/checkpoint、seed、终止原因及死锁判据。timeout 不能直接冒充 deadlock。
- safety 成功案例必须全程无碰撞并成功到达目标；保留实际 simulator 的连续运动
  碰撞检查结果，不能仅凭渲染图或采样位置间距判断安全。
- 用 `docs/batch_videos.md` 中的统一入口生成 MP4 和可追溯 manifest。
  大 batch 最终报告之前必须运行该入口并成功；视频缺失则 batch 交付未完成。
  没有合格案例时报告缺口，不得伪造成功，也不能默默跳过场景。
- 新增/修改 batch launcher 时将视频生成命令接入最后的成功路径，使用非零退出码
  阻止缺视频的 batch 被标成完成。历史脚本尚未统一改造，运行它们也必须遵循此要求。
- 图像/视频生成只读取仿真证据，不改控制器、物理、事件或成功标准。
- rollout 运行前继续遵守 `shared_rollout_db/README.md` 的 preflight/cache 规则。

## 仓库清理和备份

先 commit/push 可版本管理的代码、配置和文档，再删除确认无用的文件。
Git 认证/账户失败时暂停并向用户确认账户。大型数据、checkpoint、SQLite、
未判明用途的研究证据先保留；Git push 不代表这些忽略文件已经备份。
`archive/legacy_pre_git_split` 已从活动目录移除，需历史代码时从清理前提交恢复。

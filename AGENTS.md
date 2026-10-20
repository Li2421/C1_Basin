# 全仓库要求

## 实验前先检查历史证据

每项实验开始前，先检索历史实验的配置、数据、checkpoint、预测和报告，确认
同一问题是否已测试，以及旧证据能否复用。记录找到的实验路径、兼容性和仍缺的
证据；能够通过旧结果、旧预测重算或旧 checkpoint 推理回答的问题，不重复训练
或生成 rollout。只有明确存在证据缺口时才补跑，并说明差异与必要性。
更换 critic 输入编码本身不使 execution semantics 相同的旧 rollout 标签失效。
rollout 运行前遵守 `shared_rollout_db/README.md` 的 preflight/cache 规则。
timeout 不能直接当作 deadlock。图像只读取仿真证据，不改控制器、物理、事件或成功标准。

## 视频（可选）

默认不要求视频交付。用户或 prompt 明确要求视频时，按 `docs/video_delivery_optional.md` 执行。

## 仓库清理和备份

先 commit/push 可版本管理的代码、配置和文档，再删除确认无用的文件。
Git 认证/账户失败时暂停并向用户确认账户。大型数据、checkpoint、SQLite、
未判明用途的研究证据先保留；Git push 不代表这些忽略文件已经备份。
`archive/legacy_pre_git_split` 已从活动目录移除，需历史代码时从清理前提交恢复。

## 可扩展场景母版

母版为 `bottleneck_family/`：以已有死锁研究的 Gap 双向交换为主配置，
Double-Bottleneck 为串联瓶颈变体；依据见 `docs/scalable_scene_audit_20261007.md`。

- 支持 N=2/10/20/50/100 的方向；固定地图数量扫描保持机器人半径、动力学和
  已有任务不变，新增机器人必须参与共享资源竞争，不能平铺独立二机器人案例充数。
- 障碍变化与数量变化分开测试。容量不足或通道封死必须报告，不能通过缩小机器人
  或把几何不可达标成 deadlock 来掩盖。
- 区分几何有效、单机器人有路、联合可解、控制器成功和死锁解决五层证据。
  当前母版只完成工程层验证；`stalled_groups` 只是诊断候选，正常排队也可能满足。
- 旧四场景是机制/回归基准；现有固定数量 checkpoint 不能宣称直接支持 N=100。

# 四场景 strict LOSO 视频索引

本目录复用 `diagnostics/physical_state_critic_minimal_v1/strict/` 的八组冻结预测，以及 `diagnostics/orthoflow3_loso_root_cause_v1/` 的历史候选和真值。15 个选中 continuation 的共享数据库 preflight 全部为 `EXACT_REUSE`；旧记录缺少逐步位置，因此只对这 15 个精确的 `(state, eta, controller, future_index)` 重放以导出轨迹。重放的**结局与 episode 步数均与历史缓存一致**，没有重新训练或生成新的 Q 标签。

所有 MP4 保留初态至终止的**每一步**，并把播放帧率调整到视频少于 20 秒。视频中的 `sim t` 是原始仿真时间；播放倍速写在画面和 [manifest.json](videos/manifest.json) 中。最长 18.50 秒。15 条轨迹全部无碰撞，连续运动最小安全余量由 simulator 的 swept-distance 信息逐步导出；视频帧数已用 `ffprobe` 与轨迹状态数逐条比对。

| 场景 | 场景内 η 成功 | LOSO η 成功 | η 失败案例 | 其他必要对照 |
|---|---|---|---|---|
| Toy Give-Way | [state15 成功](videos/toy_within_state15_success.mp4) | [state20 成功](videos/toy_loso_state20_success.mp4) | [场景内 state20 死锁](videos/toy_within_state20_deadlock.mp4)；[LOSO state15 死锁](videos/toy_loso_state15_deadlock.mp4) | [同初态/同 seed 对照 15](videos/toy_state15_within_vs_loso.mp4)；[对照 20](videos/toy_state20_within_vs_loso.mp4) |
| Double-Bottleneck | [state06 成功](videos/db_within_state06_success.mp4) | [state01 成功](videos/db_loso_state01_success.mp4) | [LOSO state06 超时](videos/db_loso_state06_timeout.mp4) | [state06 场景内/LOSO 对照](videos/db_state06_within_vs_loso.mp4)；[safety-only 超时](videos/db_safety_baseline_state06_timeout.mp4) |
| Four-Way Intersection | [state00 成功](videos/four_within_state00_success.mp4) | [state00 成功](videos/four_loso_state00_success.mp4) | 历史有效 continuation 没有可证实的所选 η 失败 | [safety-only 超时](videos/four_safety_baseline_state00_timeout.mp4)；[场景内/LOSO](videos/four_state00_within_vs_loso.mp4)；[safety-only/LOSO](videos/four_state00_safety_vs_loso.mp4) |
| Ring Exchange | [state16 成功](videos/ring_within_state16_success.mp4) | [state12 成功](videos/ring_loso_state12_success.mp4) | [场景内 state12 超时](videos/ring_within_state12_timeout.mp4)；[LOSO state16 超时](videos/ring_loso_state16_timeout.mp4) | [同初态/同 seed 对照 12](videos/ring_state12_within_vs_loso.mp4)；[对照 16](videos/ring_state16_within_vs_loso.mp4) |

场景内 Double-Bottleneck 的所选 η 为 24/24 robust 成功，因此不存在该面板的失败样本；提供原始 safety-only 失败作为**不同控制链**的对照。Four-Way 的场景内/LOSO 所选 η 只有 21/24、22/24 **已确认 robust**，其余主要是数值/缺失种子导致的未决标签，不能冒充物理失败；提供原始 safety-only 超时，并保持此区别。所有 `timeout` 均仅标为超时，绝不自动归类为死锁。Toy 两条 `deadlock` 使用原 simulator 的严格判据。

四个主要 η 成功片分别是 Toy state15、Double-Bottleneck state06、Four-Way state00、Ring state16 的场景内视频；LOSO 的四个 η 成功片也列在表中。每条 η、Q16、robust 标签、checkpoint SHA256、缓存 rollout UID、源记录、视频/trace SHA256 均见 [manifest.json](videos/manifest.json)。这批案例是依据已经打开的历史标签选取的**演示样本**，不构成新的盲测或新的跨场景性能估计；总体数字仍以 [strict LOSO 历史报告](../physical_state_critic_minimal_v1/strict/REPORT.md) 为准。

导出入口：`make_plan.py` → `shared_rollout_db.plan` → `replay_traces.py` → `render.py`。Four-Way/Ring 重放加载了本目录保存的历史 MACFlow 源快照（SHA256 `d02e61d0b5447f0da206b839ab292d5f7655ec44e6be866b15a5227d7cb06fac`），避免随后 Gap1 模型扩展造成的代码指纹漂移。标准 batch 视频渲染器只加了可选播放 FPS/标题参数，默认原速行为不变；`tests/test_batch_videos.py` 的 13 个测试均通过。

# 四场景 batch 视频交付

每个大 batch 必须有 **12 个正式视频**：每场景两个不同死锁解决案例和一个独立
safety 成功案例。其中 `deadlock_show.mp4` 为展示版，仍必须保留全过程。
四场景为 `toy_giveway`、`double_bottleneck`、`four_way_intersection`、`ring_exchange`。
数量按 2026-10-07 用户描述保守执行：两个死锁解决 + 一个 safety 成功。

## 运行入口

系统需有 `ffmpeg` 和 `ffprobe`，Python 依赖见 `requirements-video.txt`：

```bash
python3 -m venv .venv-video
.venv-video/bin/python -m pip install -r requirements-video.txt
.venv-video/bin/python -m new_benchmark_common.batch_videos \
  --plan diagnostics/MY_BATCH/video_plan.json \
  --output diagnostics/MY_BATCH/videos
```

优先使用 `scripts/run_batch_with_videos.sh PLAN OUTPUT -- COMMAND ...`，它会先运行
实验，再自动执行视频验收。COMMAND 必须等待计算结束（例如 `sbatch --wait`，
不能只提交异步任务）。也可将上述命令放在大型 batch launcher 的最后一步，
脚本须启用 `set -euo pipefail`。
只有退出码为 0、`videos/manifest.json` 的 `complete` 为 true 才能宣布交付完成。
历史 launcher 分散在各个 `diagnostics/` 实验目录中，没有自动接入；运行历史
launcher 后也必须执行上述命令。新 launcher 必须接入，不得跳过。
`--validate-only` 可提前检查素材，但不代表视频已交付。

缺少合格成功案例、实际死锁证据或完整 trace 时，命令失败。应报告缺口，并按
原实验协议补齐真实案例；不得把 timeout 改名为 deadlock，也不得更改事件阈值。
Four-Way/Ring 的部分旧协议没有 runtime deadlock 判据，此类旧 timeout 记录
不能直接充数；必须有已定义且记录在案的死锁证据。

## batch plan

`video_plan.json` 的路径相对该文件所在目录。下面是单场景片段，实际必须包含四个场景：

```json
{
  "batch_id": "MY_BATCH",
  "scenarios": {
    "toy_giveway": [
      {"role": "deadlock_show", "trace": "traces/toy_resolved_1.npz", "baseline": "traces/toy_deadlock_1.npz"},
      {"role": "deadlock_resolution", "trace": "traces/toy_resolved_2.npz", "baseline": "traces/toy_deadlock_2.npz"},
      {"role": "safety_success", "trace": "traces/toy_safe.npz"}
    ]
  }
}
```

两个死锁视频左右对照同一完整初态、seed、环境配置下的死锁基线与成功干预。
短的一侧结束后停留在终态，另一侧继续播放到结束。不裁剪，不跳帧，不加速。
画面包括墙体/障碍、真实半径的机器人、目标、已走轨迹、时间、终止事件、
controller、seed、安全开关和累计最小连续运动安全余量。

## trace 导出契约

从真实 simulator rollout 导出 NPZ（`np.savez_compressed`），不允许 pickle：

| 字段 | 内容 |
|---|---|
| `positions` | `[T+1,N,2]`，初态 + 每一步实际位置 + 终态，世界坐标 |
| `steps` | `np.arange(T+1)`，完整 episode，不能只导出 recovery 后缀 |
| `goals` | `[N,2]` 世界坐标目标 |
| `walls` | `[W,2,2]` 实际墙段；Ring 用形状 `[0,2,2]` 的空数组 |
| `swept_clearance` | `[T]`，每个 transition 的 simulator 连续运动最小净安全余量（扣除适用 margin），不能用离散帧距离替代 |
| `metadata_json` | `np.array(json.dumps(metadata))`，下述元数据 |

```python
metadata = {
    "batch_id": "MY_BATCH",
    "scenario": "toy_giveway",
    "rollout_id": "actual_rollout_uid",
    "controller": "actual_controller_uid",
    "checkpoint": "actual_checkpoint_path_and_hash_or_nonlearned_controller_version",
    "seed": 123,
    "source_record": "actual_raw_record_path_and_id",
    "initial_state_sha256": "hash_of_full_simulator_initial_state_including_velocities_and_timers",
    "config": actual_config_dict,  # 包含 dt、agent_radius；Ring 还需 outer/obstacle_radius
    "start_step": 0,
    "episode_steps": T,
    "complete": True,
    "termination": actual_terminal_event,  # success/deadlock/timeout/collision/numerical_failure
    "collision": actual_collision_boolean,
    "safety_enabled": actual_safety_boolean,
    # 死锁基线额外必须提供：
    "deadlock_detected": True,
    "deadlock_criterion": "actual_detector_name_version_and_thresholds",
}
```

需完整保存当前执行 episode 的 augmented 初态，而不仅是机器人位置。已有缓存若与
本 batch 精确兼容，按共享数据库规则复用并在 `source_record` 记录原始证据；只在新
导出元数据中关联本 batch，不修改原始实验记录。渲染器检查素材完整性和标签一致性，
事件本身的真实性仍须由 simulator 原始记录和审查确认，渲染器不会重新判定死锁。

输出 manifest 记录 trace/video SHA256、完整 provenance、帧数和 dt。视频与大型
trace 保留在实验目录，不进普通 Git。失败重跑会移除旧成功 manifest，避免旧视频
掩盖当前 batch 失败。此工具不自行启动实验或生成虚构的研究案例。

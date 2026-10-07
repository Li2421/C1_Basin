# Basin C1：四场景实验

当前维护四个场景，共享 Flow-BC、安全投影、rollout 缓存和研究工具：

| 目录 | 职责 |
|---|---|
| `toy_giveway/` | 双机器人 Give-Way；兼容入口复用 `single_integrator/` 的实现 |
| `double_bottleneck/` | 四机器人双瓶颈 |
| `four_way_intersection/` | 四向交叉路口 |
| `ring_exchange/` | 环形交换 |
| `single_integrator/`、`flowbc/`、`shared_control/` | 共享/历史兼容的物理、策略和控制实现，仍被使用 |
| `bottleneck_family/` | 可变 N 的 Gap/双瓶颈母版，几何与动力学已实现；大规模解锁待验证 |
| `new_benchmark_common/` | 共享数据、训练、评估和完整仿真视频工具 |
| `shared_rollout_db/` | 跨实验 rollout 索引与去重；运行前遵守其 README |
| `diagnostics/` | 实验脚本、研究报告及本地生成证据 |
| `datasets/`、`results/` | 本地数据和历史结果，大型资产不进普通 Git |
| `docs/`、`tests/`、`scripts/` | 协议、审查记录、测试与启动入口 |

可扩展母版的选型依据和四场景审查见 [母版审查](docs/scalable_scene_audit_20261007.md)，
使用入口见 [bottleneck_family](bottleneck_family/README.md)。
2/10/50-agent 的 Joint Flow Matching 训练与完整视频进度见
[Gap 1 规模试验](docs/gap_flow_scaling_20261007.md)。

## 大 batch 必须交付视频

每场景至少 3 个完整仿真视频：2 个死锁解决案例（其中 1 个用于 show）+ 1 个
safety 成功案例，共至少 12 个。从初态播放到终态，必须能看到整个过程。
仓库级规则见 [AGENTS.md](AGENTS.md)，trace 导出和验收见
[视频交付协议](docs/batch_videos.md)。无合格案例必须报告缺口，不能伪造或用截图代替。

```bash
# 命令应等待计算完成并导出 video_plan.json 及完整 traces。
# sbatch 必须使用 --wait，或在所有 shard 完成后的最终汇总 job 中执行验收。
bash scripts/run_batch_with_videos.sh \
  diagnostics/MY_BATCH/video_plan.json diagnostics/MY_BATCH/videos \
  -- bash path/to/experiment.sh
```

准备视频环境：`python3 -m venv .venv-video`，然后
`.venv-video/bin/python -m pip install -r requirements-video.txt`。
系统另需 `ffmpeg`、`ffprobe`。历史实验 launcher 尚未逐个改造，需通过上述入口
运行或在完成后显式执行视频验收命令。

## 测试和历史

```bash
bash scripts/test.sh
.venv-video/bin/python -m unittest discover -s tests -p test_batch_videos.py
```

共享基线测试可通过 `C1_PYTHON` 指定带 numpy/scipy/jax 的解释器。
历史分支 `direction-a-stochastic`、`direction-r-risk` 的职责保留在 Git 历史中；
当前 `main` 已包含四场景研究代码，不能再按旧 README 当作纯 Give-Way 基线。

2026-10-07 清理前快照已推送：`43b5264`。旧实现副本 `archive/legacy_pre_git_split/`
已移除，可从该提交恢复。清理范围、保留项和 Git 备份边界见
[本次目录审查](docs/workspace_cleanup_20261007.md)。

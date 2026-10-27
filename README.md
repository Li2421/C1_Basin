# Basin C1：异构场景流水线（hetero set）

当前流水线使用 3 个种子的 generator 组合，为每个场景提出 16 个 η；critic 在菜单标签与留一 generator 提案标签上重训，只在这 16 个提案中选一个。

## 当前代码与结果

| 目录 | 内容 |
|---|---|
| `diagnostics/hetero_set_critic_selfcheck_v1/` | 场景、标签、冻结选择、验证/测试结果；完整工作区快照也在这里 |
| `diagnostics/hetero_set_learned_planners_v1/` | 学习型规划器代码、checkpoint 和评估 |
| `diagnostics/hetero_set_test_videos_20261027/` | 测试视频与对应证据 |

完整源码和运行说明见 [工作区快照](diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/README.md)。
两批冻结文件位于快照的 `hetset/FINAL_FROZEN.json`（8 个验证场景及第一批 12 个测试场景）和 `hetset/FROZEN_T2.json`（第二批 12 个测试场景）。
快照中的脚本仍使用原任务临时目录绝对路径；临时目录失效后，按快照 README 重定位路径再运行。快照和冻结结果在本次清理中保持逐文件不变。

## 共享依赖

`shared_control/`、`single_integrator/`、`bottleneck_family/`、`new_benchmark_common/` 及旧场景兼容模块继续保留。`shared_rollout_db/` 包含共享代码和其他实验线使用的数据，本次不改数据库。

`diagnostics/` 中保留的旧版目录用于控制器与 η 菜单、复放轨迹、哈希审计或兼容入口，保留它们不表示继续使用旧版学习方法。具体依赖和保留理由见 [清理记录](docs/cleanup_hetero_set_20261027.md) 与 [路径清单](docs/cleanup_hetero_set_20261027.json)。

## 历史资料与验证

2026-10-27 清理前的源码、文档和选定小型配置快照已推送：`db959cb`。
336 个旧实验目录及其他过时内容已移出工作区，保留于 `/home/zhihan/research/Basin_C1_retired_20261027/`，原位置不留链接。
这是同磁盘归档，不是独立灾备；大型数据和 checkpoint 没有通过 Git 备份。恢复旧实验时，按清单恢复其原相对路径。

共享实现的回归检查入口为 `bash scripts/test.sh`；可用 `C1_PYTHON` 指定现有环境。
实验、证据复用与视频要求以 [AGENTS.md](AGENTS.md) 为准；[视频默认可选](docs/video_delivery_optional.md)。

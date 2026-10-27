# Hetero set 工作区清理（2026-10-27）

## 保留边界

当前流水线是 3 个种子的 generator 组合、每场景 16 个 η 提案、critic 仅从这 16 个提案中选择。冻结版本为 `FINAL_FROZEN.json` 和 `FROZEN_T2.json`。

完整保护以下目录，清理前已对 7,193 个目录项建立校验清单（文件内容 686,701,022 字节）：

- `diagnostics/hetero_set_critic_selfcheck_v1/`，含完整 `workspace_snapshot_20261027/`。
- `diagnostics/hetero_set_learned_planners_v1/`。
- `diagnostics/hetero_set_test_videos_20261027/`。

用户列出的运行依赖大目录和 `shared_rollout_db/` 整体保留，不做内部瘦身。另保留当前快照实际引用的 `eta_diverse_scene_test_v1/`（复放）、`cross_scene_homogeneity_audit_v1/`（哈希审计）、共享场景与策略库。

纠正依赖清单中的一处遗漏：快照 `cps_diag/load.py` 顶层读取 `diagnostics/hetero_scene_pipeline_v1/LABEL_TABLES.json`。因此该旧目录本轮整体保留。`cross_scene_pipeline_v2/` 也整体保留，以满足用户对依赖大目录先确认后瘦身的要求。

`gap_flow_scale_20261007/n20_cpu_priority_governor.py` 的长期调度进程仍在运行，其目录保留。用户明确指定的 17 个 multiprocessing 孤儿进程已核对 UID、PPID=1、命令及任务工作目录后发送 SIGTERM。

## 清理方案与备份边界

清理清单见同目录 `cleanup_hetero_set_20261027.json`。计划把 336 个无关实验目录、旧 3-D 实现、旧根目录结果及过时文档/分析入口从活动工作区移到：

`/home/zhihan/research/Basin_C1_retired_20261027/`

先将代码、文档和小型配置提交并推送，成功后再逐项同文件系统 rename，保留原相对结构和文件 inode，不在当前目录留下兼容链接。目录项逐个核对并记入外部归档的操作日志。归档与工作区使用同一块磁盘：这是移出活动目录，不是独立灾备，也不会释放同等磁盘空间。没有永久销毁历史数据、checkpoint、SQLite 或未判明研究证据。

Git 快照只包含版本化文件、源码、文档和不超过 2 MB 的选定 JSON 配置；大型配置、标签、checkpoint、轨迹、图片和数据库仍依赖本地原件/归档，不能宣称已由 Git 备份。原先指向其他存储盘的符号链接不会被解引用或删除其目标。历史数据库记录可能仍包含旧绝对路径；复用被归档的实验时，应按清单恢复对应原路径后再使用旧缓存。

当前快照仍含原 `.claude/jobs/` 绝对路径；这属于已有的运行边界，本次目录清理不改快照或控制逻辑，恢复运行时按其 README 重定位路径。

## 执行状态

清理前盘点与保留目录哈希已完成。Git 备份和目录移出待执行；完成后在此补充提交号与校验结果。

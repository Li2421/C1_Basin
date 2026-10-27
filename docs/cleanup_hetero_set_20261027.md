# Hetero set 工作区清理（2026-10-27）

**最新状态：用户随后明确要求删除 retired；该归档目录现已永久删除。下文“移出/归档”部分记录的是此前阶段，不能再作为数据恢复位置。**

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

清理清单见同目录 `cleanup_hetero_set_20261027.json`。已将 336 个无关实验目录、旧 3-D 实现、旧根目录结果及过时文档/分析入口从活动工作区移到：

`/home/zhihan/research/Basin_C1_retired_20261027/`

先将代码、文档和小型配置提交并推送，成功后再逐项同文件系统 rename，保留原相对结构和文件 inode，不在当前目录留下兼容链接。目录项逐个核对并记入外部归档的操作日志。归档与工作区使用同一块磁盘：这是移出活动目录，不是独立灾备，也不会释放同等磁盘空间。没有永久销毁历史数据、checkpoint、SQLite 或未判明研究证据。

Git 快照只包含版本化文件、源码、文档和不超过 2 MB 的选定 JSON 配置；大型配置、标签、checkpoint、轨迹、图片和数据库仍依赖本地原件/归档，不能宣称已由 Git 备份。原先指向其他存储盘的符号链接不会被解引用或删除其目标。历史数据库记录可能仍包含旧绝对路径；复用被归档的实验时，应按清单恢复对应原路径后再使用旧缓存。

当前快照仍含原 `.claude/jobs/` 绝对路径；这属于已有的运行边界，本次目录清理不改快照或控制逻辑，恢复运行时按其 README 重定位路径。

## 执行状态

- 清理前快照：`db959cb7f3c9fa60cdf9a22488fd5b8f70f016c1`，已推送至 `origin/research/ring-controller-generalization-20261014`，移出前核对远程分支提交一致。
- 已移出 537 个路径，其中 336 个为旧实验目录。每个路径的 device/inode/size 均与移出前一致；源路径已不存在。文件没有被复制、改写或永久删除。
- 原盘点 352 个实验目录中保留 16 个；清理期间另出现 `baseline_comparison_v1/`，不在计划中，原样保留，因此完成移动时实际有 17 个实验目录。
- 三个指定目录清理前后的 7,193 个目录项完全一致；全部普通文件 SHA256 相同，符号链接目标和目录项相同。文件内容总计 686,701,022 字节。
- 未执行训练、rollout 或评估。导入验证使用 `/tmp` 下的源码副本，不写受保护快照。
- 外部归档的 `cleanup_audit/` 中保存完整保留依据、移动日志、inode 清单和前后 SHA256；恢复时需确认原位置没有新文件，再恢复清单对应路径。

清理前后隔离导入检查均通过：18 个仓库模块解析到相同路径，两个冻结 JSON 的 SHA256 一致，critic 菜单形状为 `(49, 3)`。17 个指定孤儿进程复查均已退出。

移动后的磁盘占用：活动 `diagnostics/` 约 351 GiB，仓库外归档约 311 GiB（`du -sh`，不解引用符号链接）。文件仍位于同一磁盘，本轮不宣称释放这些空间。

## 后续：按用户要求永久删除 retired

用户明确要求“把 retired 删了吧”。已确认目标是普通目录、没有子挂载、没有进程将其用作工作目录；随后仅对 `/home/zhihan/research/Basin_C1_retired_20261027/` 执行递归删除，不跟随符号链接，也未删除其他归档位置。

删除命令成功返回，目标路径已不存在。三个 hetero set 结果目录仍存在，`FINAL_FROZEN.json` 和 `FROZEN_T2.json` 删除前后 SHA256 完全一致。共享数据库和当前运行依赖未改动。

本轮删除包含旧归档里的数据、checkpoint、图片、文字记录和本地完整清理审计。Git 中的 `db959cb` 保留已提交代码、文档和选定小型配置；未版本化数据不能通过该提交恢复。原先符号链接指向的其他磁盘文件不在本次删除范围内。

删除前后文件系统可用空间净增加 333,593,657,344 字节（约 310.7 GiB；并行任务可能影响该差值）。

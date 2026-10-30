# 2026-10-30 输出迁盘

用户授权迁移适合放数据盘的产物，并要求以后新增大型输出使用其他盘。
沿用 2026-10-25 的 SHA256 校验及原子目录/软链接交换流程；新增支持纯运行 checkpoint 的 .pkl 文件，只复制和哈希，不反序列化。

计划见 storage_migration_20261030.json。源目录只包含未版本化 JSON/NPZ/PKL，拒绝软链接、硬链接、近一小时修改及进程打开的目录；复制前后检查 inode/size/mtime/ctime/目录项，逐文件比对 SHA256，落盘后原子切换并移除已校验的 home 副本。

代码、环境、小型报告保持 home 路径。52 GiB 的 best_correction_search_v1 混有代码，因此只迁其中各纯 traces 子目录。当前 paper_baseline_protocol_v1、eta_diverse_scene_test_v1 的缓存及共享数据库仍被进程使用，留待任务结束后处理。

目标：/mnt/ssd/zhihan/Basin_C1/ 下相同相对路径。
审计：/mnt/ssd/zhihan/Basin_C1/audits/storage_20261030/，含 preflight.json、每目录 SHA256 清单、状态和最终 summary.json。未显示 complete 的目录不得视为已迁移。

后续实验遵守 AGENTS.md 的输出存储规则，启动前用 scripts/prepare_output_dir.py 为纯产物子目录创建 SSD 目标及兼容链接。旧脚本不会因环境变量自动改路径，必须检查其实际输出位置。数据盘未挂载时脚本报错，禁止回退 home。

恢复：先停止相关读写，将 SSD 内容复制到 home 同级临时目录，与审计 SHA256 清单逐文件核对后替换软链接；恢复验证前保留 SSD 副本。此次为迁移，不是额外独立备份。

## 完成验收

25 个目录、331,285 个文件，内容合计 513.49 GiB，全部逐文件 SHA256 一致后完成切换。最终核对全部目录项、大小及 25 个原路径链接，并通过 50 个原路径样本的哈希/NPZ 完整性检查。无遗留 migration-local-copy。

系统盘可用约 1030.3 GiB，较预检净增 511.3 GiB（同期任务写入会影响净值）；使用率从 86% 降至 71%。SSD 尚余约 2.8 TiB。详见审计目录 verification.json。

迁移脚本、计划及输出规则在开始搬迁前已提交推送：0564515、5291ea7，远程分支 maintenance/storage-20261030。原有在运行任务与数据库留在 home，本轮未停止任务。

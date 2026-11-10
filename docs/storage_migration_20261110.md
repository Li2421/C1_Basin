# 2026-11-10 系统盘产物迁移验收

用户授权尽可能迁移系统盘内容，以不影响实验为前提。本轮已完成并验收，通过原路径软链接继续访问 SSD 内容。

## 结果

- 共迁移 800 个纯产物/缓存目录，以及 SQLite 主库、WAL、SHM 三个文件；合计 **7,203,385 个文件、237.17 GiB 内容**。
- 系统盘可用空间从约 389 GiB 增至 **642.86 GiB**，净增 **254.24 GiB**；使用率由 89% 降至 82%。SSD 剩余约 1.71 TiB。
- 内容大小与实际释放空间不同：大量小文件存在文件系统分配开销，系统盘净变化也可能受到同期其他写入影响。

| 类别 | 目录数 | 文件数 | 内容 GiB |
|---|---:|---:|---:|
| 实验产物 | 602 | 4,550,620 | 194.58 |
| 共享日志 | 197 | 2,652,222 | 21.72 |
| pip 缓存 | 1 | 540 | 3.42 |
| SQLite 主库及 WAL/SHM | — | 3 | 17.45 |

## 位置与兼容

实验产物及日志目标为 `/mnt/ssd/zhihan/Basin_C1/` 下的相同相对路径。数据库目标为 `/mnt/ssd/zhihan/Basin_C1/shared_rollout_db/database/`；原 `shared_rollout_db/rollout.sqlite`、`-wal`、`-shm` 分别保留兼容链接。pip 目标为 `/mnt/ssd/zhihan/cache/pip`，原 `~/.cache/pip` 保留链接。原有 13 个日志软链接目标核对一致。

代码、Python 环境和含代码/配置/报告的混合目录留在 home，仅从中选择纯产物子目录。uv 缓存与环境存在硬链接，本轮保留；回收站也保留，避免改变桌面恢复行为。未整迁任何混合实验目录，未运行训练或 rollout。

## 校验与 SQLite

普通目录采用既有迁移流程：确认 SSD 挂载与空间、拒绝 Git 跟踪文件及打开中的目录、拒绝硬链接/内部软链接、要求文件至少一小时未修改；复制后逐文件 SHA256 比对，并检查目录项、类型、权限、大小、mtime 和源 inode/ctime 稳定性。落盘后原子交换目录与软链接，再移除已验证的 home 副本。

SQLite 在无打开进程、WAL 为零字节时连同 WAL/SHM 复制。三个文件复制时哈希一致；源库与目标库 quick_check 均通过，原路径只读连接正常，主库再次 SHA256 比对一致。SHM 为可变共享内存文件，只读连接可能更新它；复制前哈希与最终主库哈希分别留证。

最终验收核对全部 800 个目录的 complete 状态、兼容链接、逐文件校验清单数量，并对每个目录从原路径抽样重算一个文件的 SHA256；核对数据库三个链接、原有日志链接和迁移残留。全部通过。完整哈希比较在迁移时覆盖每个文件，最终抽样用于验证原路径可读性，不代替完整校验。

## 异常与续跑

第一次主任务在 602 个 diagnostics 目录完成后，因 journals 内已有软链接触发保护性停止，退出码为 1；未切换或删除 journals。随后保留已有链接，改为迁移 197 个普通日志子目录。复核 `shared_rollout_db/src/merger.py` 确认 `.merged` 是已导入的日志文件；`.merged`、`.quarantined` 和历史 `.tmp` 均按研究证据原样保留、复制校验，不重新合并。

修正计划与格式支持先提交推送，再续跑。已完成目录复用完成记录，未重复搬运。最终 results、pip、sqlite 三个任务退出码均为 0。

## 文档、审计与版本

- 迁移计划：`docs/storage_migration_20261110.json`、`docs/storage_migration_20261110_pip.json`。
- 小型验收摘要：`docs/storage_migration_20261110_verification.json`。
- 工具：`scripts/migrate_storage_20261110.py`、`scripts/migrate_sqlite_20261110.py`、`scripts/verify_storage_20261110.py`。
- 完整审计：`/mnt/ssd/zhihan/Basin_C1/audits/storage_20261110/`，包含逐目录 SHA256 清单、summary、verification、日志、PID、退出码和续跑记录；第一次结果保存在 `results.attempt1.*`。`journal_preflight.json` 是扩展日志后缀支持前的初步筛查，最终状态以 `results/summary.json` 和 `verification.json` 为准。
- 独立版本分支：`maintenance/storage-20261110`。迁移前提交并推送工具、计划和说明（de225b3、3a6a2cc）；日志修正先推送 bb4f6e7、7ebfca6 后执行。当前研究工作区的既有未提交修改未混入维护提交。

## 恢复与后续使用

恢复普通目录前先停止相关读写，确认 home 空间充足，把 SSD 内容复制到 home 同级临时目录，按 SHA256 清单核验后再替换原软链接。数据库恢复需要主库与 WAL/SHM 在所有读写进程退出后一起处理；恢复验收前保留 SSD 副本。

本次是存储迁移，不是独立备份。以后新增大型输出仍按 AGENTS.md 写入数据盘，启动前检查真实输出位置和挂载；挂载缺失时禁止回退写 home。

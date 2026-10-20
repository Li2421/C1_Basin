# 2026-10-20 HDD 分类迁移记录

迁移前代码、配置及文档已提交并推送：`55daa82`，分支
`research/ring-controller-generalization-20261014`。

归档根目录：`/mnt/hdd/zhihan/Basin_C1/archive_20261020/`。
原路径保留绝对符号链接；使用这些路径需要挂载 `/mnt/hdd`。
这是迁移，不是第二份独立备份。

| 分类 | 目录数量 | 文件字节数 |
|---|---:|---:|
| browser_previews | 4 | 63846838 |
| package_caches | 3 | 53801515 |
| python_caches | 465 | 101635260 |
| render_environments | 2 | 169200388 |

共迁移 474 个目录，388,484,001 字节文件内容（原目录磁盘占用约 399 MiB）。
每个目录复制后，比较源、目标所有文件的 SHA256、目录项和符号链接目标，
并再次检查源未变化，才移除源副本并建立链接。全部 474 个链接已验证可访问。
逐项路径与校验结果见归档根目录 `migration.jsonl`。

分类目录内部保留仓库相对路径，便于查找和恢复。恢复时先确认该路径仍为归档链接，
将目标完整复制到同文件系统临时目录并校验，再用该目录替换链接；完成验证前保留 HDD 副本。

## 保留项目

依据 `docs/workspace_cleanup_20261007.md`，旧实验存在交叉引用，不能按日期
认定为无用。本轮保留所有实验数据、checkpoint、拟合结果、预测、rollout、
SQLite 及 journals；数据库和正在运行的 CPU governor 不受迁移影响。
未启动任何训练、推理或 rollout。

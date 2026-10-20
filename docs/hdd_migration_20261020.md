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

## 初次迁移保留项目

依据 `docs/workspace_cleanup_20261007.md`，旧实验存在交叉引用，不能按日期
认定为无用。本轮保留所有实验数据、checkpoint、拟合结果、预测、rollout、
SQLite 及 journals；数据库和正在运行的 CPU governor 不受迁移影响。
未启动任何训练、推理或 rollout。

## 重复轨迹清理

后续经用户授权，18,590 个旧版本重复轨迹文件已替换为指向 N=50 v19 或
N=20 v7 保留版本的相对符号链接。重新校验 SHA256 后移除重复文件内容
28.31 GiB，估算释放文件块 28.35 GiB。manifest、报告和 checkpoint 保留。
操作与源目标映射记录：
`/mnt/hdd/zhihan/Basin_C1/audits/duplicate_removal_20261020.jsonl`。

## 跨类型2预测缓存迁移

状态：已获用户授权，准备执行；完成数量以归档目录 `summary.json` 为准。

源目录：
`/home/zhihan/research/Basin_C1/diagnostics/crosstype2_mixed_zoo_selector_v1/critic_fits/`。

归档目录：
`/mnt/hdd/zhihan/Basin_C1/archive_20261020/prediction_caches/crosstype2_mixed_zoo_selector_v1/`。

仅迁移 `分支/源集合/step*_predictions.npz`，预计 238,800 个文件、
84,873,221,270 字节（79.04 GiB）。归档目录 `files/` 内保留相同相对路径；
原文件位置建立绝对符号链接。checkpoint、训练输入、曲线和收据留在工作盘。

每个文件复制前计算源 SHA256，复制后校验目标且确认源未变化，再用链接
原子替换源副本。归档目录 `migration.jsonl` 记录源路径、目标路径、大小和
SHA256，`summary.json` 记录完成状态和链接验收；迁移程序为
`scripts/archive_critic_predictions.py`。源链接依赖 `/mnt/hdd` 挂载。

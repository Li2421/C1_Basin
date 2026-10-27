严格结束审计未通过，原 INTEGRITY_END.json/.receipt 保留。542326 个受保护文件中仅已有迁移说明内容变化，加 Git HEAD 变化，共两项差异。只读提交核对确认同一存储维护提交只给该文档追加28行；其他受保护内容没有报告差异。这不是自动豁免。实验、诊断、统计、数据库核查与图表均已完成；最终完成标志尚未生成。

原规范 docs/prompt_baseline_literature_scenes_distributed_v2_20261020.md 第零节第3条：“开始与结束必须一致”；第4条：“docs/ 中已有文件被修改或删除，仍按差异处理。”

提交：

0d102abf949801641aba498733a03d8255fe6c39 2026-10-21T10:26:12+08:00 Record home and data-disk layout audit and restored environment paths

变更：

```diff
diff --git a/docs/hdd_migration_20261020.md b/docs/hdd_migration_20261020.md
index 1c5e2be..6c11596 100644
--- a/docs/hdd_migration_20261020.md
+++ b/docs/hdd_migration_20261020.md
@@ -61,3 +61,31 @@ SHA256，`summary.json` 记录完成状态和链接验收；迁移程序为
 8 个模型分支分别通过原路径 `np.load` 读取，且与原 `COMPLETE.receipt.json`
 的 SHA256 一致；冻结配置中引用的 90 个 checkpoint 均仍为工作盘上的普通
 文件，SHA256 与冻结配置一致。验收记录为归档目录 `verification.json`。
+
+## 2026年10月21日存储布局核查
+
+按用户转述的老师要求，代码和环境应留在 home，数据使用独立数据盘。
+机器有 `/mnt/ssd`（可用约 3.5 TiB）和 `/mnt/hdd`（可用约 18 TiB）。
+当前仓库在 home 占约 522 GiB，其中 `diagnostics/` 约 498 GiB，
+`shared_rollout_db/` 约 24 GiB，`datasets/` 约 1.2 GiB；数字为核查时快照。
+
+建议的后续布局如下，SSD 数据目录尚未切换；老师如有指定路径，以指定路径为准。
+
+| 内容 | 位置 |
+|---|---|
+| Git 代码、脚本、配置和文档 | `/home/zhihan/research/Basin_C1/` |
+| Python 环境及安装依赖 | 保留在现有 home 路径 |
+| 活跃数据、checkpoint、rollout 与数据库 | `/mnt/ssd/zhihan/Basin_C1/` |
+| 历史预测缓存、旧实验归档 | `/mnt/hdd/zhihan/Basin_C1/` |
+
+已经完成的调整：将 `diagnostics/axis4a_addendum_v1/plot_env` 和
+`diagnostics/controller_loco_real_flow_v1/plot_vendor` 从 HDD 链接恢复为 home
+中的实际目录，共 169,200,388 字节。源目标文件 SHA256、目录项及链接目标
+核对一致，HDD 原副本保留。验收记录：
+`/mnt/hdd/zhihan/Basin_C1/audits/environment_home_restore_20261021.json`。
+因此，初次迁移中的 `render_environments` 两项现在是保留的历史副本。
+
+核查时 `baseline_literature_scenes_distributed_v2` 的任务仍在运行，SQLite
+主库及 WAL 也在写入。本次没有移动活跃数据或数据库。后续迁移须先分离
+`diagnostics/` 中混放的代码与产物，保留原路径兼容链接；数据库须在所有
+读写进程退出后统一切换数据库位置及 WAL/SHM 路径，不能在线逐文件移动。
```

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，以及 round29 的 `diagnostics/nominal_safety_baselines_v1/HANDOFF.md` 与 `MERGE_RECEIPT.json`。严格遵守 AGENTS.md。

# 一、目的：把 round29 的记录并入全局共享 DB

round29（`diagnostics/nominal_safety_baselines_v1/`）的 prompt 只允许在新目录写文件。因此它把全局共享 DB 复制到本目录（`rollout.sqlite`，约 6.6 GB），在副本上完成了 preflight、journal、单进程 merge 与 postflight。全局 DB `shared_rollout_db/rollout.sqlite` 里没有它的 2,048 条 rollout（experiment_uid = `exp_nominal_safety_baselines_v1`）。

本任务只做一件事：把这 2,048 条记录，连同它们依赖的登记信息（experiment、controller_config、eta、state 等），原样并入全局 DB。不新增、不重跑任何 rollout。

# 二、要求

1. **只用官方入口写全局 DB。**
   - rollout 记录必须通过 `shared_rollout_db/src/` 中的官方 ingest 逻辑（`historical_ingest.Ingestor`，单进程）写入，不得手写绕过校验的 SQL。
   - 登记行（experiment、controller_config、eta、state 等）如果官方入口不能自动创建，就从 round29 的副本 DB 中原样复制，用 INSERT OR IGNORE。
   - 对全局 DB 中已存在的同 uid 登记行，逐字段核对；如有不一致，停下报告，不得覆盖。
2. **journal 来源。**
   - round29 的 journal 在 `diagnostics/nominal_safety_baselines_v1/journals/exp_nominal_safety_baselines_v1/`，文件已被标成 `.merged`。
   - 不得修改或重命名这些文件。如果官方入口需要 `.jsonl` 后缀，就复制到本任务的新目录后再交给它，并记录每个文件的 sha256，确认与原文件一致。
3. **写入前（preflight）：**
   - 确认当前没有其他 merger 或写 DB 的任务在运行，保证单进程写入；
   - 记录全局 DB 各表的行数；
   - 确认 2,048 个 rollout_uid 在全局 DB 中都不存在。若有已存在的，核对 raw_record_hash：一致就跳过；不一致就停下报告。
4. **写入后（postflight）：**
   - `exp_nominal_safety_baselines_v1` 恰好 2,048 条；
   - 每一条的结果字段与 raw_record_hash，都与 round29 副本 DB 中的对应行逐字段一致；
   - 其他 experiment 的行数与写入前相同；
   - unknown 数与 round29 报告一致（0）。
5. **不删除副本。** round29 目录下的 `rollout.sqlite` 保留不动。在报告中写明它的路径与大小，并说明：合并核验通过后可以删除，但删除需要用户确认（AGENTS.md：SQLite 等大型数据先保留）。

# 三、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/nominal_safety_baselines_v1_ingest/` 中写文件。全局 DB 只能通过官方入口追加写入，这是唯一的例外。不修改、不删除 round29 目录与其他历史资产，不做 git commit。
- **源码保护：** 不得修改、删除、移动或重新格式化任何已有源码，包括 `shared_rollout_db/src/` 中的 merger 与 ingest 代码。需要不同行为时，在新目录中写新脚本，只读 import 官方模块；如果只有改源码才能继续，就停下报告。git 只允许只读命令（status、diff、log、rev-parse、ls-files），严禁 commit、stash、checkout、reset、restore、clean、add 等任何会改变工作区或历史的命令。开始与结束时各记录一次 `git status --porcelain` 与 `git diff` 的 sha256，二者必须一致。
- **不新增 rollout，不重算任何结果。** 本任务不涉及仿真，不需要视频。
- **资源：** 单进程，1 个 CPU。本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 四、交付

全部交付物放在 `diagnostics/nominal_safety_baselines_v1_ingest/`：
- PREFLIGHT.json：写入前各表行数、uid 冲突检查；
- 复制的 journal 及 sha256 对照表；
- POSTFLIGHT.json：写入后各表行数、逐条核对结果；
- INGEST_REPORT.md：第一段直接说明是否已完整并入、核对是否全部通过，以及副本是否可以删除。

# Round30 交接

工作根目录 `/home/zhihan/research/Basin_C1`。任务来源 `docs/prompt_round30_teacher_student_macflow_20261015.md`，现处于 BLOCKED_REQUIRED_INTERFACE_MISSING；没有后台作业或待合并 journal。

最高约束：所有新代码和输出只可写 `diagnostics/teacher_student_macflow_v1/`；已有源码及官方仓库、原 Python 环境只读；git 只读；共享 DB 仅允许官方 merger 追加。不得通过 monkey patch 注入噪声。

先读 REPORT.md、INTERFACE_AUDIT.json 和 HISTORY_AUDIT.md。round28 教师采样器仅接收 seed/key，没有显式 ξ 接口。任务同时要求原 runtime 代码路径和显式 ξ，并要求缺接口时停止。尚未获准变通。不能直接开始训练或复制 sampler 后宣称满足原路径要求。

已完成：起止 git 完整性审计通过；14 教师 checkpoint hash 校验通过；round28 冻结文件校验；相关旧学生/Stage-I 代码检索；静态接口审计。脚本 audit.py 只用标准库，不导入已有项目代码。重跑会覆盖本轮结束审计产物，应先保留原收据；INTEGRITY_START.json 不由该脚本覆盖。

若接口问题得到解决：先补历史训练数据/dev 结构审计、pilot seed 与 DB/round28 保留区间去重；写完整预注册并封存 UTC/hash，明确全部训练参数和 14 个 seed；完成学生训练与 dev checkpoint 选择；能力门；无标签 R5 和 A3/A0/G_E 选择封存；评估身份与 manifest/preflight 必须早于提交；只执行缺失；journal 后官方单进程 merger/postflight。始终恢复并记录 JAX x64=False；资源合计至多 18 CPU shard/2 GPU shard。TS1–TS3 只能由新学生证据判定。专家不运行。unknown 不重跑；timeout 不称 deadlock。

本次没有预注册或实验数据；results.json 的 NOT_RUN 状态不能作为统计结论。完整交付尚缺训练、测试、能力门、探测、DB 流程、评估、统计表与耗时，详见 REPORT.md。

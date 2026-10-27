# 历史证据检查

v2 已封版：`../baseline_literature_scenes_distributed_v2/V2_COMPLETE.json`。
已核查第二阶段 DYNAMICS_IMPLEMENTATION、DYNAMICS_DIAGNOSTICS、PREREGISTRATION、CORRECTION_REVISION、HANDOFF、DYNAMICS_STAGE；原始 raw 的逐做法碰撞数、unknown、步数与运行耗时重新计算于 HISTORY_RAW_COUNTS.json，输入 sha256 逐条记在 EVIDENCE_INPUTS.jsonl。

窄通道、二维拥挤随机、圆形围场安全层发生改变，原第二阶段标签不能替代修复后的标签；计划分别 22528、39424、5760，共 67712 个请求。新身份须通过官方注册及 preflight，仅执行 genuinely_missing。单元门和试跑门是前置条件，不满足时不运行全量。

门口第二阶段 23040 请求和全部第一阶段：执行语义未变，按 v2 原始 state/controller/eta/seed 身份直接引用，不重跑。原 raw、trace、checkpoint、预测、配置与报告保留原地只读。无需 critic 训练，无需新 rollout 来计算旧结果的碰撞率。

gamma 不按研究结果调参；网格、头对头初始条件及随机快照在单元门中冻结。碰撞、约束无解与数值 unknown 分别处理；timeout 不等于死锁。

# Round30 续跑交接

任务文件：`/home/zhihan/research/Basin_C1/docs/prompt_round30_teacher_student_macflow_20261015.md`（当前版本hash见PREREGISTRATION.receipt.json）。工作根目录Basin_C1；Python为 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。全部新代码/输出仅本目录；已有源码、PoC、官方仓库、原环境只读，git只读。此次没有后台作业、DB写入或待合并journal。

第一次产物已原样移到attempt1_blocked/并验证hash。当前状态：**BLOCKED_NOISE_CONSISTENCY_GATE_FAILED**，不是旧版接口缺失。新版接口native_fn/trace_fn已使用，14教师checkpoint和实际训练manifest hash均匹配。

先看REPORT.md与NOISE_GATE.json：预注册计划1,008样本，首个wide教师train_nominal_000000 transition0在Runtime.nominal(traced=True)内部assert_allclose失败，最大绝对差9.8882097e-09，原容差rtol1e-10/atol1e-11。独立初始noise逐位一致、native与非traced nominal经相同变换后的动作逐位一致，仅为这个失败样本的诊断。原断言未改；已按任务“任何一项不通过就停下报告”停止。

证据：NOISE_GATE_FAILURE.npz、NOISE_GATE.json、NOISE_GATE.receipt.json、noise_gate.log；开始/结束完整性审计及PoC所有.py hash；GATE_IMPORTED_SOURCES.json和结束复核；TRAINING_CONFIG.json/TEACHERS.json/PREREGISTRATION.md及收据。历史检索见HISTORY_AUDIT.json，原文件兼容性描述见attempt1_blocked/HISTORY_AUDIT.md。

代码：core.py为本目录工具，不修改已有模块；prepare.py只读匹配训练数据与DB历史元数据；noise_gate.py实施先决门并保护已有gate证据不被覆盖；finish_audit.py只用标准库做结束审计。不得直接重跑noise_gate或覆盖收据。新尝试如获授权，应先保留本次证据并重新开始完整性审计；不得自行放宽原门容差或修改既有文件。

尚缺：学生训练代码/训练/选择与离线测试、pilot去重与能力门、R5、冻结A3/A0/G_E选择、rollout preflight及缺失任务、默认journal与官方单进程merger/postflight、统计与论文表、推理耗时。TS1–TS5全部NOT_RUN。没有“学生能力不足”的证据，不能将门失败写成能力门INCONCLUSIVE。

后续需先解决任务要求与原traced断言容差之间的前置门问题。没有授权时维持停止。若任务有明确更新，按新文档重新冻结流程；保持原teacher sampling和D-local实现、精度、物理、事件及成功标准。教师raw实际chart依其frame而定，global教师为world、local教师为radial/tangential；执行经原to_world。所有新闭环运行必须先预注册及官方preflight，再提交。资源合计上限18CPU shard/2GPU shard，专家仅引用旧结果，unknown不重跑、timeout不称deadlock。

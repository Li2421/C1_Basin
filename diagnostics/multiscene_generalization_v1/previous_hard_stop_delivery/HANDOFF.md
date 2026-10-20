任务：执行 docs/prompt_round39_multiscene_generator_cross_scene_20261017.md 及第二节用户补充决定。工作根目录 /home/zhihan/research/Basin_C1，Python /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。所有新增文件仅 diagnostics/multiscene_generalization_v1/。当前硬停止，不依赖本对话可按以下证据理解状态。

严格起终完整性审计未完全通过：其他目录出现新增或改动，差异已列出；本轮导入和复制的既有源码未变，恢复阶段仓库状态与差异哈希一致。历史执行链十六条记录的首动作、完整位置轨迹、终止类型与步数逐位复现通过；单机器人等价性测试因我设置的十字路口撞墙样例目标位于障碍内而在初始化时中断，未观察到轨迹不一致，按测试门未通过即停止的要求停下整轮。多场景终点一至四均未评估；单门隔墙、串联瓶颈走廊、多门隔墙、十字路口、环岛、房间与门及随机障碍场的能力门均未运行，修正作用与动作偏好未知；训练族数为零、一、三、五时的规模曲线没有遗憾结果；随机障碍场锁定测试未打开，评估零次。

本报告是硬停止状态交付，不是完整实验交付。测试设置错误属于本轮实现错误，不能解释为单机器人环境与原环境轨迹不一致，也不能解释为某族导航能力失败。测试在十字路口撞墙样例的 GeneratedEnv 初始化处抛出 invalid start/goal geometry：原目标为横坐标正负六米、纵坐标一米，位于角障碍内。未执行该样例的物理比较，也未执行后续环岛样例，因此缺少至少三个族的完整等价性证书。四机器人单门等价性检查在此之前已执行到单机器人测试入口，但整套门没有通过收据。

补充决定已重新阅读，并写入 PREREGISTRATION_AMENDMENT_20261017.md，附 UTC、sha256 与授权 prompt 的 sha256。SingleAgentGeneratedEnv 仅用于示范与导航能力门，原 step 的差异仅删除机器人对碰撞及相关字段；stalled_groups 返回空列表。四机器人 GeneratedEnv 的 step、reset、snapshot 均直接继承原 BottleneckEnv。原函数 diff 与原文件 sha256 分别见 SINGLE_AGENT_STEP.diff、SINGLE_AGENT_COPY_PROVENANCE.json。合法碰撞目标的最小测试修正提案已保存为 TEST_FIXTURE_CORRECTION_PROPOSED.diff，尚未应用或重跑。

共享数据库：本轮登记只追加 experiment 身份，原登记行哈希前后一致；十六条真正缺失的确定性潜势控制器验证 seed 使用官方 future_index=3900，控制器忽略随机键，执行语义与历史参考一致。执行后默认官方 journal 十六个、单进程官方 merger、postflight 精确复用十六条，缺失零、unknown 零。参照的是机器人数量泛化轮在双机器人单门上的十六个先导状态与只加安全投影，未声称覆盖历史流匹配策略的所有精度和观察接口。初版 preflight 的扩展 seed_key 与官方入库键不兼容，已在任何执行之前改为合法 future_index；初版未产生 journal 或 rollout，遗留计划原样保留，见 ABANDONED_REPRODUCTION_PREFLIGHT.json。任何旧 unknown 均未重跑。

资源：等待其他任务释放 CPU 后才提交本轮一个 CPU shard；提交前队列为空，本轮没有 GPU 作业。开工、等待、提交与结束的队列已记录。没有 Git 写操作、包安装、读取禁止目录或改动任何现有研究源码。完整性全树快照另观察到其他目录的源码新增或改动；原始开工与恢复阶段的差异分别列于 INTEGRITY_FINAL_COMPARISON.json，未归因于本轮，也未恢复这些文件。

仅完成并验证了历史复现；geometry.py、single_agent.py、test_geometry.py 是当前测试门实现，其余 catalog、观测、专家、训练及 worker 代码只是尚未执行和验证的准备代码，不能直接当作已完成管线。导航能力门与多机器人能力门未运行，不能记为“未通过”；规模曲线为空，不能记为零遗憾；锁定测试仍关闭。死锁修复指标仍是成功率相对只加安全投影的提升，timeout 不称 deadlock。

恢复顺序：先解决合法等价性样例并获得完整门证书；再有效性检查与几何清单、示范、训练、导航能力门、先导能力门、探测、前六族选择冻结与分析，最后最终模型冻结及锁定测试。后续所有依赖代码和统计都仍须验证，目标标签不得提前打开。随机障碍场的独立训练来源另有设计缺口，具体见 GAPS.json；用前六族示范合并训练四个策略的方案尚未获答复。

本轮执行 docs/prompt_round30e_decentralized_blackbox_planners_20261016.md。全部新代码与产物在 diagnostics/decentralized_blackbox_planners_v1，唯一目录外写入为官方共享数据库登记、journal和官方merger。不要重启coordinator.py；所有unknown与轨迹保留，禁止重新运行科学任务或覆盖证据。

阅读 REPORT.md 首段与逐族终点，然后阅读 results.json。主论文表 PAPER_TABLE.md、PAPER_TABLE.csv；三个族独立表与合并DEVELOPMENT表另存。CAPABILITY_GATE.json 有能力不足和无需修正两类排除，RHR_GRID_RESULTS.json 有完整24候选、RHR_SELECTED.json 为每族预先冻结选择。FROZEN_CHOICES.json、CONSTANT_CHOICES.json 及收据证明先于pilot/tune/main；无训练。

pipeline.py、episode_frozen.py、fair_rhr.py 从round30d逐字复制；controllers.py、legacy_controllers.py、gaussian.py 从round33逐字复制；metrics.py、stats.py 从round30g复制，仅支持模块导入路径改变。来源与diff见 COPY_PROVENANCE.json、VENDOR_PROVENANCE.json。A*实现 astar.py 是本轮新写。分布式模型预测控制需要其余机器人的位置和速度；高斯行为克隆需要四机器人联合观测；A*是自身/局部邻居接口。不得统称三族都不通信。

数据库登记行在registration/，每个phase都有protocol、CASES、MISSING_CASES、PLANNED_ROLLOUTS、官方CACHE_PREFLIGHT/POSTFLIGHT（探测的POSTFLIGHT报告无任务标签）、MERGE_RECEIPT、SUBMISSION。原始轨迹在probe/pilot/tune/main/raw；32条校验重放另存，均逐位一致。不要把timeout称deadlock。EPISODE_METRICS.jsonl、MECHANISMS.json复用round30g机制定义；各族_FAILURE_COMPOSITION.json与FAILURE_RESCUES.csv含来源和救回计数。

完整性审计、导入来源及冻结时序最终核验见 INTEGRITY_COMPARISON.json、IMPORTED_SOURCE_END.json、FINAL_VERIFICATION.json。DELIVERY.receipt.json 是最终交付有效性依据；PREREGISTRATION_initial.receipt.json 是执行前路径适配的历史收据，当前收据为PREREGISTRATION.receipt.json。所有新族终点CONFIRMATORY，历史学生和合并分析DEVELOPMENT；GAPS.json逐项列出未知或能力缺口。没有修改旧源码、重训、包安装、git写操作或视频任务。

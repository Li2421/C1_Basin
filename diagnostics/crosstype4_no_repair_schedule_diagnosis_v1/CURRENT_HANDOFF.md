# 当前运行交接

跨类型4全部为开发证据。最新prompt已完整阅读；开始审计和历史核查封存。16条复现、8条前缀/进度等价、32条提前结束验证均通过，32条覆盖全部19类型。

A Slurm数组26273正在运行，真实补跑上限28222；已直接复用13718个DB请求、22个提前结束等价请求。A不重训、不重选旧η对。实时进度见MONITOR_STATUS.json。

独立协调器PID 2496733 已启动，持有COORDINATOR.lock，每30秒检查；顺序为A收集/5000次bootstrap诊断，再B训练标签精确preflight与补跑，191源集合×2分支GPU训练、源内选择、逐位冻结推理检查、19折模型/最终模型冻结、新24状态选择密封、最后B评估/统计/结束审计/交付。coordinator.py和全部后续实现已准备；B尚未运行，不声称有新成功率结果。

初估全轮7.3–12.7小时，见OPENING_TOTAL_RUNTIME_SUPPLEMENT.md。当前prompt资源上限18CPU/2GPU，提交前全用户squeue检查。不得采用先前跨类型3的旧资源上限或新实验身份。

所有新代码/输出只在本轮目录；共享DB仅官方设计登记/journal/merger。旧目录与源码只读，禁止目录按前缀剪枝；不做Git写操作、不装包。

冻结收据不可覆盖。A原worker已逐位恢复到原收据SHA；动态源复用在单独rollout_worker_v2.py及A_EXECUTION_FROZEN_v2.receipt.json中，原收据文件哈希全部一致。B使用单独rollout_worker_v3.py先执行零→零实际源。原先协调器缺导入的启动错误已修复，旧失败保留在MONITOR_FAILURE_2496616.json，MONITOR_STARTUP_RECOVERY.json说明恢复。

13项critic/编码/源隔离检查与8项统计检查通过。B GPU训练和冻结推理尚待真实数据验证，若协调器异常查MONITOR_FAILURE.json及对应Slurm日志，不自动重跑unknown或未完成claim。科学结果报告只在各阶段完整后生成。

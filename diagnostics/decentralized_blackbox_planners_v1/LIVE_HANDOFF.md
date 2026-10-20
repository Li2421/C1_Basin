本轮为 docs/prompt_round30e_decentralized_blackbox_planners_20261016.md；仅本目录新文件可写，官方共享数据库追加流程为唯一例外。当前运行中的协调器见 COORDINATOR_PID.json、coordinator.log、RUN_STATUS.json。不要重新启动协调器或重新提交任何科学任务；先检查是否仍存活以及各阶段已有raw、journal、收据。

链路：32条历史逐位复现 → 576条无标签探测 → 冻结critic/常数选择 → 576条能力门 → 6912条公平右手规则调参 → 冻结逐族规则 → 15实例×32state×2seed×7臂主评估6720条 → 固定统计/机制分析 → 保护完整性结束审计 → delivery.py生成报告和最终收据。A*排除3个两臂先导成功率均100%的实例，余3个使该族INCONCLUSIVE；其余两族各6实例。

所有任务用指定 .venv-c1/bin/python -B，CPU backend，显式关闭JAX x64。无训练。资源上限全部用户18 CPU shard、2 GPU shard，本轮无GPU作业。resource_watch.py记录调参及后续占用。每次提交保存squeue，预留pending数组元素，worker开始再查运行总CPU。unknown保留不重跑，任何.partial trace不得覆盖。

预注册在任何新轨迹前封存。最初debug分区复现作业22825因时限停留PENDING，取消后用cpu分区22826完成；不是科学重跑。机制代码仅导入路径适配的收据补充见PREREGISTRATION_PATH_AMENDMENT.json，原收据保留。pipeline/episode/fair_rhr逐字复制round30d，controllers/legacy_controllers/gaussian逐字复制round33；COPY_PROVENANCE、VENDOR_PROVENANCE、COPY_DIFF记录来源。机制metrics/stats从round30g复制，只有支持模块路径变动。

完成以DELIVERY.receipt.json与FINAL_VERIFICATION.json为准。若COORDINATOR_ERROR.json出现，只补尚未执行的工程/分析步骤；不得为了工程恢复重跑既有unknown或完整轨迹。任何已有源码内容或数据库路径规则失败必须停止。科学阶段pilot/tune/main只允许官方登记、提交前preflight、missing任务、默认journal、单进程官方merger、postflight。若尚未生成REPORT/HANDOFF，说明尚未完成，不得声称完成。

# Round30b handoff

报告：REPORT.md；总结果：results.json；主表：paper_table_confirmatory.md；论文development表：paper_table_development.md。2/10学生通过蒸馏门。主分析TS1=INCONCLUSIVE，TS2=INCONCLUSIVE，TS3a=INCONCLUSIVE，TS3b=INCONCLUSIVE，TS6=INCONCLUSIVE。合并TS1=INCONCLUSIVE，TS2=INCONCLUSIVE，TS3a=INCONCLUSIVE，TS3b=INCONCLUSIVE，TS6=INCONCLUSIVE。

根目录/home/zhihan/research/Basin_C1，本轮唯一工作目录diagnostics/teacher_student_macflow_v1b。Python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；绘图用同目录.venv-c1-plots。不要重训、重跑unknown或更改已有文件。原prompt为docs/prompt_round30b_frame_consistent_students_20261015.md。代码均本轮新增/复制；不得改round30、PoC或官方仓库。

执行链：integrity.py start → prepare.py（hash历史证据/预注册/噪声门复用）→ numeric.sbatch（equivariance→train→offline门）+replay.sbatch（16逐位门/组件）→benchmark.sbatch→stage.py pilot→stage.py probe（官方pilot merger/postflight及能力门）→stage.py mainphase（R5/冻结选择/主评估preflight）→postflight.py main→analysis.py→plots.py→validate.py→integrity.py end→deliver.py。automate.py是一次性顺序协调器，状态在AUTOMATION_STATUS.json、automation.log；已完成阶段收据不可覆盖，不要重复启动。每个入口core先断言DB包和默认journal路径。运行全过程恢复JAX x64=False。代码不monkeypatch已有模块。

固定资源合计20CPU/2GPU shards；提交前squeue查看用户所有任务，数组throttle只使用剩余额度。RESOURCE_MONITOR.json保留记录。协调器在scontrol提高并发时因已完成子任务返回非零而停止；Slurm已实际更新剩余子任务，所有trial原样继续，没有重复提交或重跑。原日志/STOPPED状态保存在AUTOMATION_INTERRUPTION.json及AUTOMATION_STATUS.json；finalize_existing.py只接续已有批次的合并、分析、审计和交付，最终状态以FINAL_STATUS.json为准。两阶段标签写入Basin_C1/shared_rollout_db/journals并官方单进程merger，DB原有登记行hash一致。旧local学生索引8/10/12/13只读复用，旧global结果已被取代且不删除。unknown保留不重跑；timeout不是deadlock。

完整性：通过。详见INTEGRITY_COMPARISON.json、IMPORTED_SOURCE_AUDIT.json及DB_PATH_AUDIT.json。所有结果统计与表格可从冻结JSON重新计算，无需rollout；重新分析必须输出新文件名以保留收据。

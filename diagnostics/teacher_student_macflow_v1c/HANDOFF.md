# Round30c handoff

任务已完成：8名候选中8名通过修正门；CONFIRMATORY=10名，SENSITIVITY=8名，DEVELOPMENT=14名。TS1=PASS，TS2=NOT_PASS，TS3a=PASS，TS3b=INCONCLUSIVE，TS6=PASS，TS6b=PASS。tuned全局参数speed=0.1m/s、trigger=20steps。报告第一段及三张paper_table_*包含完整结论和CI。

工作根目录/home/zhihan/research/Basin_C1，唯一新增目录diagnostics/teacher_student_macflow_v1c。主环境/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；绘图同目录.venv-c1-plots。输入任务docs/prompt_round30c_regate_and_tuned_rhr_20261015.md；PREREGISTRATION.md及收据在任何新指标前封存。没有训练或换checkpoint，STUDENTS.json指向round30b10个global与round30旧4个local，均核hash。

链：integrity.py start→prepare.py→Slurm regate.py→tests.py及planning.py initial/pilot→orchestrate.py：pilot官方合并/能力门→R5/freeze_choices→12点tune官方合并→freeze_rhr→main preflight和Slurm缺失任务→main官方合并→analysis.py→finish.py（教师补表、网格图、验证、完整性结束、报告）。阶段和作业ID见RUN_STATUS.json/*_SUBMISSION.json/orchestration.log。已完成，不要重复启动协调器/训练/rollout，unknown不得重跑。要重算统计，使用analysis_arrays.npz和results.json，输出新文件名保留收据。

新8人在修正门前没有闭环标签，主分析的旧2人及DEVELOPMENT旧4人的标签此前打开过，研究顺序如实披露。三套主表与6个终点、alwaysS500−A3、失败/救回表都在results.json及REPORT.md。RHR网格是全体有效学生等权平均选一个全局参数，不是按学生分别选；η没有在学生上调参。REGATE的.05/.20阈值只作敏感性，旧6人判定不变。数据与教师标签均复用，缓存manifest/pre/postflight和登记行审计全留存。所有源码和环境保护规则继续适用，只写本目录；DB官方流程是唯一外部写例外。资源全用户20CPU/2GPU shard，提交前查squeue。

审计：protected_content_pass=True，DB路径断言通过；复现门引用及tuned原参数点逐位测试通过。COPY_DIFF.patch、IMPORTED_SOURCE_AUDIT、DB_PATH_AUDIT、INTEGRITY_COMPARISON与RESULTS_VALIDATION为交付证据。旧30/30b文件从未修改。timeout不是deadlock，unknown不是已知失败。没有视频交付。

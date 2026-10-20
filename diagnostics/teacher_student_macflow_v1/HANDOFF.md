# Round30 最终交接

任务根目录 /home/zhihan/research/Basin_C1；任务源 docs/prompt_round30_teacher_student_macflow_20261015.md。先读REPORT.md与results.json。本轮数值实验/统计交付已完成，但严格完整性/写入范围审计未通过，不能声称完全合规：pilot有runner哈希衔接偏差，且448份journal误写PoC副本；并行任务还新增了Git状态条目。原源码内容hash检查结果见INTEGRITY_COMPARISON.json。

正式结果、成功率/CI、TS1–TS5、转移/继承/耗时见results.json、paper_table_complete.md/CSV（原封存表仍在paper_table.md）、transition_table.md、inheritance_table.md、inference_timing_table.md。6个学生的dev最优为step0，详见DISTILLATION_REPORT.md，不能把它们都当作成功蒸馏。训练设置、选择与结论未按闭环调整。

旧三次产物保留在attempt1_blocked/、attempt2_gate_tolerance/、attempt3_db_registration/。第三次1008样本噪声门直接复用；第四次新增DB元数据登记授权与复制实现/16条复现门。原预注册及第一次补充字节不变，第四次补充及UTC/hash收据另存。

复制core.py/pipeline.py保持原文件字节，episode_frozen.py函数体与round28一致。episode_extended.py通过独立runtime.action包装接入nominal-only和1.5m RHR，冻结循环不改。学生私有noise=fold_in(key_for(state,seed,step),i)。所有JAX导入后恢复x64=False。源码、checkpoint、参数、物理/事件/成功标准保持冻结。

官方DB元数据由registration.py只追加INSERT OR IGNORE并保存前后原行hash；标签由默认官方append_journal及python -m shared_rollout_db.src.merger入库。先导入并校验Basin_C1的shared_rollout_db，再加载会插入PoC/source路径的runtime，禁止改cache_writer.ROOT或模块全局。不要重复pilot误导入问题。越界PoC journal路径见JOURNAL_PATH_DEVIATION.json；文件原样保留，本轮没有授权删除PoC目录文件，后续不得擅自清理。

新task rollout：pilot448、main6272、teacher1792，总8512；另448个无标签probe和16个明确要求的历史一致性重放，不计为新任务标签。辅助worker仅处理原main尾部缺失，原worker跳过已有raw；最终唯一性见ROLLOUT_DEDUP_AUDIT.json。unknown保留不重跑，timeout/停滞不称deadlock。

main/raw与traces、teacher/raw与traces、pilot/raw与traces、probe/raw、data与students均保留。各阶段manifest/preflight/收据早于提交；学生checkpoint与选择在主评估标签前封存。若需重新统计，复用这些标签，不再训练或生成rollout；在本目录新子目录保存新的分析，不覆盖任何原始证据/收据。

原完整报告和原始JSON均为最终证据。ANALYSIS_COMPLETE.receipt.json封存统计结果；DELIVERY.receipt.json封存交付文件；FINAL_DB_AUDIT.json和RESULTS_VALIDATION.json核对旧行、标签、去重及历史参考。不要运行旧attempt中的脚本或round27含全局替换的参考脚本。

本轮提交的任务已结束，不依赖继续运行的作业；squeue中可能仍有其他用户研究任务，非本轮交付依赖。资源授权按最新用户指示，当前全用户合计18CPU shard、2GPU shard。没有安装包、改原环境、git写命令、专家新规划或视频交付。任何后续源码/状态差异都应留存并解释。

本轮已完成分布式学生线的只读机制收尾。先读REPORT.md、CLOSEOUT_FINDINGS.json及PAPER_TABLE.csv。任务源为docs/prompt_round30g_student_line_mechanism_closeout_20261017.md；预注册和UTC/SHA256收据在本目录。全部DEVELOPMENT，不训练、不执行环境、不查询模型、不写共享DB；DB从未打开。

输入历史目录：decentralized_pipeline_v1、teacher_student_macflow_v1、teacher_student_macflow_v1b、teacher_student_macflow_v1c、teacher_student_macflow_v1d、teacher_student_macflow_v1f。开始/结束156,997文件的完整hash清单及Git状态、diff hash在INTEGRITY_START/END.json；比较见INTEGRITY_COMPARISON.json。并行任务新增git条目单列，非本轮所写。所有新增源码和输出限本目录，原目录不改。

核心入口：prepare.py只解析和配对已有标签/轨迹；metrics.py是纯轨迹函数；worker.py离线提取；analyze.py/stats.py做七项分簇统计；supplement.py导出到达分布、同碰撞时刻位置和逐教师配对差；paired_diagnostics.py做配对早期指标；transition_tables.py导出全模式×结果转移；reporting.py写七张表和论文表；plots.py画8组PNG/PDF；finalize.py封存交付。没有任何rollout或训练入口。不得重新执行历史runner作为分析捷径。

29,024条角色记录对应28,128条唯一轨迹，联合教师两行的只跑臂复用同一896条记录。MANIFEST.json逐行列出原始raw/npz路径、sha256、controller/state/seed与来源。features/shard*.jsonl保存唯一UID的全部逐局指标；EPISODE_METRICS.csv已展开为角色。分析1至7分别为analysis1_*至analysis7_*.json，results.json包含分析结果及链接；ANALYSIS_COMPLETE.receipt.json锁定原始聚合结果。最终综合判断在CLOSEOUT_FINDINGS.json，最终交付收据为DELIVERY.receipt.json。

重要结论：教师只跑碰撞59.0%，先前29.5%是与安全臂混合后的口径；名义失败大多已同向；安全超时多有运动和无进展，严格低速停滞少；修正提升早期目标进展，但不普遍提升早期同向率；零动作也能同向却推进不足；共享噪声减少碰撞、增加超时，但方向不一致率下降CI跨零，同等超时观察长度的运动无进展差CI也跨零。不能把timeout写成deadlock，不能把教师相关性写成数学上限。

所有CI=5000次controller再state簇bootstrap，seed2026101707，保留两seed/臂/变体配对；零动作两个seed逐位相同，不当64个独立state。unknown不重跑、上下界保留；缺失的集中式教师安全基线、教师公平右手规则及冻结critic等保持缺。额外未预注册归一化时长敏感性已单列ANALYSIS_DEVIATION.json并排除正文结论。

Python：/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；绘图用同父目录.venv-c1-plots/bin/python。PYTHONDONTWRITEBYTECODE=1，缓存/临时目录仅本目录。分析最多8CPU，无GPU。旧源码只读导入ring_exchange.expert.circulation_signature作32条专家参照测试，未调用规划或环境step。初始精度在全部导入后恢复x64=False；几何与统计使用numpy float64。专家缺少单列velocity，从已有actions恢复post-step速度；运动学误差上限2.22e-16m。

无需重跑任何轨迹。若要修改分析，先保存版本和新预注册，保留所有旧收据/原始证据；只读打开共享DB也必须file:...?mode=ro。任何后续源码修改仅限新获授权目录。

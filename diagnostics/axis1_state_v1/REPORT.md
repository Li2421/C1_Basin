完整性比对未通过；额外导入源码的首轮哈希补录晚于启动，完整性审计存在时间收据局限。ST1：SUPPORTED，A3交叉拟合遗憾[-1.5066964285714286, 1.2276785714285714]pp，A3−安全基线95%CI [67.85714285714286, 92.74553571428571]pp。ST2：SUPPORTED，对应数字[-2.455357142857143, 0.33482142857142855]pp和[44.640066964285715, 78.23660714285714]pp。ST3（合并）：undetected，95%CI [-1.8973214285714284, 1.3392857142857142]pp。 设计前提审计：14个测试checkpoint与A3训练56个checkpoint/参数重合均为0，仅14个配方已见；因此下列ST数值结论针对指定cohort，纯State轴尚不能判定。

本轮评估指定14个原Ring Flow在新state上的表现：FF、H700、seed0–1、四臂修正菜单，以及nominal only和安全基线；无训练、无新探测。16/16历史复现通过，首动作、结局、步数及完整位置轨迹逐字节一致。

新state按预注册生成并扫描全部DB的UID/content/可恢复物理坐标；无法恢复的旧坐标仍是缺口。所有A3/A0/G_E选择先于新评估标签冻结。共16128条记录经官方journal与merger流程；unknown保留，postflight将unknown标成missing时不重跑。见main_POSTFLIGHT.json、DB_COMPLETENESS.json。

按controller选取的oracle以state半划分交叉拟合；逐state oracle以seed半划分。同时报告样本内上界。置信区间按controller再按state重抽5000次，种子保留在簇内。unknown的不确定性使用保守上下界，不称作失败。

逐state选择的分组结果、nominal→安全基线逐条转移、偏移分层、成功率变化，以及所有方法的碰撞/超时/unknown/遗憾/灾难/伤害见results.json和AXIS_TABLE.md。timeout仅为timeout；无deadlock结论。

历史证据仅标DEVELOPMENT：R2收益5.21pp [2.65,7.88]；R16旧T2报告遗憾0.75pp但有旧state输入局限；R26对齐T2的实际均值见HISTORICAL_DEVELOPMENT.json。历史与本轮菜单、评价state划分不相同。专家在本轮无逐state评估标签，N/A；不为参照额外生成rollout。

源码保护与收据：INTEGRITY_BEGIN/END/COMPARISON、COPIED_SOURCES和各.diff保留。PoC source补录时点明确披露于POC_SOURCE_SUPPLEMENT.json；不能回补为运行前凭证。没有修改已有源码、环境、物理、安全、事件或成功标准，没有git写操作。外部并发修改若存在，以审计具体差异为准。

round33–36尚未开始；按用户要求一次仅执行一个md。

完整性比对发现的目录外变化（不回滚，不修改这些文件）：
- added: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/deliver.py`
- added: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/onsets.py`
- added: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/run_helper2.py`
- added: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/run_helper3.py`
- added: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/table_complete.py`
- added: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/validate_results.py`
- modified: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/analysis.py`
- modified: `/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/final_audit.py`

以上为审计检测到的目录外变化；本轮没有执行修改它们的操作。完整哈希与git状态依据保留于INTEGRITY_BEGIN/END及EXACT_CHANGED_PATHS.json。

资源调度补充：用户要求总计使用18个CPU；原6个worker仅短暂暂停以移交未开始的case，保留当前episode和全部已完成证据。15416个未执行case经追加preflight转入互斥共享队列，数值episode函数逐字不变。raw目录内预留标记不是证据，最终已逐条校验全部16128条真实记录且预留标记清零。动态并发维持其他用户作业计入后的总18CPU目标；收据见SPEEDUP_OWNERSHIP、SPEEDUP_PREFLIGHT及SPEEDUP_IMPLEMENTATION_AUDIT。

从1a到1b，A3成功率变化区间为[-30.245535714285722, -28.85044642857143]pp，平均遗憾变化区间为[-3.6830357142857144, 1.8415178571428572]pp（描述性unknown包络）。

按预注册偏移参数分层的结果：
- radius_below_2.3=True：n=32，A3成功率[61.83035714285714, 62.27678571428571]%，交叉拟合遗憾[-2.455357142857143, 0.33482142857142855]pp。
- speed_above_.11=True：n=32，A3成功率[61.83035714285714, 62.27678571428571]%，交叉拟合遗憾[-2.455357142857143, 0.33482142857142855]pp。
- angular_spread_above_.4=False：n=13，A3成功率[60.16483516483516, 60.98901098901099]%，交叉拟合遗憾[-2.4725274725274726, 0.8241758241758242]pp。
- angular_spread_above_.4=True：n=19，A3成功率[62.96992481203007, 63.1578947368421]%，交叉拟合遗憾[-2.443609022556391, 0.0]pp。

逐controller/方法的成功率与遗憾见CONTROLLER_DETAILS.json；nominal碰撞后的转移比例已列于results.json的fractions字段。

ST1数值标准通过：指定原14 checkpoint在同分布新state上的标准成立；这些精确checkpoint未在critic训练集中，不能映射为已见controller上的纯State泛化成立。

ST2数值标准通过：本轮指定state偏移下的标准成立；含精确checkpoint未见这一共同条件，不能单独归因为state偏移稳健性。

ST3合并结论：未检出差别；逐组oracle潜在增益与置信区间见results.json。该潜在增益来自交叉拟合，不能把样本内oracle当作可实现收益。

Controller身份前提核查（在主结果打开前完成）：R16 source_data读取R15的56个新seed controller，R15 CHECKPOINT_NOVELTY_AUDIT及protocol表明测试原14与训练56的checkpoint/参数哈希均无重合，只有配方相同。证据与收据见CONTROLLER_MEMBERSHIP_AUDIT.json及.receipt.json。此前prompt和本轮初始预注册中的已见controller表述错误，保留原文而不追改；ST1/ST2/ST3按原规则计算，不替换模型或controller、不追加rollout。历史T2亦适用该范围局限。纯State轴属于设计缺口。

最终复核补充：16128条记录及全部轨迹SHA256已独立复核，84条unknown保留，DB无冲突、无未执行缺口，详见INDEPENDENT_DELIVERY_REVIEW.json。Basin的HEAD未改变，但git diff总哈希改变，状态新增`docs/prompt_round30b_frame_consistent_students_20261015.md`。已有未提交Markdown差异涉及['AGENTS.md', 'docs/controller_generalization_ring_summary_20261014.md']；开始快照未保存这些文档逐文件哈希，不能精准归因总diff变化。此局限明确保留，详见GIT_AUDIT_DETAIL.json。旧完成收据及报告已保留为PRE_REVIEW版本；本补充不改数值、统计或科学结论。

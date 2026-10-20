# B 等待明确决定；A/C 已交付

A/C已完成且最终数值复核、源码完整性均通过。GA1=GENERATOR_VALID，中位数93.75%；GA3未触发。当前没有B rollout、没有PSTAR_DECISION.receipt.json，PSTAR_UNDECIDED.json不是放弃或默认选择。NUMERICAL_COMPLETE和SUPPLEMENTARY_DELIVERY为当前A/C交付版本，所有旧round35/36始终只读。

若用户之后明确选候选1/2/自定义或放弃：先把填写后的决定原文、UTC和SHA写入本轮新决定文件/收据，保留原输入与本次已完成交付的完整版本。不要原地破坏旧完成收据的文件hash链；B完成时提供明确的新版本/修订收据。候选1精确值0.9681620884830543；候选2精确已冻结点0.9728120313305867，另保留完整896中位数的识别区间和点约定，不冒充观测到字面896中位数。

离线代码已有：test_selected_progress.py、plan_B.py、run_B.py、B_driver.py、analyze_B.py、B_ANALYSIS_FROZEN.receipt。B需要选定阈值测试先封存，然后2688逻辑请求；F13已有前缀严格复用，只执行新臂的缺失后缀。A/C、历史复现门、原生成器对照与基线不重跑。原Sprog从未生成，已知其他臂unknown仍不重跑。

资源监控与B等待器目前已退出。旧NUMERICAL_COMPLETE标记会使资源监控/等待器在阶段空隙退出；未来启动B前必须妥善版本化本次完成标记、保持当前交付可核验，并使用B专用运行/完成状态或更新本轮编排，不能直接把旧完成状态当作新B已完成。所有有计算量的工作按Slurm/全用户合计18CPU、最多2GPU分片控制，只暂停本轮进程组。默认官方journal与单进程merger，开merge前确认ps/squeue无round30merge。原Python固定；plot_env只渲染图，不用于仿真或统计。

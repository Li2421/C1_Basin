# 跨类型4交接

/home/zhihan/research/Basin_C1/diagnostics/crosstype4_no_repair_schedule_diagnosis_v1

阅读顺序：REPORT.md → DIAGNOSIS.md / AXIS_TABLE.md → GAPS.md → EXECUTION_COUNTS.json。

A使用旧密封选择，不重训：A_OBJECTS.json / A_ACTION_LINKS.json / A_ALL_ACTION_OUTCOMES.csv / A_BOOTSTRAP.npz / A_DIAGNOSIS_RESULTS.json；figures/A_*.svg可直接导出。

B候选为8个冻结提议+零，441唯一动作，13维编码；b_candidates.py、actions.py及CANDIDATES.receipt.json定义身份。TRAINING_ACTIONS.npz保留34槽与原26索引；TRAINING_BUNDLE.npz可复用标签，动作编码改变不使旧执行标签失效。

模型：FROZEN_PIPELINE.json及收据包含19留一类型模型和全部类型最终模型；critic_fits/保存全部source-only曲线、概率、权重及checkpoint。inference.py严格恢复h/c float32、controller统计float64。

选择：NEW_EVAL_CHOSEN_ACTIONS.json / NEW_EVAL_CHOICES.receipt.json 早于新状态首次提交。新24状态与NEW_EVAL_INPUTS.npz可读，不可再声称全新确认性state。

局部等价标签：各阶段*_ALL_LOCAL_EQUIVALENCE.json，均明确实际源UID、轨迹SHA和未切换证明，绝不写DB；共享DB只有真实rollout标签。phase_complete.py分别核查DB和本地证据，未知不重跑。

运行：coordinator.py按A优先、B训练标签、19折模型训练、冻结、B新评估执行；每次提交调用submit_aux.py/resource_guard.py，资源全用户18 CPU/2 GPU。监控信息在MONITOR_STATUS.json；异常不会自动重跑未完成claim。

审计：INTEGRITY_START/END/COMPARISON及HISTORY_READ_INVENTORY.json；所有代码和输出仅本轮目录，旧源码/PoC/官方仓库/venv只读，未Git提交。

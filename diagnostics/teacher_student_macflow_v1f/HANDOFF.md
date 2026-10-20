工作目录为本文件所在目录；Python为/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。任务来源：docs/prompt_round30f_student_competence_dagger_shared_noise_20261016.md，只读。只允许写本目录和官方DB追加流程。

请先读REPORT.md、results.json、RUN_STATUS.json、PREREGISTRATION.md、RESOURCE_EXCEPTION.json、INTEGRITY_COMPARISON.json。COPY_PROVENANCE.json和COPY_DIFF.patch列出复制来源；IMPORTED_SOURCE_AUDIT.json列出实际导入源码hash。所有checkpoint、闭环采集npz、标签、逐轮统计、训练history和门预测均保留。不要重新训练冻结学生，不要重跑unknown。

coordinator.py负责资源检查、复现、训练DAG、全局冻结、pilot、R5、盲选择、主评估、单进程官方merger、统计与交付。没有后台并发时方可启动；先检查COORDINATOR_PID和ps。所有Slurm提交走quota.submit，用户总18CPU/2GPU shard，提交前先读squeue。资源任务失败写GAP并继续独立部分，不自动重试；需要接续时读coordinator_state.json及GAPS.json。不得以改旧源码、包或环境绕过失败。

所有新评估按design→preflight封存→仅missing→默认journal→官方单merger→postflight。合并前检查ps和squeue。复现为任务指定的16条历史记录逐位重放，单列，不写DB重复标签。闭环采集只用于训练，不进入评估统计。

最终报告与表格已复核。FINAL_REVIEW.receipt.json是当前交付收据；原自动交付文件保存在automated_delivery/，原DELIVERY.receipt.json保留为历史收据。原始科学参数、checkpoint、预测与标签均未更改。

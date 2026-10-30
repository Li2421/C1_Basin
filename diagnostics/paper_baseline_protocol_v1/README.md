# 当前执行政策

2026-10-30 用户已删除内部 B 的全局、按类型和固定首组参数覆盖评估。该方式不再作为内部参照或后续步骤，不是临时暂停。

- 内部修正族的表达能力按逐案例搜索 η 的 Oracle 覆盖评估。
- 正式部署和泛化对比应使用真正的泛化 pipeline。
- 外部论文方法已完成的训练选参及历史队列结果保留，见 COVERAGE_272_REPORT.md。
- tuning.py 现在仅支持外部论文方法；旧内部选参及固定覆盖入口已移除。
- dispatch.py 与 worker.py 会在执行前拒绝原内部固定配置队列。
- pipeline.py、freeze_analysis.py、resume.sh 与 watchdog.py 的旧集成执行/恢复逻辑已移除，不会提交或重启作业。
- 原始证据和执行快照保留；不改历史成绩，不删除数据，不因本次调整重跑。

当前没有运行中的实验。已完成的外部历史覆盖可通过 coverage_272_resume.sh 精确缓存复用；该脚本不启动内部 B 选参。

验证见 FIXED_B_REMOVAL_CHECK.json；政策见 USER_REMOVED_FIXED_B_WORKFLOW.json。本政策取代本目录旧协议文本中关于内部固定 B 选参及覆盖的要求。

## 只读学习价值诊断

最新最佳B族的逐案例η敏感性、纯旋转符号对照和证据边界见[ETA_LEARNING_VALUE_REPORT.md](ETA_LEARNING_VALUE_REPORT.md)。仅重算旧证据，未恢复固定参数选参，未新增rollout或训练。

# 跨type2接续输入

本轮工作流完成，开发优越性未达到；完整锁定主终点因隐式能量类型缺失而不完整。请先读 `REPORT.md`、`GAPS.md`、`results.json`，不要把本轮标为跨类型自适应选择成功。最终版本为 `ridge_v1`，九次尝试全部保留。

直接复用的输入：

- 冻结模型：`FROZEN_PIPELINE.json`、`FROZEN_PIPELINE.receipt.json`；内部包括24维机制特征岭回归和同结构R5-PCA2模型。用 `selector.Regression.from_dict` 恢复模型，再 `choose(features)`，不要重新拟合预处理。`choose_locked.py` 是一次性本轮选臂入口，不应用它覆盖既有收据。
- 特征定义：`features.py`，顺序在冻结模型的feature_names。每controller为16state×三臂×300步无成败探测，初观测8键动作查询。`assemble.py`、`initial_actions.py` 给出提取入口；`DEVELOPMENT_FEATURES.npz`、`LOCKED_FEATURES.npz`、对应FEATURE_TABLE和PROBE_TRAJECTORY_INDEX为已封存输入。全部特征不可用则回退按比例时间表。
- 动物园：`ZOO_MANIFEST.csv` 每实例的类型、配方、checkpoint路径/哈希、能力门；新策略在 `policies/`，训练配置与收据、训练曲线均保留。共有56个开发实例、6个锁定强化学习实例；六个隐式能量占位是实现缺口，不能当作已训练或已排除controller。
- 完整标签：`DEVELOPMENT_BUNDLE.npz` 与 `LOCKED_BUNDLE.npz`，对应标签索引、DB登记/preflight/postflight及execution receipts。标签1成功、0已知非成功、-1unknown；原始碰撞/timeout单列。零样本方法不得把目标controller标签作为输入。
- 结果：`iterations/ridge_v1/RESULTS.json`、`CONFIRMATORY_RESULTS.json`、`AXIS_TABLE.csv`、`AXIS_METRICS.csv`、`CONTROLLER_SELECTION_DETAILS.csv`、`SCALE_CURVE_RESULTS.json`及 `scale/`。`DEVELOPMENT_SELECTION_VALUE.json`区分自适应选择与统一好臂，样本内自适应空间约0.391pp。

锁定标签已经打开一次。下一轮不能继续把这些结果称为未见过的确认性证据；任何据此开发的修改须另设确认对象。隐式能量实现修复必须先通过原合成门，不放宽阈值、不替换锁定类型，不回写本轮结果。

所有Python执行使用 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`。新工作另开被授权目录；本轮冻结代码和数据不覆盖，不做git写操作，不装包。禁止目录按decentralized_/teacher_student_/multiscene_前缀剪枝。DB只能官方设计登记、默认journal、单进程merger；先搜历史和精确preflight，unknown不重跑。当前用户授权全局18 CPU、5 GPU shard，已有任务优先。用户允许最终精确preflight后调整旧类型预算，本轮已定18,846次，开发追加标签预算使用0。

最终审计见 `INTEGRITY_START.json`、`INTEGRITY_END.json`、`INTEGRITY_FINAL_COMPARISON.json`；最终交付校验见 `FINAL_DELIVERY.receipt.json`。没有需要继续等待的本轮训练或证据采集任务。

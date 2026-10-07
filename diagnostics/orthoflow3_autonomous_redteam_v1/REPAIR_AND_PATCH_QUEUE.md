# 最小修复与并行工作隔离

## 已实施（本审计独立版本）

1. `scientific_evidence.summarize_canonical16`：用canonical seed身份去重，未认证/缺失保留区间，完整有效16次才返回Q16；B15逻辑不变。实际16320条历史seed重聚合为 `H2_k16_corrected_per_state.json` / `H2_k16_corrected_summary.json`。
2. `exact_label_or_proxy`：不允许不同(state,eta,controller)的邻近标签产生exact robust证书。H1原发展结论加限定，不改原始物理结果。
3. `finite_logit_argmax`：隔离复现已存在于Phase-A代码的数值稳定有限排名，使用相同logits、相同候选、真正同值时仍first-index；不是连续优化、校准或新architecture。

初始/修复回归有明确对照：旧`run_k16.evidence`在15成功+1数值未认证时给标量0.9375，独立修复返回Q16=null、区间[.9375,1]，B15仍true。旧float32 sigmoid把[18,22,20]变成[1,1,1]，首候选错误覆盖raw-score次序；logit排名及float64控制选第二个。真实8个Four探针选择身份改变但两边均B15；Ring68个输入无此变化。

## 不改并行任务的代码

Phase-A采集/训练/冻结链仍由先前任务负责。canonical源文件、模型、v2表、旧报告均只读；本审计没有注入新训练目标、标签或修改其文件。独立目录版本就是这次报告修复，不声称已替换旧生产脚本。

### 待合并候选（无自动应用）

- 历史 `run_k16.py:267–277,344–345,350–360` 若再次使用，应同时迁移为exact-Q/null+lower/upper API及区间regret/exploitation，不能仅把Q16改成null后让下游减法失效。完整可用替代分析是本审计 `audit.py k16`，不需要重新rollout。
- 历史 `analyze_train_dev.py:67–89,141–177` 若用于未来发展决策，必须查精确candidate tuple或显式标注UNDERRESOLVED，不能以nearest proxy关闭训练/数据诊断。
- 保留Phase-A已经使用的logit有限排名；报告before/after时明确旧baseline也按logit重新选了候选，不应将tie处理差异算作纯权重提升。此点在Ring当前队列无差异。

不需要在本次红队任务重训或重启旧实验；修改旧脚本反而会扰动并行任务的冻结hash。所有修复均为可移除的独立目录工件。

# Fundamental-cause audit (2026-10-11)

这是对“为什么 controller–Q 关系学不到可迁移形式”的只读分析。

- 0 rollout、0 训练。
- 只读取 `controller_ff_factorial_repair_v1` 的原始标签、dataset、已保存的 source-only 预测和 raw rollout 摘要，未修改其中任何文件。
- `analysis.py` 生成全部数字，输出保存在 `results.txt`。
- 解读见 `docs/controller_generalization_audit_20261011.md` §10。

范围：K4 面板（η 索引 0/5/10/15），两类 affine controller 家族（goal-reference 和 Flow-reference，各 5 个算子），每场景 40 个 natural states。属于 DEVELOPMENT 证据，不是新的确认集。

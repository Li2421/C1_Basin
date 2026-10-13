# P1 原生 v10 精确复用执行补充

在 P1 新 rollout 前，静态审计 `P1_STATIC_AUDIT.json` 证实新 checkpoint adapter 对原生 v10 的 η0 和 η13 首步动作逐字节等同于历史 native runtime（最大差0）；同一checkpoint、FF、安全层、state、seed及评价程序均沿用原实现。故 P1 execution protocol 中的原生 v10 使用历史 `real_state_expansion/protocol.json` 的原 controller UID，直接读取数据库已有 Q16 标签中的 seed0–3，**不为相同执行语义新造 UID 或重复 rollout**。

原 `P1_PROTOCOL.json` 作为 P1 checkpoint 库存快照保持不变；实际运行及DB preflight 使用另存且冻结的 `P1_EXECUTION_PROTOCOL.json`。其他13个checkpoint的profile不变。本补充不修改P1样本、K16、统计门槛或失败处理。

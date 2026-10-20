# Round19 Four controller axis

阶段 0 等价门未通过：G1 的历史记录没有首动作字节，无法完成规定的逐位比较；G0/G2 在此硬停止后未运行。Four 是否存在跨 controller 的 η 泛化问题为 INCONCLUSIVE；H_compress_Four 与由无标签探测学习 E 均为 INCONCLUSIVE（未进入相应阶段）。

## 事实与门结果

预注册已在任何 Round19 等价门、探测或新成功标签前写入并留 UTC/SHA256 收据。
旧 round14B 的 v13 等价门在其原有适配器上 8/8 通过；这不是本轮新适配器的 G0。
只读扫描共享 DB 对应的原始 journal：v12 best 1,360 条，v10-wide best 1,024 条；源文件都存在，原始记录哈希均与 DB 匹配。两组记录的 `first_action`、`first_action_bytes` 和 `trace_path` 均为 0 条；DB rollout 表也无首动作列。详见 `G1_EVIDENCE_AUDIT.json`。
G1 要求每个 checkpoint 8 条历史记录的终止原因、步数和首动作字节逐位复现。没有历史首动作，关键比较无法成立；按预注册判为 NOT_PASSED_UNVERIFIABLE，并执行阶段 0 硬停止。G0/G2 的本轮执行状态均为 NOT_RUN，不将历史 v13 结果冒充新适配器验证。
只读配置审计找到 13 个同环境 best 和 12 个参数不同的 final；这只是候选清单，G2 的 config/前向门尚未运行。

## 解释与范围

这一停止说明当前历史证据不足以验证逐 checkpoint 适配层，并不证明 Four controller 共享 η regime，也不否定 Ring 的机制。阶段 1–4 均未开始，故没有新探测、state、rollout、标签或模型。没有触及共享 DB 写入。
本轮未形成 Four 评价 batch 的视频；无合格 deadlock 判据或同初态死锁基线，不能把 timeout 用作 deadlock_show/deadlock_resolution。视频交付标记为 INCOMPLETE，不声称完成。

## 可追溯文件

`PREREGISTRATION.md`、`PREREGISTRATION.receipt.json`、`HISTORICAL_EVIDENCE_AUDIT.json`、`G1_EVIDENCE_AUDIT.json`、`GATE_REPORT.json`、`results.json`、`STOP.receipt.json`。

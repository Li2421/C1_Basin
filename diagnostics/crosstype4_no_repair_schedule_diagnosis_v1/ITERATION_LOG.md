# 跨类型4实现记录

新轮次独立目录；前轮只读。完整重读提示词最新版本，包含32条提前结束等价验证。本轮先做开发性差距拆解，再增加零η与进度选项，不更改其他网络结构。18 CPU/2 GPU；资源解析已在新目录修复gres/shard:前缀，未改旧轮次。

复现清单修正：跨类型3修订后的新评估无原九类学习型，首版错误假定有Flow而在生成清单前断言失败；启动脚本未等待成功条件，误提交26215后取消。该作业缺清单，未执行物理rollout。修正版选8条进度+8条安全，覆盖新三类和两模型预测控制类。FF原生/适配分支在额外前缀与提前结束等价门覆盖。本修正只根据历史清单元数据，不依赖本轮结果。

## Native Flow source lookup correction
Before registration or execution, the equivalence design found that the native Flow base manifest retains a legacy-prefixed controller alias. The historical crosstype1c/3 action cases use the payload digest instead. Corrected fixed-source lookup to use the identical canonical digest used for action cases; no physical execution, label or cache semantics changed; no reruns.

## Worker freeze amendment before first A submission
A design sealed its worker while a dynamic actual-source reuse extension was being prepared. Detected the hash mismatch before any A submission. Restored rollout_worker.py exactly to its original sealed SHA, saved the extension as rollout_worker_v2.py, and issued A_EXECUTION_FROZEN_v2.receipt.json with runtime dependencies. The old receipt is preserved and all its file hashes match. No A label was executed or changed; no reruns.

## Coordinator startup correction
The first background monitor attempt lacked explicit imports for queue and shell quoting. It stopped before submitting or executing any new work; the active A array continued independently. Added explicit imports and retained the original failure record in MONITOR_FAILURE_2496616.json. No scientific rollout rerun or configuration change.

# 论文基线轮进展

原环境复现保留死锁感知控制和旋转木马；Chen因默认场景帧间连续碰撞退出，其余三个无作者代码/参数的方法退出。方形与十字验证/测试的物理一致初态合法率均128/128，主表按预登记规则使用一致版本；冻结学习模型训练于旧版本，不重训。

正式调参和测试以冻结文件及下表为准；不把单机诊断当作正式成功率。

| 阶段文件 | 已产生 | 完成标记 |
|---|---|---|
| NATIVE_REPRODUCTION_RESULTS.json | True | True |
| PORT_FIX_BEFORE_AFTER.json | True | True |
| NARROW_PASSAGE_DIAGNOSTIC.json | True | True |
| CONSISTENT_BOX_SCENES.json | True | True |
| PORT_GATES.json | True | True |
| ENV_REUSE_PHASE_COMPLETE.json | True | True |
| PARAMETERS_FROZEN.json | False | — |
| FORMAL_TEST_STARTED.json | False | — |
| CONSISTENT_OURS_RESULTS.json | False | — |

## 返回值修复训练单机对照

| 方法 | 基准 | 修复前成功/总数 | 修复后成功/总数 | 前/后未处理异常 |
|---|---|---:|---:|---:|
| Zhang T-RO 2025 死锁感知控制 | A | 219/219 | 219/219 | 0/0 |
| Zhang T-RO 2025 死锁感知控制 | B | 216/279 | 231/279 | 63/26 |
| Lee IROS 2025 旋转木马 | A | 202/219 | 202/219 | 0/0 |
| Lee IROS 2025 旋转木马 | B | 187/279 | 187/279 | 0/0 |

新增尝试预留8753/50000；实际轨迹、逐候选计划、preflight与官方journal保留。

原环境离散/连续安全差异、依赖修复、参数出处与当前缺口详见NATIVE_REPRODUCTION.md、MARGIN_ALIGNMENT.md、GAPS.md。

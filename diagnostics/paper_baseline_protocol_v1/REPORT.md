# 最新分析：最佳B族存在明确的逐案例η选择信号

详见[ETA_LEARNING_VALUE_REPORT.md](ETA_LEARNING_VALUE_REPORT.md)：208/272例已有同案例成功/失败η对照；纯旋转符号改变使33例成败翻转。只分析旧证据，新增rollout=0、训练=0。独立泛化尚未验证。

# 当前优先结果：已完成旧272案例对应队列的外部固定参数覆盖

详见 [COVERAGE_272_REPORT.md](COVERAGE_272_REPORT.md)。内部 B 的全局/按类型/固定首组选参覆盖已按用户指示删除；旧整体流程与恢复入口已停用，见 README.md。

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

## 训练集选参进度（非独立测试成绩）

| 基准与方法 | 完成 | 选中组 | 扩到32组 | 选中组训练成功率 |
|---|---|---:|---|---:|
| 基准甲：Zhang T-RO 2025 死锁感知控制 | True | 10 | False | 0.6428571428571429 |
| 基准甲：Lee IROS 2025 旋转木马 | True | 25 | True | 0.23214285714285715 |
| 基准乙：Zhang T-RO 2025 死锁感知控制 | True | 12 | False | 0.25390625 |
| 基准乙：Lee IROS 2025 旋转木马 | True | 10 | False | 0.1171875 |
| 基准乙：四列加通道单向放行 | True | 31 | True | 0.5390625 |

当前批次：tune_B_ours_four_v005_r1，978/2048；CPU实际16、上限18。

## 返回值修复训练单机对照

| 方法 | 基准 | 修复前成功/总数 | 修复后成功/总数 | 前/后未处理异常 |
|---|---|---:|---:|---:|
| Zhang T-RO 2025 死锁感知控制 | A | 219/219 | 219/219 | 0/0 |
| Zhang T-RO 2025 死锁感知控制 | B | 216/279 | 231/279 | 63/26 |
| Lee IROS 2025 旋转木马 | A | 202/219 | 202/219 | 0/0 |
| Lee IROS 2025 旋转木马 | B | 187/279 | 187/279 | 0/0 |

新增尝试预留17785/50000；实际轨迹、逐候选计划、preflight与官方journal保留。

原环境离散/连续安全差异、依赖修复、参数出处与当前缺口详见NATIVE_REPRODUCTION.md、MARGIN_ALIGNMENT.md、GAPS.md。

## 用户优先交付：旧272案例对应初态上的已调参外部方法

本批已完成，两种方法各136个物理初态（17代表地图×状态0–7），与原C1双nominal272回合对应。固定使用B训练集选出的参数，不逐案例搜索，不当作独立泛化结果。死锁感知控制31/136（22.79%），成功平均17.018秒，全队列截断37.357秒，碰撞22、超时83、终局数值未知0；40回合曾发生未处理异常，保留“端口问题未完全解决”标识。旋转木马9/136（6.62%），成功平均15.967秒，全队列截断39.281秒，碰撞123、超时4、终局数值未知0。

逐场景、置信区间、计算开销和端口适用性见[COVERAGE_272_REPORT.md](COVERAGE_272_REPORT.md)，逐回合身份与指标见COVERAGE_272_ROWS.csv和COVERAGE_272_ANALYSIS.json。全部272回合经官方preflight（0可复用、0歧义）后补跑，官方journal已合并并读回；逐初态/配置身份与272条轨迹SHA256核验通过，250个冻结数值文件未变。大型原始结果与轨迹保存在已挂载SSD的coverage_272_artifacts。Git仅提交代码、小型报告和冻结元数据，不构成大型原始数据独立备份。

内部B固定配置选择与覆盖评估已按用户要求删除，旧resume.sh/pipeline/watchdog均不再启动或恢复该流程。没有触发新状态12–19正式测试。后续泛化比较交给真正具备泛化能力的pipeline，内部表达能力按逐案例Oracle搜索评估。仅复演已完成的外部历史队列可用coverage_272_resume.sh（精确缓存去重）。

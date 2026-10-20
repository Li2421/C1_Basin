严格完整性审计未完全通过：20 项间接导入源码缺少本轮开工逐文件快照，仅与更早的不可变源码快照核对一致；已采样的保护源码、仓库状态与差异哈希起终一致。族终点一：6 个族的六个实例均通过观测能力门；其中 5 个族满足实现要求并纳入确认性评估，3 个族的按比例切换时间表相对只加安全投影基线的配对增益置信区间下界大于零。族终点二：零样本扩展选择模型的完整菜单交叉拟合平均遗憾的未知上下界为 1.77 至 2.50 个百分点，可能伤害数 0，条件一通过；条件二因前提未满足而不确定。族终点三：四个先导状态（每个控制器八条闭环标签）选择的平均遗憾的未知上下界为 0.89 至 1.56 个百分点，通过。集中式随机采样模型预测控制：能力门通过，样本均值偏持续右手规则，但尚无可靠偏好证据。交叉熵方法模型预测控制：能力门通过，样本均值偏持续右手规则，但尚无可靠偏好证据。优先级时空 A* 规划：六个实例已通过观测能力门，实现单元测试失败，偏好为缺口。确定性行为克隆策略：能力门通过，偏好时间表。混合密度行为克隆策略：能力门通过，偏好时间表。扩散行为克隆策略：能力门通过，偏好时间表。

工作根目录：/home/zhihan/research/Basin_C1。用户要求全部本轮写入位于 diagnostics/blackbox_families_v1/；其他既有源码只读。不得读取或写入 diagnostics/decentralized_blackbox_planners_v1/。不做 Git 写操作，不安装包。Python 为 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。

设计依据为 docs/prompt_round38_more_blackbox_controllers_rhr_menu_20261017.md 和 docs/generalization_axes_protocol_20261015.md；PREREGISTRATION.md 与对应收据早于本轮新执行。固定参数持续右手规则、十八个学习实例、全部选择与公平调参规则均有冻结收据。所有选择先于主评估，不能用主评估重新选择方法或参数。

数据库实验身份为 exp_blackbox_families_v1。各 phase 的 *_PLANNED_ROLLOUTS.json、*_CACHE_PREFLIGHT.json、*_PREFLIGHT.receipt.json、*_INSERTED_ROWS.json、*_REGISTRATION.receipt.json、*_MERGE.log 和 *_POSTFLIGHT.json 串起官方流程。未知直接身份仍在数据库，官方 planner 会把它们列为缺失；使用 *_POSTFLIGHT_UNKNOWN.json 证明已存在，不得重跑。旧场景别名的首次流匹配 journal 只被归为摘要；修正身份后原轨迹全部入库，原记录保留。

控制器实现：集中式随机采样模型预测控制逐字复用 legacy_controllers.py；交叉熵方法在 controllers.py；优先级时空规划在 spacetime_astar.py；三种学习策略在 learned.py，训练入口 train_policies.py。持续右手规则的类逐字来自原实现，见 RIGHT_HAND_CLASS_SOURCE.json。按进度切换逐字复用 progress.py。episode.py 仅替换控制器原始动作来源和新增修正臂，物理、安全层、事件与成功标准均使用保护源码。时空搜索的等价编译加速在 fast_astar.py 与 astar_kernel.cpp 中，二十四条优先级路径与原实现逐位一致，这仅验证编译加速等价，并不证明原算法满足直线要求；搜索顺序、平局规则、几何及控制参数均未调整。最初 Python 作业继续完成，没有取消或重跑已认领任务。编译命令与版本见 FAST_ASTAR_BUILD.json。

表格结果与逐条证据入口为 AXIS_TABLE.md、METHOD_METRICS.json、EVALUATION_LABELS.npz、EVALUATION_RAW_INDEX.json 和 NOMINAL_TO_SAFETY_TRANSITIONS.json；五千次配对控制器/状态簇 bootstrap 在 analysis_statistics.py。所有 unknown 保留下/上界；超时不是死锁。族终点一至三为前瞻 CONFIRMATORY（确认性）证据，留一族与训练标签分析为 DEVELOPMENT（开发性）证据。

当前统计入口为 analysis_valid_families.py，原 analysis.py 与统计函数保留封存。优先级时空规划通过了上游观测能力门，但斜向直线路径单元测试失败，因此合并仅包含其余五族三十个控制器、一万五千三百六十条完整八臂评估；该族二百四十四条工程评估已保留，二千八百二十八条未认领身份按 MAIN_BLOCKED_UNCLAIMED.json 跳过。不要因原始 manifest 仍将这些身份列为缺失而补跑，也不得把它们伪装成数值未知。工程修正参考从未用于本轮执行，补齐该族需要新身份、预注册与选择冻结。

资源限制是用户全部任务合计十八个 CPU shard、两个 GPU shard；每次提交前保存 squeue，不可无条件启动新的十八路数组。没有视频交付。

缺口与补齐条件：
- 起始导入来源审计覆盖缺口：20 项间接导入源码未在本轮开工快照单独采样。其历史不可变哈希与本轮记录一致，已采样源码起终一致，但不能追补当时快照。路径与对照见 IMPORTED_SOURCES.json；下一轮须在任何导入前覆盖 PoC 的 source、validation 和 vendor 依赖。无需因此重复训练或重跑本轮轨迹。
- 共享数据库中 4607 个历史状态缺少可恢复的完整物理布局，无法证明与这些记录的物理去重；另有 2086 个已知非四机器人状态，其数组布局与本轮不同，不属于重复风险。原始去重可见性计数过宽，细分证据见 STATE_DEDUP_VISIBILITY_REFINEMENT.json；全部可恢复且相容的状态及本轮状态彼此去重已通过。
- 优先级时空 A* 规划：观测能力门六个实例通过，但初始直线单元测试只覆盖轴向路径，斜向空地图测试偏离超过一个栅格，违反实现要求；该族确认性评估与偏好为缺口。保留已认领的二百四十四条工程评估证据，二千八百二十八条未认领任务已跳过；没有取消在途任务、改控制器或重新选择。已另写工程修正参考及斜向路径、时空占用测试，位于 corrected_astar_reference.py 与 CORRECTED_ASTAR_ENGINEERING_TESTS.json；它从未用于本轮冻结执行，不能替代本轮证据。补齐仍需要通过完整单元测试，并在任何新的确认性标签之前重新预注册、探测、调参和冻结该族选择；不得用现有五族结果补选参数。
- 五个有效族的确认性评估有 25 条未知事件（按八臂表格计数，物理身份去重数另见审计）；全部保留并报告成功率上下界，没有重跑，不能据此断言已知失败。
- 未满足至少一个新族的固定参数持续右手规则高于按比例切换时间表至少十个百分点的前提，因此零样本识别偏好族的条件二不确定。
- 集中式随机采样模型预测控制：固定参数持续右手规则与按比例切换时间表的配对差置信区间跨零，不能给出可靠偏好。
- 交叉熵方法模型预测控制：固定参数持续右手规则与按比例切换时间表的配对差置信区间跨零，不能给出可靠偏好。

完整性审计比较见 INTEGRITY_COMPARISON.json。补充导入来源的起始比对使用早于本轮的不可变源码快照，未在本轮初始快照重复采样，范围逐项列于 IMPORTED_SOURCES.json；最终逐路径、冻结收据与数据库覆盖核查见 FINAL_VERIFICATION.json。

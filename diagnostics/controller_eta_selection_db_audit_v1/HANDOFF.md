# 独立交接：State-conditioned η Selection 数据库优先调查

本文件供没有本次对话历史的研究 Agent 使用。项目为 Basin_C1 / OrthoFlow3；η 是闭环修正场的三个连续参数，本次问的是在真实 FF/OrthoFlow3/硬安全投影执行下，**读取初始物理 state 后选择离散候选 η，是否可靠优于同一场景/controller的一条固定 η**。此前 Toy/DB 宽度及独立因素研究常出现固定η成功饱和，曾有人提出选η主要是场景级问题；但上游候选面板清单没有遍历共享数据库。

## 本次实际完成什么

在仓库根目录先核对 `AGENTS.md`、`shared_rollout_db/README.md`、历史协议、数据库 schema、checkpoint/预测及原始证据；然后预注册并只读扫描共享数据库。新目录脚本 `inventory.py` 以覆盖、UID和执行语义建面板，`extract.py` 按同一 state/η 共用 literal seed_key 取真实 rollout，`e0_analyze.py` 做独立 8/8 seed 的双向交叉拟合和2000次 state bootstrap，`e0_decision.py` 剔除漏掉已知最强固定η的候选池。没有更换成功判据、筛掉失败state或把数值未知当失败。后续复用既有 Ring 前瞻冻结预测作 E2确认，用旧全交叉因子数据做 E4 zero-shot/1/2/4-shot场景prior，审计 E5 连续采样。新 rollout **0**、新神经网络训练 **0**。详见 `REPORT.md` 的真实实施过程、代码和命令。

数据库现有2,330,826条 rollout，Q16有效格子57,223。E0结构面板171个（≥12 state、≥2 η），metadata初筛106个，最终97个能对齐全部η的16个共同seed；9个 seed 对齐失败并保留原因。E0复用180,928条不同现有 rollout；历史Ring E2再引用65,536条、因子E4引用3,072条，与E0/E2的61,648条交叠扣除后，**一共引用187,888条不同旧物理 rollout**。13个旧清单前缀均核实于 `LEGACY_13_AUDIT.json`，但遗漏了最大的 Ring FF true-t0 面板。97个面板并不独立，尤其32个合格Ring面板大量重叠。

## 最关键定量结果

| 证据 | 固定η | state-conditioned | 配对差、CI | 解释 |
|---|---:|---:|---:|---|
| E0 Ring FF H700，312 state×13共享η×16seed | 88.50% | 94.29% | +5.79pp [3.35,8.81] | 每个state需8个 discovery seed 的交叉拟合参考，证明可用收益空间；不是部署策略。 |
| E0 Ring FF H700，141×16×16 | 90.07% | 95.39% | +5.32pp [1.73,9.49] | 含历史最强固定η13，四组额外随机split方向一致。 |
| **历史 E2 前瞻 Ring新初态256×K16×16seed** | 88.33% | **93.54%** | **+5.21pp [2.65,7.88]** | 冻结、可部署Q(s,η) TRAIN512，三拟合各+5.47/+5.66/+4.49pp；同一K16所有固定η均被比较。 |
| 历史 E2 Q(s,η) TRAIN32 | 88.33%固定 | 85.07% | 比TRAIN512低8.46pp [6.14,11.08] | 训练覆盖量影响可部署选择，不保证少样本可学。 |
| E0 Toy TT H850，24×16×16 | 84.11% | 98.44% | +14.32pp [1.82,26.31] | 小state数、CI宽，次要支持；同Toy的FF K16面板固定η已100%。 |

E0 原始数值门槛使37个面板暂通过；根据预注册“候选池须包含强固定η”，4个被剔为探索性（Ring3、Toy FF1），主判定**33个：Ring32、Toy TT1**。这些不是33次独立验证。Ring 312×13的 `U=11.50%`、`I=0`，全部可行 state 的B15覆盖至少需4个η，说明收益不靠极少失败state或删除通用强η产生。历史 E2 自身在所有256个新初态、8个独立evaluation seed上，Q(s,η) 对每次 bootstrap 重新选出的最强固定η也高+5.21pp [2.64,7.94]。历史实验有296/65,536次数值未知，原分析保留上下界；本次 E0仅用有效二值记录。历史 E2 相对当年模型开发独立、在预冻结协议下完成；其61,648条记录与本次 E0 复用重叠，故**不是本次筛查后才产生的新确认**。

E4旧因子面板：12个物理配置、24 horizon 配置、2个解析controller、K4、Q2；目标state 0–7用于无标签机制特征/校准，8–15只评估。在留物理配置测试上，source-only固定η73.96%，零目标标签 raw参数77.86%、机制代理77.99%，目标4-state prior-only78.39%，测试集 hindsight最佳固定η80.21%。机制代理相对 source prior +4.04pp CI [−0.91,9.51]，不稳定；留拓扑时+4.56pp [0.26,9.38]，但相对 raw参数仅+1.43pp [−2.73,6.64]。4-state校准较可靠地恢复场景regime，**没有证明机制代理的跨拓扑不变性**。旧四场景校准文件显示 DB 1-state prior-only约98.8%、Ring 4-state prior-only约76.0%，但 checkpoint/场景混杂，不能当成零标签或跨机制迁移。E4使用上游已存在的 factor原始数据，新增采样成本0；若从零部署，K4×2controller×2seed对应每校准state 16次rollout。

## 假设与研究决策

- **被证伪：** “OrthoFlow3 的 η 选择在所有当前设置下主要是场景级、初态自适应无可靠收益。”Ring固定FF/K16的部署选择收益已经前瞻确认。
- **得到支持：** 该 Ring 原生初态分布下，state×η 排序可被有足够训练覆盖的 Q(s,η) 使用；仅提高 Q correlation 不够，必须报告实际 top-1成功、固定η对照、regret及 state错配。
- **尚未解决：** 同一 state×η 规律能否跨几何、跨controller、跨拓扑、跨agent count和不同horizon迁移；这种规律是否由可计算物理机制坐标概括；连续η success basin形状/连通性是否可泛化。旧四场景 LOSO在共同η上没有可靠 state匹配增益，显示当前方法的迁移缺陷仍可能存在，但旧场景的控制器、geometry与时限混杂，不能与固定Ring正结果互相否定。
- **方向建议（推论）：** 保留强 state-conditioned η selection 作为研究主线，首先解决跨geometry时 state×η 交互的表示与数据支持；并行研究 controller/geometry-conditioned 场景prior和few-shot校准。不要因为旧Toy宽度饱和转向“纯场景η”一条线，也不要因为Ring正结果宣称通用迁移已解决。

## E3 和 E5 的明确边界

符合E0门槛的Ring FF只有**单一地图的同语义Q16**。先前 Toy width MANY/FEW/EXTRAP虽显示更多几何改善 Q MSE（约−0.032），选择成功差却为0，因为固定η饱和；它不是强Ring条件下的E3。严格四场景LOSO也有controller/geometry混杂。要在Ring做最小正式4训练+2测试几何、每几何16 state×16η×16seed，一个controller至少24,576次新rollout，两个controller至少49,152次，尚需几何有效性pilot和模型训练，已超出本任务“只做针对性最小补充验证”的数据边界。**E3正式迁移未执行；结论 INCONCLUSIVE**，缺口和恢复设计在 `E3_COVERAGE_GAP.json`。

E5全库确有局部密集连续η数据：45个state/controller组各≥64个不同Q16 η；Q64精度下12组各≥64个不同η、最多334个。但是Toy8个密集组的完整η交集仅15，DB4组仅58；上游连续critic审计还发现边界样本稀缺与同时留state/η时排序失败。不能用这些局部点或K4/K8/K16离散面板推断连续 success basin 的形状/体积/拓扑。E5本次只审计、不增加密网格。

## 下游直接可用的文件

- 机器主结果：`diagnostics/controller_eta_selection_db_audit_v1/results.json`；完整报告：同目录 `REPORT.md`。
- E0预注册和时间/hash：`PREREGISTRATION.md`、`preregistration_receipt.json`；E4冻结补充：`ADDENDUM_E4.md`、`addendum_e4_receipt.json`。
- E0所有97面板的数值、CI、固定/自适应、鲁棒及set-cover：`E0_PANEL_RESULTS.json`；资格表 `E0_DECISION.json`；每格真实标签/rollout UID：`panel_data/*.npz`；结构/seed/原始来源：`inventory.json`、`PANEL_INDEX.json`、`SOURCE_FILES.json`、`REUSE_AUDIT.json`。
- 历史部署确认：`diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/{protocol.json,audit_results.json,verdict.json,verification.json,frozen_predictions.npz,seed_outcomes.csv}`。原始65,536次rollout在其 `raw/`，同时在共享数据库。
- E4：`E4_REGIME_RESULTS.json`、`e4_regime.py`；上游 `diagnostics/controller_state_geometry_transfer_v1/data/factors.npz` 与 `calibration_results.json`。
- E3缺口/估算：`E3_COVERAGE_GAP.json`；E5计数：`E5_DENSE_ETA_AUDIT.json`；旧连续critic失败诊断：`diagnostics/orthoflow3_continuous_basin_critic_v1/failure_diagnosis.md`。
- 只读权威数据库：`shared_rollout_db/rollout.sqlite`，本任务校验 SHA256 `7fe981632aea449762233f21431d3ed1a9fea6866edb7fc65d69094b46d6e509`。Git基线 `a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e`；复算命令在 `REPORT.md` §8。所有新增代码、派生文件都在本结果目录，旧原始数据不应覆盖。

**最值得做的下一项实际研究：** 建立一个受控 Ring 几何连续族，并在生成标签前先做几何有效性/控制器语义和最小成本pilot；预注册包含历史最强固定η的K16候选池、train/test物理state及future seed双重隔离，首要判据用top-1成功与matched−mismatched、固定η regret，Q相关只作辅助。若资源不足，先从现有Ring 256初态的冻结预测中分析错误类型及训练覆盖需求，而不新造大批rollout。

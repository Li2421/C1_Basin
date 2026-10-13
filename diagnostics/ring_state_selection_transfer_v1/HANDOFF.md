# 独立交接：Ring Controller–Q 与η Selection

## 研究问题和位置

本项目是Basin_C1 / OrthoFlow3。核心目标是学习可迁移Q(s,C,η)中的Controller作用，并在未见Controller下有效选η。本轮按用户修订执行N1+C0→C1→条件性C2/C3，N2/N3仅辅助。几何N4/N5已后置。本目录为`diagnostics/ring_state_selection_transfer_v1/`；本文件可脱离对话阅读，完整方法/统计在`REPORT.md`，结构化结果在`results.json`。

## 实际完成的工作

核对历史Ring Q512模型/协议/标签并重新推理冻结checkpoint；复用canonical学习曲线checkpoint，补两个嵌套state排列的公平拟合，比较标量Q、16-head、kNN、prior。原Q512本来就是物理23维+η的MLP，不是entity encoder。只读普查Ring数据库并构造精确state/η/seed匹配的Controller面板，保留数值未知。新增连续FF算子κ=−1,−0.5,0,0.5,1的缺失标签，原生FF另作anchor，固定nativeFlowv10端点/geometry/K16/H700/安全层。执行静态EffectiveBasis可计算性、投影非线性核查，及适用的Controller表示泛化诊断。N2/N3、checkpoint规则诊断均零新增rollout。

累计复用174,784个唯一已有rollout，新增20,736次，新增网络拟合141次、复用N1冻结checkpoint30个。新增rollout episode累计12.20CPU小时。共享数据库只按preflight/journal/单merger流程追加，未覆盖历史数据。完整原始UID索引`REUSED_ROLLOUT_INDEX.json`和`c1/cases.json`；数值未知作为未知保留。

## 最重要的定量结果

- **历史确认复现成功**：Q512=93.54%，固定η13=88.33%，+5.21pp，原max-over16 CI[2.64,7.94]pp；冻结logits最大误差0、选择完全一致。历史256state的确认身份属于旧实验；本轮对其新比较都是DEVELOPMENT。
- **样本效率**：历史VAL-NLL协议、三子集/三fit下，原标量Q的开发n*=512；16-head n*=128；kNN n*=64但只有约+0.85pp。heads128=91.77%、heads256=92.83%，Q512=93.54%。heads128相对scalar128 +4.74pp CI[2.65,7.02]；但仍显著低于Q512。不能声称更少state已经达到同等性能。
- **不只有数据因素**：仅改canonical scalar128的验证checkpoint目标，成功率89.27%→91.72%；同规则却使scalar512降到88.33%。模型选择/目标确实影响selection，不能把当前问题完全归为数据不足或完全归为网络结构。
- **C0直接Controller关系**：固定geometry/Flow/state/seed，FF vs TT的平均Q下降约13.5–14.7pp；全局最佳固定η均13、fixed错配遗憾0，但强state×Controller排序反转在独立半分上成立。保守censor交互能量CI为正。任意argmax ties可把TT的matched−mismatched夸大到71.35pp，prior-aware tie后只有约7.81pp且CI触0；FF方向8.59pp CI[1.04,17.71]。不要把交叉拟合oracle当成可部署selector。
- **真实Flow旧证据**：v9/v7在180相同state、两个η、Q16下偏好不同候选，fixed跨Controller损失22.92pp / 9.24pp。条件是committed原生第一步和旧execution，η池亦不同；不能当成本轮自由t0 FF C3的完成证据。
- **C1门结果**：A（极端κ固定η遗憾门）=True；B（至少3/5κ的可靠state headroom）=False；selection-transfer合格=False。各κ具体成功率和CI见下表。

| Controller | 固定 η (两半) | 固定成功率 | state oracle 成功率 | gain / CI95 | 未知数 |
|---|---|---|---|---|---|
| rotate_negative | [10, 10] | 98.83% | 98.05% | -0.78pp / [-2.34, 0.00]pp | 81 |
| -0.5 | [4, 10] | 97.66% | 97.66% | +0.00pp / [0.00, 0.00]pp | 46 |
| isotropic | [13, 4] | 95.31% | 96.48% | +1.17pp / [0.00, 3.12]pp | 34 |
| 0.5 | [10, 10] | 99.61% | 99.61% | +0.00pp / [0.00, 0.00]pp | 41 |
| rotate_positive | [4, 10] | 90.23% | 97.66% | +7.42pp / [1.95, 13.67]pp | 64 |
| flow | [13, 13] | 95.31% | 97.27% | +1.95pp / [0.00, 5.86]pp | 20 |

- **C1更强检查**：κ=+1相对discovery固定η的+7.42pp虽通过单Controller主规则，但同时对照完整K16的最强固定η后仅+5.86pp，CI[-0.39,8.98]pp；不能视为可靠新benchmark。极端κ错配固定regret +5.08pp也不稳定于两半（10.16/0），反向CI跨0。
- **C2正式状态**：NOT EXECUTED — GATE FAILED (formal C2 selection transfer)。C1只有κ=+1通过state headroom门，少于预注册至少3/5；不能把源面板饱和时的Q拟合当作selection泛化。原极端controller固定遗憾门通过，全family20比较的更严格同时门未通过。保留完整K16及全部state，不扩大或缩减候选制造收益。 Q-only fallback已实际跑七个表示/消融×三fit；细表见REPORT和C2_FALLBACK_RESULTS.json。只有16训练state/源Controller、8完全heldstate，不代替正式selection确认。
- **C2表示诊断的实际限制**：R4_action的Q MSE约0.145，比无Controller约0.179低，但正确/错配描述的Q差CI跨0；8个完全held state上固定η观测成功100%，没有selection提升空间。不能据此宣称R4支持Controller泛化，也不能据此排除其潜力。
- **C3正式状态**：NOT EXECUTED — C2 selection-transfer gate not qualified; asset/legacy audit completed。已完成数据库/真实checkpoint盘点和可兼容旧协议分析，未声称新的真实heldFlow迁移成功。
- **有效修正基底**：解析RB在未投影affine积分误差<3.4e−8，但完整投影FF导数与RB可差23%–94%；η=0时五算子动作仍相差0.333–0.361联合L2。nominal14发生alias，compact64能够区分；局部响应的可计算性不等于终端Q充分性。
- **辅助state解释/外推**：物理特征linear ridge的heldout η-contrast R²约0.056，selection无可靠收益；预先定义chirality半空间N3未发现稳定额外外推损失，但只检验一个较温和切分，不证明普遍state鲁棒。

## 已支持、否定和尚未解决

SUPPORTED：历史nativeRing state-selection收益；Controller影响Q且不止整体level；特定legacy真实Flow会改变最佳固定η；原协议的样本规模与模型选择影响selection。

REJECTED：从“最佳固定η不变”推断Controller只改Q水平；将未投影RB视为完整安全FF/闭环精确响应；把低Q误差自动当作selection泛化。

INCONCLUSIVE：Controller表示是否足以迁移复杂终端Q；R4是否可靠优于compact64；真实自由t0 FF heldFlow迁移；连续η success basin；跨geometry机制不变性；Critic是否存在不可由训练目标/覆盖解释的固有表达缺陷。

C1资格门失败只限制本family作为预注册selection benchmark的资格，**不等于证明不存在Controller×state×η交互**。旧开发数据的新分析和小规模Q-only fallback不能升级为确认。

## 对后续研究的实际影响

把Controller–Q作为主线，避免再次用每Controller/geometry只有16state的失败排除可学性。N1说明模型readout和checkpoint目标需要作为控制变量。Mechanism特征至少应区分未投影映射、完整投影Jacobian与η0执行动作；局部state/key响应不包含未来轨迹/约束切换。C0显示即使同一通用固定η最优，Controller仍可能改变state层排序，不能仅看全局bestη。

最有价值的下一问：在含强固定η、真实headroom和足够state覆盖的面板上，source-only模型能否通过有效执行响应预测未见Controller引起的η排序变化，并在独立state上超过固定与错配参照？之后再固定geometry换真实Flow，最后跨geometry。若补标签，先按精确execution身份查DB；不要重复当前已追加的20736条，不重试覆盖unknown，不删除η13制造收益。

## 可直接使用的资产

- `REPORT.md` / `results.json`：完整实施、指标、证据边界及各阶段状态。
- `PREREGISTRATION.md`、`PRIORITY_AMENDMENT.md`及各`*_RECEIPT.json`：事前设计、目标修订及hash/时间证据。
- `DATA_INDEX.json`、`SOURCE_VALIDATION.json`、`REUSED_ROLLOUT_INDEX.json`、`FINAL_VALIDATION.json`：逐条provenance和一致性。
- `N1_RESULTS.json`、`N1_predictions.npz`、`fits/`、`N1_PAIRED_COMPARISONS.json`、`CHECKPOINT_DIAGNOSTIC_RESULTS.json`：学习曲线与失败诊断。
- `C0_CONTROLLER_CATALOG.json`、`C0_RESULTS.json`、`controller_panels/`、`C0_FF_TT_FOCUS.json`、`C0_CENSOR_ENERGY.json`、`C0_RANK_REVERSALS.json`：直接Controller机制证据。
- `LEGACY_FLOW_FOCUS.json`、`C3_ASSET_AUDIT.json`、`legacy_flow_panels/`、`FLOW_CHECKPOINT_INDEX.json`：真实Flow旧证据及与当前任务不兼容的原因。
- `c1/`：冻结protocol、exact preflight、cases、逐rollout raw、Slurm receipts、merge审计、labels.npz、RESULTS.json、RELATIONSHIP.json。这是本轮新增可共享数据库资产。
- `EFFECTIVE_BASIS_AUDIT.json`、`EFFECTIVE_BASIS_QUERIES.npz`、`ZERO_ETA_PROJECTED_ACTION_AUDIT.json`、`C2_STATIC_FEATURES.npz`：无rollout的Controller表示与局部有效性证据。
- `C2_DECISION.json`、`C2_FALLBACK_RESULTS.json`、`c2_fits/`：正式门和低成本诊断须分开理解。
- `N2_RESULTS.json`、`N3_SUMMARY.json`及预测：辅助state机制/分布证据。
- `RUN_COMMANDS.md`、实验脚本、`ENVIRONMENT.json`、`ARTIFACT_MANIFEST.json`：重算与恢复入口。

工作目录 `/home/zhihan/research/Basin_C1`。先读REPORT与机器结果，再读所需原始面板；不要从日志摘要或单个Q相关系数推断泛化成功。

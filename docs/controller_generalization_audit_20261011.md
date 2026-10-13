# Controller-conditioned generalization：研究重建与根因审计（2026-10-11）

本文件是独立 handoff 审计，只读取已有证据，并做了零 rollout、零训练的标签重算。
没有修改任何代码、数据、checkpoint 或已有报告。凡标“本审计重算”的数字都用附录脚本从
原始标签独立算出；其余数字引自所列文件，部分已抽查（见 §6）。

## 1. 方法 pipeline（已按代码核实）

| 组件 | 作用 | 证据位置 |
|---|---|---|
| Flow-BC / MACFlow（每场景各自训练的 frozen checkpoint） | nominal 动作 `a_flow = L_X(y_1)`，`y_0 = ξ`，10 步 Euler | `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/REPORT.txt` §1 |
| 硬安全投影 `P_U` | agent-pair/wall CBF 线性约束 + 速度 SOC，欧氏投影 | `shared_control/hard_projection.py:178` |
| OrthoFlow3 correction（η∈R³） | `δa^η = η_goal·B_goal + η_perp·B_flow⊥(a_ref) + η_rel·B_rel`，每物理步重算；B_flow⊥ 缩放 3.3037 | `shared_control/diagnostic_corrector.py:94`、`shared_control/basis_families.py`、PoC REPORT §2 |
| Placement | TT：`P_U(P_U(a_flow)+δ)`（终端修正）；FF：δ 在 pseudo-time 上被 lift 进 Flow ODE，并用 admissible-field 投影 `P_V`，`h_1∈U`。TF/FT 为混合 | `diagnostics/placement_boundary_mechanism_audit_v1/jacobian_probe.py:52-57`；PoC REPORT §1 |
| Episode 级 η | 一个 episode 内 η 固定；成功 = 无碰撞且在 horizon 内到达 | 各 protocol.json |
| Generator | `q(η|h)` 给出 K16 候选；在 controller 研究中保持冻结 | `orthoflow3_controller_intervention_generalization_v1/final_report.md` |
| Critic | `Q(s[,C],η)`：成功概率，用部分计数 Bernoulli NLL 训练，从候选中取 argmax；B15 = ≥15/16 | 同上；`controller_ff_factorial_repair_v1/protocol.json` |
| “Controller” C | 研究中用过三种定义：(a) 第 1 步起换成另一个 frozen Flow checkpoint；(b) 解析控制器 `λ(target−h)`；(c) 五个 affine FF 算子 M（isotropic/anisotropic/rotate±/coupled），叠加在 goal-reference 或 Flow-reference 程序上。当前输入为 compact64 静态场描述 + physical entity/API slots | `controller_ff_factorial_repair_v1/protocol.json` 中的 `controllers` 字段 |

注意：C 的定义随时间漂移（a→b→c）。不同定义下的“controller 泛化”结论不能互相替代。

## 2. 研究历史（按时间）

1. **09-22 → 10-02：success basin 与 η 研究。** 涉及 DB/Ring/Four 的 η3 basin，以及 TT/FF 的 basin 形变。
2. **10-08：placement 强假设检验**（`diagnostics/placement_strong_hypotheses_v1/final_report.txt`）。
   - H1：交换动作族时，标签追随修正 donor，50/52 个分歧点如此。
   - H2：topology 证据为 RESOLUTION_INSUFFICIENT。
   - H3：体积是 reshaping，有增有减。Ring 的 FF−TT 体积分数为 +0.181，6/6 状态为正；Toy 为 −0.128。
   - H4：局部法向与曲率发生了变化。
   - 可写的结论是 reshape 以及 Ring 有界窗口内的 net enlargement；不能写成普遍扩大或 topology 改变。
3. **10-07/08：第一轮 controller 研究**（`orthoflow3_controller_*`）。结论是 PRINCIPLED_NEGATIVE，主要发现：
   - 换掉未来 Flow 会改变 Q。
   - H8/H20/H100 摘要无法给出可迁移的 selection。
   - Ring LOSO：state-aware critic 为 0–1/60，η-only 为 27/60。
4. **10-08：physical-only 状态条件化**（`transferable_state_conditioning_v1`）。
   - joint 训练时 Toy 有真实的 state 价值。
   - LOSO 时所有 selector 都退化为单一 fixed η。
5. **10-09 → 10-10：根因程序，阶段 1–42**（`controller_*`，台账见 `diagnostics/controller_cross_scene_root_v2/TEST_AND_CAUSE_LEDGER.txt` R01–R39）。
   - 修正了多个局部缺陷。
   - 四折 LOSO 一直没有修复。
   - 阶段 35–42 转向“一步 FF 局部动作”诊断。
6. **10-07 → 10-10（并行线）：Gap1/bottleneck_family N 扩展。** 内容包括 Flow 胜任性和 η 覆盖，最新提交为 `a83a76d`，与 controller 泛化是两条独立的线。

## 3. Controller 泛化证据表

“Held-out 类型”的取值：
- S：只换新 state。
- C|scene：同场景内换新 controller。
- Scene(+Flow)：held scene，此时每场景自己的 Flow 也必然是新的，两者混在一起。

| # | 实验 | 配置 / 数据规模 | Held-out 类型 | 关键指标与结果 | 结论 | 路径 |
|---|---|---|---|---|---|---|
| E1 | 未来 controller 交换 | Toy，16 states × 4 η；从第 1 步起换 Flow；1024 条复用 + 1088 条新 rollout | 无（因果干预） | 8/64 个 pair 从 16/16 变为 0/16，9 个通过 Holm 校正；**fixed common η 不受影响**（15/16→16/16） | 已证明：仅凭 h 无法决定 Q | `orthoflow3_controller_conditioning_probe_v1/final_decision.json`、`controller_pair_results.csv`（已重算 8） |
| E2 | 干预训练 + K16 LOSO | 4 场景 LOSO；H8/H20 上下文；7,216 条新 rollout | Scene(+Flow) | Ring：H8 full 0/60，contrast 1/60，η-only 27/60，oracle 60/60。事后 Ring few-shot：用 2 个 family 时 η-only 已到 47/60，state-aware 只有 44–47/60 | 失败；主因是**场景级 η 偏好迁移** | `orthoflow3_controller_intervention_generalization_v1/final_report.md` |
| E3 | 短响应上下文 C1/C3 LOSO | 47,486 个 pair，305 states，3 seeds | Scene(+Flow) | Ring C3 为 1/60，η-only 为 27/60，C0 的预测选中概率 0.9997，而真实 Q 只有 0.019 | 失败；严重过置信 | `orthoflow3_controller_context_loso_v1/final_report.txt`、`primary_summary.csv` |
| E4 | 同场景 held Flow | Ring/Four source families：第三个 Flow、v8 和 v11。v11 为 12 states × 16 η，3,072 条新 rollout | C\|scene | v11：η-only、correct H20、wrong H20 都选中 11/12；correct H20 的 NLL 0.69–0.74，比 η-only 的 0.53–0.56 还差。fingerprint 对 held controller 的 reversal：Four 1/3，Ring 0/8 | 不支持 held-controller selection；而且 headroom 本来就很小 | `orthoflow3_controller_information_probe_v1/final_report.md` |
| E5 | State-residual 析因 | Ring，3 个 Flow，46 个 TRAIN families × 16 η；10,628 条新 rollout | S（同 controller） | H20 full 的 NLL 为 .248，η-only 为 .341，但 B15 两者都是 15/17。把 h 遮掉后结果与 full 相同；H80 更差 | H20 提升了概率，selection 没有提升；h 没有增量 | `orthoflow3_controller_state_residual_factorial_v1/final_report.md` |
| E6 | “state-only ≥ full” 的来源 | PoC，K16，每个 held scene 8 个 states | Scene(+Flow) | B15/32：state-only 为 25/24/24，full 为 24/24/23，η-only 为 22/19/22。Ring 内对 state 做全部 8! 种置换，top1 都不变；换成场景 prototype 后有 23/24 个 run 的 B15 不变 | **SCENE_LEVEL_CONDITIONING_DOMINANT**；其 H20 很可能受 R02 缓存 bug 污染 | `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/field_pipeline_v1/representation_input_test_v1/{decision,mechanism_decision}.json` |
| E7 | H20 缓存 bug 与静态场查询 | 复用 58,752 条 outcome；Ring native 128 states，Q4，2 个已见 Flow；新增 34,816 条确认 rollout（未见解析 C6/C7） | S；C\|scene；单源→Scene | ① bug：108 个 states 中 96 个串了 controller 前缀。② Ring native：matched 97.07%，per-C fixed 95.21%，global fixed 90.82%，state-only 27/32，branch_nominal 31/32。③ 同场景未见 C6/C7 的 Q MSE：Toy .255→.085，Ring .200→.055。④ Toy→Ring 的 MSE 为 .647，η-only 为 .260 | 同场景内泛化到新 state 和新解析 C **成立**（Q 层面）；跨场景失败 | `controller_cached_ff_relearn_v1/ROOT_CAUSE_REPORT.txt`、`development_table.csv` |
| E8 | 信息 × 监督析因 | 4 场景 × 10 C（5 goal + 5 Flow affine），每场景 40 states，K4；15,360 个标签（6,400 新）；75 fits | Scene(+Flow) | LOSO 的 matched−mismatch CI 全部跨 0。full 减 per-C fixed：Toy −.21、DB −.86、Four −.03、Ring −.55。joint Ring：TRAIN corr .955，VAL −.061 | 失败；几乎全部 regret 来自场景级 η（见 §4） | `controller_ff_factorial_repair_v1/DEVELOPMENT_REPORT.txt` |
| E9 | 缓存并集 | 97,280 条 outcome，784 states，60 fits | Scene(+Flow)；C\|scene（Q16 C6/C7） | held DB 的选中成功率：η-only .17，slots .78，state-erased .00。四折 triple skill 全为负 | 并集没有修复；场景级 η 的选择**对建模细节不稳定** | `controller_cached_union_v1/REPORT.txt` |
| E10 | 交叉覆盖 32→128 | 补 7,680 个 K4Q4 标签；36 个新 fit | S；Scene | source Ring VAL 上 Q MSE 降 36%，S×η 误差降 40%（3/3 seeds）。LOSO 四场景 triple CI 全为负 | 局部覆盖是真因；但这不是迁移的原因 | `controller_crossed_coverage_v1/REPORT.txt`、`controller_crossed_coverage_loso_v1/REPORT.txt` |
| E11 | Native 采样质量 | native Ring 的采样质量从 5.3% 恢复到 20%；30 fits | S | native64 成功率 92.09→95.41%，CI [+.68, +6.74]pp。LOSO 未修复 | 局部修复 | `controller_native_mass_repair_v1/REPORT.txt` |
| E12 | Initializer alias | Four/Ring：32 states × 4M × K4；每场景新增 1,024 条 | 输入充分性 | compact64 完全相同，但 Q 下降 8.4pp（969→883/1024）。严格 η 排序反转 0 个（Four）；Ring 有 1 个发现、0 个确认 | 输入确实不充分；**影响主要在 Q 水平，不在排序** | `controller_initializer_alias_q_v1/REPORT.txt`、`controller_initializer_alias_ring_q_v1/REPORT.txt` |
| E13 | 阶段 35–42：一步局部动作 | 160 个 natural states，K4，静态 query，0 条 rollout；每阶段约 12–24 个 fit | Scene | source 端 12/12 为正。held 端：DB/Four 多为负，Ring 偶尔为正。阶段 42 换成 per-agent readout 后，DB/Ring 的 held S×η 转正，**Four 却恶化约 25 倍**（skill −7.4e-4，而真实能量约 1e-5） | 即使函数可从输入精确重建（δ 误差 1e-8），held 几何上也学不出来；指标量级极小，与决策的相关性没有验证 | `controller_local_*`、`controller_correction_*`、`controller_unprojected_mediation_local_v1`、`controller_local_readout_validity_v1/REPORT.txt` |
| E14 | joint-four fresh TEST | 128 个新 TEST states，8,832 条 Q16 | S（joint）；Scene | joint 下 state vs η-only 的 B15：Toy 27/25、DB 32/32、Four 29/30、Ring 27/26。Toy 的 matched−mismatch 为 +8.6pp。LOSO Ring 为 7/32 | 只有 Toy 的 state 价值成立 | `transferable_state_conditioning_v1/STATUS.md` |

**真正涉及 held-out controller 的测试：**
- E4（同场景 held Flow）和 E7/E9（同场景 held 解析 C6/C7）。
- **没有任何测试在跨场景的同时把 controller 与几何分离开**：LOSO 中每个场景使用自己的 Flow checkpoint（`protocol.json` 里 `flow_checkpoint_path` 按场景不同）。
- 共享的 goal-reference 程序在 DB 中全部失败，没有 η 监督信号（R16，以及 §4 中的表）。

## 4. 本审计的重算：LOSO 失败的分解（零 rollout）

数据来自 `controller_ff_factorial_repair_v1/raw_labels.npz` 和 `dataset.npz`，限 Flow-reference 层，每场景 40 states × 5 C × K4。

各场景对每个 η 的平均成功率（取 5 个 Flow 算子的均值）：

| 场景 | η0 | η5 | η10 | η15 | 主要失败类型 |
|---|---|---|---|---|---|
| DB | 0.00 | 0.00 | 0.17 | **0.99** | timeout |
| Four | 0.00 | 0.00 | 0.86 | **1.00** | timeout |
| Ring | 0.25 | 0.12 | **0.79** | 0.18 | timeout |
| Toy | 0.08 | 0.43 | **0.61** | 0.43 | timeout / deadlock |

η 取值：η10 = (1.246, −0.289, 0.186)，即强 goal、负 safe-perp；η15 = (0.529, 0.308, 0.005)，即弱 goal、正 safe-perp、几乎没有 relative。全部 12,800 条 Flow 层 outcome 中碰撞为 0。

- **LOSO η-only 就是“source 场景平均后取 argmax”。** 按这一规则算出的 held 成功率与原报告完全一致：DB .171、Four .998、Ring .180、Toy .434。
- 场景按最优 η 分成两种 regime：{Toy, Ring} 偏好 η10，{DB, Four} 偏好 η15。只要 held 场景属于 source 中的少数派，η-only 就会失败。
- 交叉拟合 headroom（用 seed0 选、seed1 评，再反过来；naive oracle 偏乐观）：

| 场景 | per-scene fixed η | naive state oracle | 交叉拟合 state oracle |
|---|---|---|---|
| DB | 0.992 | 0.995 | 0.958 |
| Four | 0.998 | 1.000 | 0.952 |
| Ring | 0.792 | 0.806 | 0.728 |
| Toy（per-C fixed） | 0.647 | 0.703 | 0.627 |

  也就是说，**state-adaptive 的可得 headroom 不超过 0.003–0.056**，交叉拟合后反而为负；但 LOSO 的 regret 有 0.2–0.86。
- 在 regret（naive oracle 减 LOSO full）中，由“选错场景级 η”造成的比例：DB 约 99%，Ring 约 97%，Toy 约 79%，Four 约 78%。
- Controller 改变 Q 水平，但很少改变最优 η：
  - DB/Four/Ring 的 5 个 Flow 算子最优 η 一致（Four coupled 的 η10≈η15 例外）。
  - Ring 的 goal-rotate 与 goal-isotropic 之间 Q 相差约 0.8，但 η 曲线平坦。

范围限制：
- 只用了 K4 面板，regime 划分可能比 K16 更粗。
- K16 证据方向一致：E2 few-shot；E6 置换不变与 prototype；E14 LOSO 退化为单一 fixed η；Ring native 中 state 相对 per-C fixed 只多出 +1.9pp。

## 5. 判断

### 已解决（有范围）
- **Q 依赖未来 controller（E1）。** 几项输入/实现缺陷是真的，局部也已修复：
  - H20 缓存归属错误（E7）。
  - endpoint alias（R03）。
  - native slot 编码（R13）。
  - native 采样质量（E11）。
  - 交叉覆盖不足（E10）。
- **同场景、新 state、已见 controller：** controller-conditioned 的 Q 和 selection 有真实增益。Ring native 相对 per-C fixed 多 +1.9pp；Toy joint 的 matched−mismatch 为 +0.11。
- **同场景、未见解析 controller：** Q 能泛化（E7）。但这个 family 很窄，Jacobian 是 −λI。
- **Placement：** FF 改变了 robust basin 的几何和体积（Ring 为正，Toy 为负），topology 改变没有证据。

### 未解决
1. 任何 held-scene 上 η 主效应（η regime）的迁移。
2. held-scene 的 state-conditioned adaptation：所有 LOSO 的 mismatch CI 都跨 0。
3. 在有 headroom 的面板上，held controller 的 selection 增益：E4 的面板接近饱和。
4. Flow 类 controller 表示是否充分（E12 已给出反例）。
5. 跨场景时 controller 与几何的可分离性。

### 背景线索的核实
- “state-only 不弱于 full”：来自 E6（PoC）。其 H20 很可能受 R02 bug 影响；之后 E7 用正确的静态查询在 Ring 内得到 branch_nominal 31/32 > state-only 27/32。所以这个现象**只适用于 bug 期 H20 + LOSO 的设定**。
- “H20 无稳定增益”：E4、E5 证实。
- “controller identity 可识别，但不能预测 η 排序”：E4 证实。H20 的 controller 可解码率约 56%（chance 33%）；held-controller 的 reversal 为 1/3 和 0/8。

## 6. 根因假设

**RC1（主假设）：场景级 η-regime 迁移失败，有效“场景样本量”只有 3。**
- 支持：
  - §4 的精确复算。
  - E2 few-shot：2 个 Ring family 就能把 η-only 修到 47/60。
  - E6：置换不变 / prototype 实验显示 SCENE_LEVEL_CONDITIONING_DOMINANT。
  - E14：LOSO 退化为单一 fixed η。
  - E9：held DB 的 η 选择随建模细节在 .00、.17、.78 之间跳动（只有 3 个点的外推）。
  - E3：Ring 上严重过置信。
- 反对 / 局限：
  - Toy 场景内的 state 价值是真的（E14 +8.6pp）。
  - K4 面板偏粗。
  - per-scene fixed 是事后 oracle。
  - 这个假设本身不解释为什么 regime 会切换（机制见 X3）。

**RC2：几何组合的外推失败（表示/归纳偏置 + 几何分布偏移），而不是信息缺失。**
- 支持：
  - 确定性、可从输入精确重建的局部映射（δ 误差 1.8e-8；M 误差 2.4e-7，R29/R39）在 held Four 上仍然学不出来。
  - 阶段 42 只换 readout，就让 DB/Ring 转正、Four 失稳。
  - Ring 的 held 上下文落在 source 包络之外（E2、E3）。
  - 单源跨场景的 Q 比 η-only 更差（E7 ④）。
- 反对 / 局限：
  - 局部动作指标量级为 1e-6，与 Q 决策的关系没有验证。
  - 只有 4 个场景，无法与 RC1 分开（两者都是“几何样本太少”）。
  - readout 结果说明架构也有影响。

**RC3：controller 表示不充分，且 controller 与场景混杂。这一点对精确 Q 是必要的，但不是 selection 失败的主因。**
- 支持：
  - E1、E12 证明了 alias。
  - 但 alias 主要改变 Q 水平，严格反转 0 个确认。
  - §4 中最优 η 在各 Flow 算子间稳定。
  - LOSO 无法隔离 controller，因为每个场景都有自己的 Flow。
- 反对：
  - Ring 在 intervention 数据中有 13 个认证反转。
  - Four fingerprint 在已见 controller 上能拿到 6/7 个反转。
  - 所以 controller 引起的排序变化确实存在，只是稀疏。

## 7. 现有实验能否区分，缺什么证据

**能区分的：**
- RC1 与“交互学习失败”：§4 已证明 regret 主要来自主效应。

**不能区分的：**
- RC1 与 RC2：需要很多个几何，而不是更多 state。
- RC3 与 RC1：缺少两类证据：
  - (i) 跨场景共享且有 η 信号的 controller 程序；
  - (ii) 预先登记 headroom 的 held-controller 面板。

**方法问题：**
- 阶段 35–42 的局部指标在决策相关性上没有设 gate。
- 多阶段反复使用已打开的 DEVELOPMENT 面板，存在分叉路径（forking paths）风险。
- `status.json` 仍写着阶段 42 “submitted”，但它实际上已完成：`completion.json` 中 `focused_gate_passed=false`。

## 8. 最有判别力的下一步

| 编号 | 实验 | 成本 | 能排除或确认什么 |
|---|---|---|---|
| X0 | 对所有已有 LOSO 面板（E2/E3/E8/E9/E10/E14，含 K16）重算 regret 分解：per-scene fixed、per-C fixed、交叉拟合 oracle；检查 LOSO 模型选的是否就是 source 多数派 regime | 0 rollout，0 fit | 在 K16 上确认或否定 RC1。如果 K16 上的 state headroom 也很小，说明继续优化交互指标不能修复 LOSO 的 success |
| X1 | “只修主效应”诊断：冻结 source 模型的 state/C 残差，只用少量 target 标签估计场景级 η offset（每个 η 一个 logit 偏置），看 held 端 state×η 残差 skill | 少量 fit，0 rollout（已有标签） | 把“主效应错”与“交互也不迁移”分开。如果修正主效应后残差交互仍为负，RC2 也成立 |
| X2 | 几何族扫描（bottleneck_family）：固定 N、固定 controller，让 gap 宽度/长度等连续变化，做 ≥10 个几何 × 少量 states × 同一 K 面板；然后在内插和外推两种设定下做 leave-geometry-out | 中等 rollout；纯 Q 数据，不需要视频；须做 preflight | 主效应在内插几何上可预测、外推不行 → 证实 RC1，瓶颈是几何覆盖。几何多了内插仍然不行 → RC2 为主 |
| X3 | 机制连接：从已有 trace 计算每个场景 × η 的进展/阻塞统计量（投影活跃比例、constraint-active 时间、通过瓶颈的时间、timeout 时的残余距离），检验某个低维物理量能否跨 4 个场景预测哪种 regime 有效 | 0 rollout（前提是 trace 已存） | 把“controller/修正如何改变闭环进展”与 η 偏好连接起来，给 RC1 提供可迁移的物理条件变量；再用 X2 验证 |
| X4 | 预先登记 headroom 的 held-controller 面板：先用 TRAIN 标签挑出最优 η 随 controller 变化的 controller 对，要求 oracle − best fixed ≥ 阈值，再打开 held 标签，比较 correct 与 wrong descriptor | 小规模 rollout | 判断 RC3 对 selection 是否真有影响。如果 descriptor 只改善校准、不改善排序，RC3 就只关乎 Q 精度 |

建议暂停阶段 35–42 这类局部动作诊断。原因是这条线的 state×η 能量约 1e-5、skill 约 1e-6，量级远小于决策相关的 regret。如果要继续，必须先证明局部改进能预测 Q 或 regret 的改进。

## 9. 仓库与可复现性风险

- 全部 `diagnostics/controller_*`（阶段 1–42 的证据和台账）都**未纳入 git**（`git status` 显示 untracked）。
- FF 算子、Flow checkpoint 和 `field_pipeline_v1` 位于 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`，这个目录**不是 git 仓库**。
- 按 AGENTS.md，应先 commit/push 可版本化的代码、配置和文档，大型数据另行备份。本审计没有执行这一步。

## 11. 验证实验结果（2026-10-11 第三轮）：§10 部分被证伪

**实验与报告：** `diagnostics/controller_geometry_identifiability_v1/REPORT.md`。

**设计：**
- 预注册的最小实验：在 Double-Bottleneck 中只改变 bottleneck 宽度（0.72–1.62，共 14 个宽度）。
- 5 个 goal-reference 算子，16 个新 state，K4 η 面板。
- 共 13,440 次 main rollout，另有 1,280 次 pilot。
- 在原宽度下逐字节复现存档记录；0 次碰撞；全部 merge 进 shared DB。

**结果：**
- **成立：** controller 和 η 的作用随几何系统性变化（A1）。
  - 远端宽度之间的算子模式相关 ≈ 0。
  - 探索性分析中，η 偏好在 w = 0.90 与 w = 1.26–1.32 之间反转。
- **成立：** 增加环境内数据不能替代几何多样性：FEW×2 ≈ FEW < MANY。
- **不成立：** rotate 效应在这个几何族里没有翻转符号，只是幅度变化约 6 倍。
- **被证伪：** “几何少 → 不可辨识 → 外推失败”。
  - 学习器能看到几何坐标、且 controller 身份精确时，3 个几何就能迁移（B2 0.80）。
  - 单侧外推同样能迁移（B2 0.62–0.64；选择等于 oracle）。

**修正后的主假设（待直接检验）：** 四场景失败的关键在于缺少一个被表示出来、各场景共享的几何/机制坐标，而不是环境数量少。

**局限：** 该几何族几乎不依赖 state（state 能量占 0–6%），比四场景简单得多。

**下一步：** 在同一宽度族上测试项目实际的 entity-encoder critic（不需要新 rollout），用来分离“表示”与“共享坐标”两种解释。

**后续（同日）：项目 entity-encoder critic 检验（R1 成立）。**
- 设置：几何只通过墙体 entity 进入，不给显式宽度。
- 结果：外推 B2 为 0.63，与显式宽度的 MLP 相同；MANY 0.91，FEW 0.75。
- 结论：在有共享坐标的几何族中，项目表示不是瓶颈。
- 已排除的候选解释：
  - 场景数量少；
  - 环境内数据不足；
  - controller 信息不足；
  - 项目 entity 几何表示。
- 剩余的主要假设：
  - 四个场景不在同一个几何/机制坐标上；
  - state 条件化本身的难度（本几何族的 state 相关能量只占 0–6%，没有检验到）。
- 详见 `diagnostics/controller_geometry_identifiability_v1/REPORT.md` §6。

**根因综合（同日第四轮）：条件不允许，而不是方法存在硬伤。**

*证据 1：方法在条件满足时能泛化。* 以下情况均已验证：
- 场景内对新 state 的泛化；
- 场景内对新解析 controller 的泛化；
- DB 宽度族上跨几何外推，包括项目 entity critic 只看墙体的情况。

*证据 2：四场景的条件。*
- 4 个环境在约 15 个场景级因素上同时不同。
- 每个 LOSO fold 中，能与 3 个源场景一致的物理描述子，对 held 场景的预测为：正确 6、错误 15、无法确定 6、源场景本身不可分 33。
- 没有任何描述子在 4 个 fold 中都给出正确预测（`scene_multiplicity.json`）。

*修正后的根因表述：* 可辨识性取决于“环境数”与“同时变化的因素数”之比，而不是环境数本身。

*仍未排除：*
- 跨几何的 state 条件化；
- 场景各自 Flow controller 的嵌套。

**方法缺陷检验（同日第五轮，`diagnostics/controller_state_transfer_v1/REPORT.md`）**
- 设置：Toy 走廊宽度族，24 个训练 state + 8 个新 state，K8 η，13,312 次 rollout，预注册。
- Q 模式层面，没有发现方法缺陷：
  - 项目 critic 在留出宽度上的 state×η 模式相关为 0.88（查表上界为 0.85）；
  - 新 state × 新宽度为 0.68（几何内新 state 为 0.60，上限 0.79）；
  - D1、D2、D3 都不成立。
- 新 state 上的交互幅度（m2）弱，受 24 个 state 的样本量限制。
- 选择层面无法评估：宽度 ≥ 0.44 时，固定的 η2 对所有 state 都是 100% 成功，自适应选 η 的可得收益为 0。
- 仍未检验：存在 state 排序反转时，选择决策能否跨几何迁移（例如 Ring 的 obstacle_radius 族）。

**以下 §10 保留为历史判断；其中“几何数量不足是根因”的部分以本节为准。**

## 10. 根本原因（同日第二轮分析，取代 §6 的并列表述）

**脚本与输出：** `diagnostics/controller_fundamental_cause_audit_20261011/{analysis.py,results.txt}`。0 rollout、0 训练，数据源为 `controller_ff_factorial_repair_v1`。

**结论：** “controller→Q 关系”不是 controller 本身的属性，而是 controller 与几何在长时域闭环中耦合出来的性质。这个耦合由“该几何里哪种失效机制先发生”决定，在不同场景里形式不同、甚至反号。数据只覆盖 3–4 个几何，每个几何对应一种不同的主导失效机制，而且 controller 嵌套在几何里。因此，可迁移的关系在数据中**不可辨识**。黑箱终端结果回归得到的是同样与数据一致、但最简单的“每个场景一张表”。§6 的 RC1/RC2/RC3 是这一点在不同层面的表现，并不是三个并列的根因。

### 证据链
1. **构造上，controller 只是交互项。** 五个算子都是 endpoint-matched：δ=0 且无约束时，执行动作完全相同。它们只改变修正怎样被执行：
   - rotate± 把执行修正旋转 ∓45.4°；
   - anisotropic 把纵向增益压到 0.2、横向放大到 0.707。
2. **交互在场景之间不一致甚至反向，而且这不是噪声。**
   - 同一个 analytic goal 基底 + 同一个算子：rotate± 在 Ring 上成功率 0.92/0.89，在 Toy 上 0.00，在 Four 上 1.00。
   - Toy–Ring 算子效应模式相关 −0.94（不相交的 state 半分）；场景内 state 半分复现 0.97–0.996。
   - Flow 基底：场景间相关 −0.51 到 +0.28；场景内 0.90–0.998。
   - split-half 无偏方差分解：controller 相关能量中只有 19%（Flow）/ 12%（goal）跨场景共享。scene×η（0.056）与共享 η 主效应（0.068）同量级。
3. **不一致的来源是主导失效机制不同。**
   - Toy：让行死锁，deadlock 占 16–29%。
   - DB：瓶颈堵塞。η10 有 63% 的 rollout 触发候选死锁；η15 能过，但中位成功步 773/850。
   - Ring：绕行时间不够。η15 中位成功步 496，η10 为 408，horizon 700。
   - Four：路口交汇。η0/η5 全部 timeout。
   - 候选解释（事后提出，只与 3 个有信息的场景一致）：自由空间里存在绕障碍的环路时（Ring、Four），旋转修正形成环流而成功；在直走廊（Toy）里则失败。
4. **学习器只能学成查表。** source-only full 模型在 held 场景上：
   - η 曲线照搬 source 多数派（Ring 预测 η15 最好，真值是 η10）；
   - 算子效应模式与真值相关 4/4 为负（−0.04 到 −0.27）；
   - state×η 在 Toy/Ring 被压扁（0.0019 vs 0.0297；0.0053 vs 0.0178），在 Four 则凭空制造（0.0020 vs 0.0002）。
5. **缺的是一个低维的场景级量。** 只用 k=1 个目标 state：DB 从 0.17 升到 0.99，Ring 从 0.18 升到 0.72；k=4 时 Ring 0.79，接近 oracle。
6. **即时行为不是充分统计量。**
   - controller 对的首步动作差与 |ΔQ| 的 Spearman：DB 0.007，Ring 0.004。
   - |ΔQ|≥0.5 的 pair 中约一半首步差低于中位数，最小只有 0.003·vmax。
   - E1 中首步完全相同，结果仍从 16/16 变为 0/16。

### 为什么 42 个阶段都修不好
它们全部是**环境内**干预：state、seed、覆盖、精度、描述子、oracle 中间量、loss、路由、readout。环境内的辨识信息增加了，所以 source 指标改善。但“几何层面只有 3 个样本、关系随几何反号”这一点没有变，所以 LOSO 不变。这对应相对于表示的 concept shift（不是 covariate shift）；不变性学习所需的训练环境数，也远多于 3 个（Rosenfeld et al., ICLR 2021）。

### 边界
- 证据来自 K4 面板、两类 affine 家族、4 个场景。
- “环流 × 环路”是事后解释，没有独立验证。
- DB 的 goal 程序全部为 0，不提供信息。
- deadlock 监测只在 Toy/DB 存在，各场景事件语义不完全一致。

## 附录：§4 复算脚本（numpy 即可）

```python
import numpy as np
d='diagnostics/controller_ff_factorial_repair_v1'
r=np.load(d+'/raw_labels.npz',allow_pickle=True); z=np.load(d+'/dataset.npz',allow_pickle=True)
S=r['success'].astype(float); O=r['observed'].astype(float)
K=[i for i in range(S.shape[1]) if O[:,i].sum()>0]          # [0,5,10,15]
P=S[:,K].sum(-1)/np.maximum(O[:,K].sum(-1),1)
scene=z['scene']; flow=np.char.startswith(z['controller'],'flow')
prof={s:P[(scene==s)&flow].mean(0) for s in np.unique(scene)}
for h in prof:                                              # LOSO eta-only = source-pooled argmax
    k=int(np.mean([prof[s] for s in prof if s!=h],0).argmax())
    print(h, K[k], round(prof[h][k],3), 'best fixed', round(prof[h].max(),3))
```
解释器：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。

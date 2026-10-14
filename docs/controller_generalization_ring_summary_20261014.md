# Ring 场景 controller 泛化研究总结（2026-10-11 至 2026-10-14）

本文汇总 2026-10-11 到 2026-10-14 共 17 轮实验，回答一个问题：**critic 能否学会 Q(s, C, η) 中 controller 的作用，从而在没见过的 controller 上零样本选好动作？**

各轮细节、数值与预注册收据都在对应的 `diagnostics/` 目录中（见第 9 节索引）。本文只引用这些报告中的数字。

**证据等级：**
- “前瞻”指选择规则或模型在新标签产生之前已冻结，并在新 controller、新 state 上评估。
- 其余结论都是 DEVELOPMENT（同一数据集内的发现或交叉验证）。

---

## 0. 一页结论

**1. 根因：动作空间的压缩（H_compress）。** 在“整局固定一个 η”的设定下，一个 controller 的 η 偏好是两个阶段效应之差的符号：

N = F13 − F10 = E − L

其中：
- E = S500 − F10：前期 η13 带来的收益；
- L = S500 − F13：后期换回 η10 带来的收益；
- S500 表示前 500 步用 η13、之后用 η10。

这两个效应在大多数 controller 上都很大（14 个原 controller 中，E > 0 的有 11 个，L > 0 的有 14 个），而它们的差的正负会随 controller 的细节翻转。因此以下手段都识别不出 controller 的“regime”，零样本选择失败：
- 静态描述；
- 200 步行为探测（只用于识别 regime）；
- 少样本的源先验。

**2. 解决方法：** 动作改为“前期 η13、第 t 步换 η10”的时间表（菜单中也保留固定 η）；controller 用 200 步无标签行为探测 R5 表示。

**3. 三次前瞻确认（均在 Ring）：**

| 新 controller 的来源 | 数量 | 主要结果 |
|---|---|---|
| 同 run 兄弟 checkpoint（round12） | 8 | 平均遗憾 1.42pp |
| 同配方、新 seed（round14A） | 14 | 平均遗憾 0.61pp |
| 新架构配方（round17） | 13 | 平均遗憾 1.70pp |

三次都没有灾难。相对旧的固定 η10 先验，成功率平均高出约 27–30pp（round12、round14A）。

**4. controller–Q 关系的学习：**
- 前期分量 E 可以从 R5 学到。在外部 controller 上的 Spearman 为 0.74–0.96。
- 通用 critic（无约束 MLP，输入为 R5 的 PCA-2）在新配方上的遗憾，比不看 controller 的 critic 低 4.23pp，95% CI 为 [−5.45, −2.96]。
- 这个 critic 能识别“换架构后 regime 翻转”的 controller（v10_local 的宽网络版），而手工规则 G_E 在这个 controller 上损失 48pp。

**5. 尚未完成：**
- 罕见的 η2 型 regime：round17 中最大遗憾 6.36pp，超过预注册的 5pp 门；
- 全新数据集配方（N1）的检验没有区分度；
- 只在 Ring 上验证；
- 跨场景失败是另一个根因（见第 6 节）。

---

## 1. 研究目标与范围

- **目标：** critic 学会 controller 与 Q 的关系，在未见过的 controller 上零样本选好动作。这个目标由用户在 2026-10-11 设定。
- **管线：** 冻结的 Flow 给出 nominal 动作，经过硬安全投影，再加上 OrthoFlow3 修正。修正由 η = (goal, safe-perp, relative) 三个权重决定，每步重算。placement 为 FF。
- **主要场景：** ring_exchange，H700（700 步），dt = 0.05，4 个机器人。
- **controller：** 14 个真实 Ring Flow 训练 run（`diagnostics/ring_exchange_stage1/*_macflow/`），以及后续由它们衍生的 checkpoint（见第 3 节）。
- **η 取值（K16 中的三个）：**
  - η10 = (1.246, −0.289, 0.186)；
  - η13 = (1.075, 0.058, 0.266)；
  - η2 = (1.142, −0.112, 0.535)。

---

## 2. 最终结论（按证据强度）

### 2.1 根因：H_compress

| 证据 | 内容 | 等级 |
|---|---|---|
| round9 切换实验 | 4 个 controller × 64 个 state，在 t = 100–500 双向切换 η。η13 的收益需要早期暴露（base_u 与 v10_local 的半效应时刻 t_half = 100），η13 的伤害在后期产生（v9_local t_half = 400，plus_w_v4_wide t_half = 500）。“先 η13 后 η10”同时优于两个固定 η：base_u 84.0% vs 53.1% / 23.4%；wide 96.9% vs 26.2% / 63.3% | DEVELOPMENT |
| round10 | 14 个 Flow × 64 个新 state。S500 比每个 controller 的最佳固定 η 平均高 26.1pp；在从未参与时间表选择的 10 个 Flow 上高 30.2pp，CI [12.7, 47.6] | 前瞻（时间表由 round9 选定）；判定为 INCONCLUSIVE（v9、v3 低约 5–6pp） |
| 分解分析（补充预注册：预测在 round10 结果出来之前写定并留收据） | P1、P2 通过：E > 0 为 11/14，L > 0 为 14/14；E、L 都 ≥ 10pp 的 controller 有 10/14。扩展动作集下，零样本平均遗憾从 32.9pp 降到 1.0pp | 预测为前瞻，数据为 DEVELOPMENT |

**它能解释此前的所有失败：**
- **静态描述 R3：** 它编码的是训练配方。配方相近的 v9_local 与 v10_local，E 与 L 的相对大小正好相反（round7 I4：“配方”预测 SUPPORTED）。
- **200 步探测 R5：** 它测到的是所有 Flow 共有的前期加速（7 个面板 Flow 的 Δprog 全部 > 0），测不到决定净偏好的后期脆弱性（round7：ρ(Δprog, ΔS) = −0.30）。
- **少样本中源先验不省标签（round8）：** N 的符号在 controller 之间不共享。
- **“给 controller 信息反而有害”（round5/6）：** 在 14 个 controller、目标本身是一个压缩符号的条件下，无约束的 controller 通路会任意外推。这是目标病态的症状。动作空间修正、数据充足之后，无约束 critic 不再有害（round16/17 中 A3 的灾难数都是 0）。

### 2.2 方法

**部署流程（零样本，不需要目标的成功标签）：**
1. 对新 controller，在 16 个固定 probe state 上，分别用 η10、η13 各运行 200 步，共 32 次短探测、6,400 个模拟步；只记录 5 个过程统计量，不看成败。
2. 按冻结的预处理得到 R5 特征，再取 PCA-2。
3. 用以下任一方式选动作：
   - **规则 G_E：** 用 ridge 预测 Ê；Ê < 5pp 选全程 η10，否则选 S500；
   - **通用 critic A3：** 64/64/16 的 MLP，输入为 [state 物理特征, R5-PCA2, 动作]，输出成功概率；在评估 state 上平均后，为每个 controller 选一个固定动作。动作菜单为 F10、F13、F2、S200–S600。

### 2.3 前瞻确认

| 轮次 | 新 controller | 规则 | 主要结果 | 判定 |
|---|---|---|---|---|
| round12 | 8 个 final.pkl（同 run 兄弟；另有 6 个参数与 best 完全相同，已排除） | G_E | 相对 F10 +26.8pp [23.3, 30.3]；平均遗憾 1.42pp [0.78, 2.64]；controller 信息相对“全部 S500”+2.49pp [1.42, 3.56]；Ê 的 Spearman 0.83，RMSE 比 0.45 | C1、C2、C3 通过；C1b 未通过（v10_final 遗憾 7.0pp，需要更早切换） |
| round14A | 14 个独立 seed（原 seed + 10000，用历史训练代码） | G_E | 平均遗憾 0.61pp，最大 3.9pp；相对 F10 +29.6pp；headroom 5/14；Ê 的 Spearman 0.74，RMSE 比 0.34 | C1、C2、C3 通过；C1b 为 INCONCLUSIVE（unknown 区间跨过 5pp） |
| round17 | 13 个新架构配方（512×4），另有 4 个新数据集配方 | A3 critic（冻结于 round16） | A3 平均遗憾 1.70pp，A0 为 5.93pp，G_E 为 5.45pp；A3 相对 A0 −4.23pp [−5.45, −2.96]；无灾难 | K1、K3 通过；K2 仅因最大遗憾未通过（v3_arch512 遗憾 6.36pp，它是 η2 型） |

### 2.4 controller–Q 关系的学习（round15–17）

- **round15（56 个独立 Flow、57,344 条标签）：**
  - E、L、N、t* 的配方 ICC 分别为 0.90、0.89、0.81、0.66。同一配方的不同 seed 几乎是重复样本，所以对新配方而言，有效样本数等于配方数。
  - 学习曲线：增加同配方的 seed，不能改善对新配方的预测（H_cov_E、H_cov_L 均 REJECTED）。
  - 对已知配方的新 seed，L 可学：R6，用 3 个 seed 训练，RMSE 比 0.53。
- **round16（通用 critic）：**
  - 预注册的主模型 A1（在 A0 上加有界残差）未通过 Q1–Q3。
  - 作为描述性对照的无约束 A3，在训练集内交叉验证与三个外部测试集上，平均遗憾分别为 1.37 / 0.56 / 0.75 / 0.54pp（按 controller 选择），与 G_E 的 1.27 / 0.89 / 0.64 / 1.42pp 持平或更好。
- **round17（对 A3 的前瞻确认）：**
  - critic 会按 controller 选不同的切换时刻：base_u、v2 的宽网络版选 S600，v9 的宽网络版选 S200，都正好是各自的最佳动作。
  - v10_local 的宽网络版发生了 regime 翻转：F10 80%、F13 30%、S500 31%。A3 选 S200（遗憾 4.7pp），G_E 选 S500（遗憾 48.4pp）。
  - 由 critic 推出的 E 与真实 E 的 Spearman 为 0.96，L 只有 0.39。
- **逐 state 选择：** 在多数测试中没有超出 controller 级选择的价值（round4、round8、round16）。可迁移的主要是 controller 级的动作。

---

## 3. 证据链（按时间顺序）

| 轮次 | 问题 | 主要结果 | 目录 |
|---|---|---|---|
| 审计 | 四场景 LOSO 为何失败 | 场景级 η regime 主导：{Toy, Ring} 偏好 η10，{DB, Four} 偏好 η15。“场景太少是根因”被 DB 宽度族实验否定 | `docs/controller_generalization_audit_20261011.md`、`controller_geometry_identifiability_v1`、`controller_state_transfer_v1` |
| R2 | Ring 有没有 state 级的选择空间 | Ring FF 有：Q512 达到 93.54%，最佳固定 η13 为 88.33%，差 +5.21pp [2.65, 7.88] | `controller_eta_selection_db_audit_v1` |
| R2b | 学选择需要多少数据；placement 与 Flow 是否改变排名 | 16-head MLP 约需 128 个 state；FF 与 TT、旧 Flow 都会改变 η 的排名 | `ring_state_selection_transfer_v1` |
| R3 | 14 个真实 Flow 普查 | 7 个处于成功率窗口内；留出 controller 的最佳 η 等于源先验，检验没有区分度 | `controller_q_transfer_round3_v1` |
| R4 | 真实 Flow 的 LOCO | regime 分布：η10×9、η13×2、η2×2、η4×1；只有 2/7 折需要 controller 信息；Stage C 中 R3 为 +5.21pp [−0.66, 11.08]，未确认 | `controller_loco_real_flow_v1` |
| R5 | 融合 Flow 族 | regime 随 λ 切换；R3 / R_oracle 出现 −30 至 −58pp 的灾难；DENSE 比 SPARSE 更差 | `controller_flow_blend_v1` |
| R6 | 有害迁移是否源于方法缺陷 | 低维化、有界残差、悲观选择消除了灾难；真实 Flow 上没有安全的正迁移（v10_local +17.6pp，但 v9_local −10.2pp） | `controller_pathway_fix_v1` |
| R7 | 瓶颈是表示还是覆盖 | 四格全失败；静态描述编码配方而非 regime；Δprog 与 ΔS 负相关 | `controller_behavior_repr_v1` |
| R8 | 少样本适应 | 预设候选菜单 + 安全规则，在 128 条标签时达到 SPT / CT；源先验与 critic 都不省标签 | `controller_fewshot_adaptation_v1` |
| R9 | η 效应何时产生 | 见 2.1；发现“先 η13 后 η10”的时间表 | `controller_eta_switch_time_v1` |
| R10 | 时间表是否普适 | 见 2.1 | `controller_universal_eta_schedule_v1` |
| 分解 | H_compress | 见 2.1 | `controller_phase_decomposition_v1` |
| R11 | 后期探测 R6 能否学到 L | 排序相关 0.79，RMSE 比 0.77，未过 0.7 门；融合族检验无区分度 | `controller_late_phase_probe_v1` |
| R12 | 前瞻确认 I | 见 2.3 | `controller_new_checkpoint_confirmation_v1/fp32_n8` |
| R13 | 跨场景时间表 | X1、X2 均 REJECTED（见第 6 节） | `cross_scene_eta_schedule_v1` |
| R13B / 14A | 前瞻确认 II | 原 seed 的复现门未通过，原因未明；预注册的备用门通过；见 2.3 | `ring_independent_flow_confirmation_v1`、`_v2` |
| R14B | Four 场景的 controller 普查 | 只有 1 个 Four Flow 与 E8 runtime 兼容，结论 INCONCLUSIVE | `four_controller_census_v1` |
| R15 | controller 总体 | 见 2.4 | `ring_controller_population_v1` |
| R16 | 通用 critic | 见 2.4 | `ring_controller_critic_v1` |
| R17 | 前瞻确认 III | 见 2.3 | `ring_critic_new_recipe_confirmation_v1` |

---

## 4. 否定或未成立的方向（负结果）

| 假设或方法 | 结论 | 轮次 |
|---|---|---|
| 场景太少是跨场景失败的根因 | 否定（DB 宽度族用 3 个宽度即可迁移） | 审计 |
| η 选择在各处都只是场景级问题 | 否定（Ring FF 有 state 级空间） | R2 |
| 静态描述 R3 编码 controller 的 regime | 否定（编码的是训练配方） | R4–R7 |
| 200 步探测 R5 能识别 regime | 否定；但 R5 能预测前期分量 E | R7、分解分析 |
| 增加锚点邻域的融合 controller 能解决覆盖问题 | 否定 | R7 |
| 源先验或 critic 在少样本下能节省标签 | 否定 | R8 |
| η 效应只在后期决定（H_late） | 部分成立：收益在早期产生，伤害在后期产生 | R9 |
| 当前物理状态能替代 controller 身份（H_state_mediated） | 否定（AUC 0.60 vs 0.78） | R9 |
| 后期探测 R6 能把 L 学准 | 否定（只学到排序） | R11、R15 |
| 同一个时间表能消除跨场景 regime | 否定 | R13 |
| 增加同配方的 seed 能改善对新配方的泛化 | 否定 | R15 |
| 用 ridge 预测切换时刻的规则优于 G_E | 否定 | R15 |
| 有界残差 critic 在数据充足时仍是最佳结构 | 否定（无约束 A3 更好） | R16 |

---

## 5. 我的判断失误与纠正

1. **审计中主张“场景太少是根因”：** 被 DB 宽度族验证否定，已改写结论（审计 §11）。
2. **猜测“η 选择在各处都是场景级”：** 被 Ring FF 的 DB 审计否定，已在旧计划文档中标注。
3. **round5 前后，把 fold4 中 R3 的 controller 先验当成了“选 η13”：** 实际是 η4，round5 时已纠正。
4. **round7 的结论映射初稿，把“R5 不能识别 regime”等同于“覆盖不足”：** 这是逻辑错误，下发 prompt 之前已改正。
5. **round11 的融合族检验没有先检查 headroom：** 结果没有区分度。此后的确认实验都加了 headroom 门。
6. **round13B 复现门失败后，判断原因是训练代码的版本漂移：** round14A 证明不是（用历史代码重训，偏差相同），真正原因仍未明。
7. **round14B 假设 Four 的其他 Flow 与 E8 runtime 兼容：** 实际只有 1 个兼容。
8. **round16 把有界残差的 A1 设为主模型：** 在数据充足、动作空间正确之后，约束本身成了瓶颈。无约束的 A3 才是好模型，但这是事后发现的，因此加了 round17 来确认。

---

## 6. 跨场景轴：另一个根因

**round13**（`cross_scene_eta_schedule_v1`）：在四场景的 FF runtime 上，比较 3 个 Flow 算子 × 40 个新 state，在固定 η 与时间表下的表现。

各场景在同一阶段需要的 η 互相冲突：

| 场景 | 情况 |
|---|---|
| Toy | 前期必须用 η10。前期用 η13 或 η15 会引发让行死锁（真实死锁判据） |
| DB | η15 要用到后期。第 364 步就换成 η10 会堵住（27%）；第 607 步才换则可以（99%） |
| Four | η10 与 η15 都接近 99%，η13 会导致大量超时 |
| Ring | 时间表优于固定 η |

- **结果：** Ring 上确认的时间表在 Toy、DB 上分别比最佳固定 η 低 15.4pp、31.9pp。加入时间表后，LOSO 平均遗憾从 68pp 降到 44pp（Four 从 97pp 降到 0.2pp），但 DB 和 Ring 仍然很差。
- **结论：** 跨场景失败不是动作压缩造成的。它源于真实的“场景 × η”机制冲突，再加上源场景太少、场景信息不可辨识，与审计的判断一致。改动作空间不能解决这个问题；需要获得场景信息，例如用同一族的大量几何训练，或在新场景上做探测或少样本适应。

---

## 7. 局限与未解决问题

**范围**
- 只在 Ring（FF、H700）上验证。标签多为 Q4，不是 B15 认证。

**确认的独立性**
- 前瞻确认用的 controller 都与训练配方有关联：同 run 兄弟、同配方的新 seed、同数据集的新架构。
- 唯一的全新数据集组 N1（plus_drift，4 个 controller）在几乎所有动作下都接近 100%，没有区分度。

**罕见 regime**
- η2 型 controller 在训练集中很少，round17 中出现了 6.36pp 的最大遗憾。

**外部测试集的性质**
- round16 的 T1–T3 标签此前已被打开过，只能算“对冻结模型的外部测试”，不是全新的前瞻确认。

**训练复现性**
- 原 Ring Flow 无法逐位复现（dev loss 相差约 6.7%），原因未明。后续训练依赖“seed 间波动”这一备用门。

**unknown 的处理**
- 数值失败一律记为 unknown，按下界计为失败，不重跑。部分门因 unknown 区间跨过阈值而判为 INCONCLUSIVE。

**视频**
- 多个评估时间表的 batch 缺少正式的死锁角色（Ring 与 Four 没有死锁判据）。按 AGENTS.md，这些 batch 的状态是“数值完成、视频交付不完整”。

**其他场景的 controller 轴**
- Four 的 21 个 Flow 与 E8 runtime 不兼容，需要改用原生 runtime（每个 Flow 用各自的归一化器）才能检验。

---

## 8. 建议的下一步

| 方向 | 内容 | 成本 |
|---|---|---|
| B. 补 η2 缺口 | 把 round17 的 17 个 controller 并入训练集，再训练新的 η2 型候选 controller（v3、plus_w_v3 配方的其他架构或 seed），做一次前瞻测试 | 约 1–2 万条 rollout |
| C. 其他场景的 controller 轴 | 用 Four 的原生 runtime 普查它的 21 个 Flow，检验 H_compress 与“探测 + critic”能否推广 | 中等，需先解决兼容问题 |
| D. 跨场景轴 | 另一个课题，见第 6 节 | 较大 |
| 资产管理 | 本研究的 prompt 文档与大部分 `diagnostics/` 目录尚未纳入 git；其中一些目录有数 GB。PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/` 不是 git 仓库，里面有 FF operator 与 Flow checkpoint。建议按 AGENTS.md 提交代码、配置与文档，大型数据另行备份 | 低 |

---

## 9. 索引

**实验目录**（均位于 `diagnostics/`）：见第 3 节的表格。

**各轮 prompt：**
- `docs/prompt_round3_controller_q_20261011.md`
- `docs/prompt_round4_loco_real_flow_20261012.md`
- `docs/prompt_round5_flow_blend_20261013.md` 至 `docs/prompt_round9_eta_switch_time_20261013.md`
- `docs/prompt_round10_universal_eta_schedule_20261013.md`
- `docs/prompt_round10_addendum_phase_decomposition_20261013.md`（含收据）
- `docs/prompt_round11_late_phase_probe_20261013.md`
- `docs/prompt_round12_new_controller_confirmation_20261013.md`
- `docs/prompt_round13_cross_scene_schedule_20261013.md`
- `docs/prompt_round13b_independent_ring_flows_20261013.md`
- `docs/prompt_round14a_independent_ring_flows_v2_20261014.md`
- `docs/prompt_round14b_four_controller_census_20261014.md`
- `docs/prompt_round15_ring_controller_population_20261014.md`
- `docs/prompt_round16_controller_critic_20261014.md`
- `docs/prompt_round17_critic_new_recipe_confirmation_20261014.md`

**更早的审计与计划文档：**
- `docs/controller_generalization_audit_20261011.md`
- `docs/next_experiments_*_20261011.md`

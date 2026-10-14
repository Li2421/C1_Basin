你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不生成任何新 rollout 或探测。** 所有“目标标签”都从已有标签中抽取，以模拟采集过程。这是纯 Q 实验，不需要视频。

# 一、目标与本轮问题

研究总目标：让 critic 学会 Q(s, C, η) 中 controller 的作用，从而在**新 controller** 上选好 η。

round4–7 的零样本路线已经基本走到头（全部为 DEVELOPMENT 证据）：

- **round6：** 约束 controller 通路后，“给 controller 信息反而有害”的问题被消除。
- **round7**（`diagnostics/controller_behavior_repr_v1/`）：真实 Flow LOCO 的“表示 × 覆盖”四格都没有达到安全正迁移。
  - **零样本无法识别 regime：** 静态 R3、200 步行为探测 R5，以及加入锚点邻域的融合覆盖，都不能在零样本下识别真实 Flow 的 η regime。
  - **ρ(Δprog, ΔS) 为负：** 14 个真实 Flow 上为 −0.30，CI [−0.85, 0.35]。7 个面板 Flow 的 Δprog 全部 > 0，因此按 Δprog 选 η13 的零参数规则 P_rule 全选 η13，E3 = 5/7。短时进展预测不了长时成功。
  - **必要 controller 上没有收益：** 安全规则全部回退到 η10，在 base_u 和 v10_local 上收益为 0。
  - **决策本身是 controller 级的二选一：** η13 对 base_u 是 +30pp，对 v10_local 是 +26pp，对 v9_local 是 −16pp，对其余面板 Flow 也有害。
  - **逐 state 选择在新 controller 上有害：** R0 的逐 state argmax 比固定源先验 η10 低约 5pp（54.9% vs 60.2%）。
  - **Q 拟合变好不等于选择变好：** R5-PCA2 的 Q 拟合 NLL 比 R0 更好（0.370 vs 0.426），但选择没有变化。

由于只有 2/14 个真实 Flow 属于 η13 型，零样本在现有 controller 总体上无法辨识 regime。“增加独立的 η13 型 controller”仍为 INCONCLUSIVE，而且需要训练新的 Flow，成本高。

本轮回答一个可部署的互补问题：**面对一个新的真实 Flow controller，至少需要多少条有标签的目标 rollout，才能安全地选对 η？源 controller 上学到的知识（regime 先验、critic）能把这个数量降低多少？**

最后一个问题是本轮对研究目标最直接的度量：critic 或源先验学到的 controller–Q 结构，体现为它们相对“只用目标标签”所节省的样本数。

# 二、必读材料（亲自核对关键数字）

1. `diagnostics/controller_behavior_repr_v1/`：REPORT.md、HANDOFF.md、results.json。
2. `diagnostics/controller_pathway_fix_v1/`：REPORT.md，以及 F2 与 R0 的冻结 checkpoint。
3. `diagnostics/controller_loco_real_flow_v1/`：HANDOFF.md、REPORT.md、LABELS.npz、protocol.json、`fits/`、`controller_prior/`。
4. `diagnostics/controller_q_transfer_round3_v1/`：P1 数据，即 14 个 Flow × 24 个 state × K16 × Q4。
5. `diagnostics/controller_flow_blend_v1/`：
   - STAGE_0_RESULTS.json，含 14 个 Flow 的 TRAIN 曲线；
   - stage_2 中的融合标签；
   - REPORT 中“目标少样本校准参照”表（这是 round5 用固定前 k 个 state 做的粗略版本，本轮取代它）。
6. `docs/prompt_fewshot_controller_adaptation_20261013.md`：本任务的早期草稿，未执行。以本文件为准。

# 三、数据划分（先审计身份与 unknown，写进预注册）

真实 Flow 面板为 round4 Stage A 的 7 个 Flow，用 7 折 LOCO 划分；必要折为 base_u 和 v10_local。

**目标标签池（模拟采集只能从这里抽取）：**
- 目标 controller 在 128 个 loco_train state 上的已有标签：8 个掩码 η 头 × Q4；
- 目标 controller 在 P1 的 24 个 state 上的已有标签：K16 × Q4。

**评估：**
- 只用目标的 64 个 loco_test state × seed 4–7。
- 目标自身的 hindsight 最佳固定 η 用 loco_test 的 seed 0–3 选出，只作参照。

**源知识：** 只能用 6 个源 Flow 的标签与冻结模型。源模型是该折 round6/round7 已冻结的 REAL6 checkpoint。

**描述性扩展：** 对另外 7 个 Stage B Flow 做留一法：目标池为 loco_train 标签，评估用 P1 的 24 个 state。因为评估与源知识共用 P1，结果只作 DEVELOPMENT 描述，不进入主判定。

# 四、设计（在任何新分析之前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 预算与重复

- 预算 B（目标 rollout 条数）∈ {0, 4, 8, 16, 32, 64, 128}。
- 每个“预算 × 策略 × 目标”重复 200 次随机抽样，随机种子预先固定。
- 每次抽样不放回；某格已有标签用尽时，记为“无法采集”并报告，不得伪造。
- unknown 标签按配对保守规则处理：既不算成功，也不算失败。

## 采集策略

| 代号 | 策略 |
|---|---|
| U | 在可用的 (state, η, seed) 格上均匀随机抽取 |
| R | 只在源 regime 集合 {η2, η4, η10, η13} 上均匀抽取（这个集合来自源知识） |
| A | 自适应：用方法 H 的后验做 Thompson 抽样，选择下一条标签的 η；state 在可用格中随机 |

## 适应方法

| 代号 | 方法 | 说明 |
|---|---|---|
| P0 | 源先验 | 固定选 η10（7 折的源先验都是 η10），对应 B = 0 |
| E | 只用目标标签 | 对每个 η 用 Beta(1,1) 后验，取后验均值最大的 η；没有标签的 η 不参与；平局时依次取源先验、编号最小者 |
| H | regime 混合先验 | 以 6 个源 Flow 的 K16 成功率曲线（源 TRAIN 标签）为混合分量，等权先验。用目标标签计算各分量的后验权重，再与目标自身的 Beta 计数合并，取后验期望最大的 η。分量曲线用 Beta(1,1) 平滑 |
| C | 冻结 critic + 目标偏置 | 用该折冻结的 R0（以及 F2 R3-PCA2 REAL6）的逐 state 预测，加上每个 η 的目标偏置：offset_η = n_η/(n_η+8) × (目标残差均值)。按 state 取 argmax |
| C-ctl | C 的 controller 级版本 | 把 C 校准后的预测在评估 state 上取平均，选一个固定 η。用于区分逐 state 选择与 controller 级选择 |

**安全版本：** E、H、C、C-ctl 都另有一个安全版本（加后缀 -safe）。只有当所选 η 优于源先验 η10 的后验概率 ≥ 0.9 时，才偏离 η10。
- E 与 H 用各自的后验计算这个概率；
- C 与 C-ctl 用 3 个 fit 一致且平均预测优势 ≥ 0.10 作为判据（同 round6 的保守规则）。

所有超参数（Beta(1,1)、n0 = 8、0.9、0.10、200 次重复、预算网格）都在预注册中写定，不得根据终点结果调整。

## 终点（只用目标 loco_test 的 seed 4–7）

对每个“方法 × 策略 × B”计算以下指标。CI 要把 state 成组 bootstrap 和 200 次抽样的变异都纳入；同一预算下所有方法共用一个 max-T 同时 CI 族。

| 编号 | 指标 |
|---|---|
| E1 | 7 个目标的平均选择成功率相对源先验：非劣界 −2pp |
| E2 | 必要 controller（base_u、v10_local）相对源先验的收益：逐个报告，并给出合并值 |
| E3 | 灾难率：比源先验低 ≥ 10pp 的目标所占比例。在 200 次抽样上取平均，同时报告“任一次抽样发生灾难”的比例 |
| G | 距目标 hindsight 最佳固定 η 的差距，以及已闭合的比例 |

**判定定义：**
- **安全正迁移（SPT）：** E1 过门、E3 = 0，且 E2 合并 CI 下界 > 0。
- **完整迁移（CT）：** SPT 成立，且 base_u 与 v10_local 的收益点估计都 > 0。

## 样本效率指标（主分析）

- **B_SPT / B_CT：** 每种“方法 × 策略”首次达到 SPT / CT 的最小预算。若在 128 以内都达不到，记为 > 128。
- **B_2pp：** 每个目标达到“距 hindsight 最佳固定 η ≤ 2pp”所需的最小预算。必要目标与非必要目标分开报告。
- **源知识的价值（本轮的核心度量）：**
  - H vs E、C vs E 的预算节省，即 B_SPT 与 B_CT 之差；
  - 每个 B 下成功率的配对差，附同时 CI。
- **逐 state 选择的价值：** C vs C-ctl。逐 state 选择在少样本下是否有额外收益，还是像零样本时一样有害。

# 五、结论映射（预注册，必须按此下结论）

| 观察结果 | 结论 |
|---|---|
| H 或 C 的 B_SPT 明显小于 E（同时 CI 支持） | **源知识可迁移为样本效率**：critic 或源先验学到了可用的 controller–Q 结构。零样本失败只是 regime 不可辨识，而不是没有学到结构。可部署路线是“源先验 + 少量目标标签” |
| H ≈ E，C ≈ E，但都能在小预算下达到 SPT | 少样本可行，但源知识没有带来额外价值；controller–Q 结构的学习在本设定下尚未体现 |
| 所有方法在 128 以内都达不到 SPT | 即使有目标标签，安全选择也很昂贵。报告瓶颈是噪声（Q 估计方差）还是 regime 判别本身 |
| C > C-ctl | 逐 state 选择在少样本下有价值 |
| C ≤ C-ctl | 可迁移的部分是 controller 级 regime，逐 state 选择没有价值（与 round4 的结论一致） |

此外报告：如果用零样本的 R5 探测来分配采集预算（R5 每个 controller 6,400 个模拟步，不带标签），等价于多少条目标 rollout。这只作描述。

# 六、硬性规则

- **不得新增 rollout 或探测，不得写入共享 DB。** 读取 DB 一律用只读或 immutable 模式。
- **文件写入范围：** 只在新目录 `diagnostics/controller_fewshot_adaptation_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **seed 隔离：** 任何选择、调参或校准都不得使用 loco_test 的 seed 4–7；hindsight 参照只能用 seed 0–3。
- **证据等级：** 所有 state 都已被打开过，结论一律标为 DEVELOPMENT。
- **JAX x64：** 记录并匹配原设置（false）。
- **资源：** 本 workstation 上用户所有任务合计不得超过 18 个 CPU shard 和 3 个 GPU shard。提交前先查看实时占用。

# 七、交付

全部交付物放在 `diagnostics/controller_fewshot_adaptation_v1/`：

- PREREGISTRATION.md 及收据；
- 代码；
- 标签池审计：每个目标可采集的格数、unknown 数，以及每个 η 的覆盖情况；
- results.json；
- 学习曲线图：成功率与 E3 随 B 的变化，按方法 × 策略画，必要目标与非必要目标分开；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 必须做到以下几点：

1. 回答：新 controller 需要多少条目标 rollout 才能安全选对 η，必要目标与非必要目标分开报告。
2. 回答：源知识（H、C）节省了多少目标标签，并按第五节的结论映射给出结论。
3. 回答：少样本下，逐 state 选择是否有价值。
4. 对研究的含义：
   - controller 泛化是否应改为“源先验 + 少样本快速适应”的路线；
   - 零样本还剩什么值得做，例如训练新的独立 η13 型 controller 的成本是否值得。
5. 每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。事实、解释和猜想分开写，不夸大。写明结论的边界：Ring、FF、K16、固定几何，以及只有 2 个必要的真实 controller。

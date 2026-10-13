你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。

- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

# 一、研究目标

解决 controller 泛化：让 critic 学会 Q(s, C, η) 中 controller 的作用。对一个训练时**从未见过的真实 Flow controller**，critic 应能：
- 只用部署时可算的信息（物理 state，以及对 controller 的静态查询）；
- 为每个 state 选出 η；
- 其闭环成功率优于不依赖该 controller 信息的选择。

**本轮唯一的主问题：** 在“确实需要 controller 信息才能选对 η”的留出 controller 上，哪种 controller 表示（如果有）能带来可靠的选择收益？

# 二、为什么这样设计（必须理解，不得改变其逻辑）

上一轮（`diagnostics/controller_q_transfer_round3_v1/`）的结论：

1. **P1 普查：真实 Flow 会大幅改变 η 决策。** 7 个合格 Flow 之间，用 A 的最佳固定 η 去执行 B，损失的中位数为 17.2pp，CI [3.1, 28.4]；state 层的交叉遗憾中位数为 27.6pp。
2. **P2′ 失败的原因是测试设计，不是方法。**
   - 留出的 TF/FT 与源 controller 的最佳固定 η 都是 η13，源的合并先验本来就能选对，controller 信息没有用武之地。
   - 本轮必须只在“源合并先验对留出 controller 是错的”的折（fold）上评价 controller 表示。
3. **原定的真实 Flow P2 没有执行。** 合格 Flow 只有 7 个，“3 个留出、其中 2 个在凸包内”的方案不可行。

本轮的对应改法：
- 改用 7 个合格 Flow 的留一法（LOCO，leave-one-controller-out），共 7 折。
- 对每一折先做“controller 信息必要性门”，只有通过门的折才计入主检验。

**已知的方法事实**（无需重新验证，直接采用）：
- 16-head MLP 在单个 controller 内约需 128 个 state。
- P0：同等标签预算下，“256 个 state × 8 个 η × Q4”的分配最好；checkpoint 以 NLL 选择即可。
- R3（compact64）对不同的执行算子存在混淆（alias）；R4_action 能区分。
- R4_openloop（P3 的短时静态探测）只改善了 Q 拟合，没有改善选择。

# 三、必读材料

1. `diagnostics/controller_q_transfer_round3_v1/` 中的以下文件：
   - `HANDOFF.md`、`REPORT.md`、`P1_RESULTS.json`
   - 代码：`flow_adapter.py`（真实 Flow 的 local/global 观测适配）、`state_generator.py`、`p1_run.py`、`p2prime_features.py`、`p3_features.py`、`p2prime_train.py`、`p2prime_analyze.py`、`p2prime_design.py`
   - 运行记录：`P1_EXECUTION_PROTOCOL.json`、`P1_CHECKPOINT_COMPATIBILITY.json`
2. `diagnostics/ring_state_selection_transfer_v1/`：其中的 `HANDOFF.md`、`FLOW_CHECKPOINT_INDEX.json`
3. `shared_rollout_db/README.md`：preflight、journal、单进程 merger 的规则。
4. `docs/next_experiments_controller_q_round3_20261011.md`：背景。

**代码复用原则：** 与 P1、P2′ 相同的执行语义、Flow 适配、特征与模型代码，一律复用，不要重写。新代码只放在新目录中。

# 四、实验设计（在生成任何新 outcome 之前，冻结进 PREREGISTRATION）

## 1. Controller

使用 P1 中固定 η 成功率落在 [40%, 95%] 的 7 个 Ring Flow checkpoint：

`base_u_plus_w_v4_wide_macflow`、`base_u_macflow`、`base_u_plus_w_v3_macflow`、`base_u_plus_w_v4_train_targeted_macflow`、`base_u_v10_local_macflow`、`base_u_plus_w_v4_macflow`、`base_u_v9_local_macflow`。

执行条件全部沿用 P1：自由 t0、FF、OrthoFlow3、硬安全投影、H700、评价器不变，K16（必须包含 η13），seed 语义与 P1 相同。每个 checkpoint 的 execution identity 和 JAX x64 设置都要逐一记录。

## 2. State（全新的自然 state，所有 controller 共用同一批）

- 用 `state_generator.py`，以新的随机序列，从原生开发分布中抽取。
- 先做哈希检查，证明这些 state 与所有历史 state（包括 512 个训练 state、256 个开发 state、P1 的 24 个 state）都不重合。
- 划分为：
  - **S_train**：128 个 state；
  - **S_test**：64 个 state。
- 7 个 controller 共用这两组 state。这样同一 state 上不同 controller 的结果可以配对比较，而 S_test 对 critic 来说一定是没见过的 state。

## 3. 标签

**S_train（每个 controller 都要采集，用作源训练标签）：**
- 每个 state 取 8 个 η，在 K16 中做分层随机分配，使每个 η 被选中的次数相同；
- Q4；
- 每个 controller 4,096 次 rollout。

**S_test（每个 controller 都要采集，用于该 controller 被留出时的评价）：**
- K16 × Q8。seed 0–3 用于选固定 η 参照和计算必要性门，seed 4–7 只用于终点评价；
- 每个 controller 8,192 次 rollout。

**合计：** 约 86,016 次新 rollout。真实 Flow 平均约 6.5 秒一次，约 155 CPU 小时。

**验证集（只用于 checkpoint 选择）：** 复用 P1 已有的 24 个开发 state 标签（K16，Q4），只用源 controller 的部分，不新增 rollout。

## 4. 留一法与必要性门（每折独立计算）

对第 k 折（留出 controller k，用其余 6 个作源）：

- **源先验 η_src：** 在源 controller 的 S_train 标签上，先在每个 controller 内对 state 取平均，再对 6 个 controller 取等权平均，然后选 Q 最高的 η。
- **留出 controller 自身的固定 η：**
  - 在 S_test 的 seed 0–1 上选 η，在 seed 2–3 上评价；
  - 再用 seed 2–3 选、seed 0–1 评价；
  - 两个方向的结果取平均。
- **必要性损失：**
  - L_k =（留出 controller 自身固定 η 的交叉拟合成功率）−（η_src 在 seed 0–3 上的成功率）。
  - 按 state 做 bootstrap（2000 次），unknown 取保守边界。
- **门：** L_k ≥ 5pp 且 95% CI 下界 > 0，记为“必要折”。
- **补充报告（不作为门）：** state 层的必要性，即留出 controller 的 state-oracle 交叉拟合成功率减去 η_src 的成功率。

不是必要折的结果照常描述性报告，但不计入主检验。**必要折少于 2 个时，主检验判为 INCONCLUSIVE**，并报告 7 个 controller 之间偏好结构的描述。

## 5. Critic 与 controller 表示

**模型：** physical23 加 controller 表示，输入 16-head MLP。
- 只对有标签的 head 计算 loss。
- 用源验证集的 NLL 选择 checkpoint。
- 每种表示训练 3 个 fit seed（17、23、41）。
- 归一化只使用源数据。
- 训练步数与优化器沿用 P2′。

**表示**（同一网络、同一数据、同一预算，只改变 controller 输入）：

| 名称 | 内容 |
|---|---|
| R0 | 不给 controller 信息 |
| R3 | compact64 |
| R4_action | 与 P2′ 同法：η = 0 时的投影执行动作，加投影后的 Jacobian（在 t0 和预先固定的 probe state 上计算） |
| R4_openloop | P3 的静态短时探测特征 |
| R3+R4_action | 两者拼接 |

所有静态特征只能通过对 controller 函数的静态查询得到，代码中封锁 `env.step`。

**规模：** 7 折 × 5 种表示 × 3 seed = 105 次拟合，通过 Slurm 使用 GPU shards。

## 6. 终点（只在 S_test 的 seed 4–7 上评价，按 state 成组配对 bootstrap，至少 2000 次）

对每个留出 controller，各表示的 critic 在每个 state 上选出 η，计算其闭环成功率，并与以下对照比较：

| 对照 | 定义 | 性质 |
|---|---|---|
| (a) η_src | 源合并先验对应的固定 η | 零样本 |
| (b) 最近源 controller 的固定 η | 先在源归一化后的 R4_action 空间中，计算各 controller 的特征均值（在 S_train 上取平均），按欧氏距离找到最近的源 controller，取它在 S_train 上的最佳固定 η | 零样本 |
| (c) R0 critic 的选择 | 不使用 controller 信息 | 零样本 |
| (d) 留出 controller 自身的固定 η | 用 seed 0–3 选出 | 只作上界参照，使用了目标标签 |
| (e) 少样本校准参照 | 用留出 controller 自己的 S_train 标签中 k = 1、2、4 个 state，重新估计最佳固定 η（同时报告 critic 加 offset 校准的版本） | 该数据已存在，不增加成本；标明“使用了目标标签” |

**错配对照（matched − mismatched）：** 把留出 controller 的表示依次换成其他 6 个 controller 的表示，取平均后与正确表示的结果比较。

**主判据**（只汇总必要折，在 state 层配对 bootstrap，对 R 与 (a)/(b)/(c) 三项比较以及 4 种表示做同时 CI 校正）。表示 R 被判为“可迁移”，需要同时满足：
- R − (a) > 0，R − (b) > 0，R − (c) > 0，同时 CI 下界都 > 0；
- matched − mismatched 的 CI 下界 > 0。

**另外报告（但不能作为成功依据）：**
- 每一折的结果；
- Q MSE 与 controller×η 模式相关；
- 各表示与 (e) 的差距：零样本能达到几个目标 state 的校准效果。

## 7. 条件分支（预先写定）

| 分支 | 条件 | 做法 |
|---|---|---|
| Stage B：增加源 controller | 必要折 ≥ 2，但没有任何表示达到主判据 | 把另外 7 个兼容但不在成功率窗口内的 Flow 加作额外源（只采集 S_train，约 28,672 次 rollout）重新训练，检验“源 controller 数量增加能否修复”。只评价原来的必要折 |
| Stage C：确认 | 有表示达到主判据 | 在必要折对应的留出 controller 上，另采 64 个全新 state（K16 × Q8），用已冻结的模型和规则做一次确认 |
| 不再追加 | 必要折 < 2 | 不运行 Stage B/C，转为描述性结论 |

# 五、硬性规则

**数据复用与预注册**
- 实验前先检索历史证据，复用兼容标签。execution 语义不同的标签不得混用。
- 每个阶段在读取新 outcome 之前，写好预注册（门槛、样本、划分、指标、分支），保存 UTC 时间和 sha256 收据。之后的任何修改都另行记录，不得覆盖原文件。

**rollout 流程**
- 顺序：study design → `python -m shared_rollout_db.plan` 做 preflight → 只提交 genuinely missing 的部分 → worker 写 raw 与 journal → 单进程 merge → postflight。
- 登记 DB 的 experiment 行时，必须在 merge 之前完成（上一轮在这里出过错）。
- 数值失败保留为 unknown：不重跑，不当作失败。配对比较时，同一 state、同一 η 的 unknown 要正确地两边对消。

**禁止事项**
- 不删减 η 来制造收益。
- 不在 S_test 的 seed 4–7 上做任何选择、调参或校准。
- 不把 Q 相关或 Q MSE 当作选择收益的证据。

**资源与文件**
- CPU rollout 最多 24 个并行（每个 worker 1 个 CPU）；GPU 训练通过 Slurm shard。
- Stage A 新 rollout 上限 90,000；Stage B 上限 30,000；Stage C 上限 10,000。
- 正式提交前，先用约 100 次 rollout 测量实际单次耗时。若推算 Stage A 在 22 CPU 上超过 10 小时墙钟时间，按预注册把 S_test 降为 48 个 state（其余不变）；不得用其他方式缩减。
- 只在新目录 `diagnostics/controller_loco_real_flow_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- 本实验是纯 Q 实验，按 AGENTS.md 不需要视频。
- 每个模型和运行时的 JAX x64 设置都要记录并保持一致。

# 六、交付

在 `diagnostics/controller_loco_real_flow_v1/` 中交付：
- `PREREGISTRATION*.md` 及收据；
- 代码、raw、labels、静态特征；
- 每折、每表示、每个 seed 的 checkpoint 与预测；
- `results.json`、`REPORT.md`；
- `HANDOFF.md`，要求不依赖本对话也能读懂。

`REPORT.md` 必须做到：

1. **逐阶段状态。** 写明 Stage A/B/C 各自是完成、门未通过，还是未执行及其原因。
2. **controller 关系表。** 给出 7 个 controller 的最佳固定 η、交叉遗憾，以及每折的必要性损失 L_k 和是否为必要折。
3. **直接回答主问题。** 在必要折上，哪种表示（如果有）可靠地优于 η_src、最近 controller 和 R0；错配描述是否会变差；与少样本校准相比效果如何。
4. **结论分类。** 按下表给出结论，并说明这一结论对“controller 泛化”目标意味着什么：

   | 结论 | 判定条件 |
   |---|---|
   | (i) 迁移成立 | 至少一个表示达到主判据 |
   | (ii) 必要但所有表示失败 | 方法或表示存在缺陷 |
   | (iii) Stage B 才成功 | 瓶颈在源 controller 覆盖 |
   | (iv) 必要折不足 | 无法检验 |

5. **严谨标注。** 假设用 SUPPORTED / REJECTED / INCONCLUSIVE 标注。事实、机制解释和猜想分开写，不夸大。Q 预测与 η 选择分开报告。

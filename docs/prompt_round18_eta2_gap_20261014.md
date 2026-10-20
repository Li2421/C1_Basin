你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，以及总结报告 `docs/controller_generalization_ring_summary_20261014.md`，并严格遵守 AGENTS.md。

# 一、要回答的问题：补上 η2 型 regime 的缺口

Ring 上 controller 泛化的方法已经确立：时间表动作 + 200 步无标签行为探测 R5 + 通用 critic A3。它已在三批新 controller 上得到前瞻确认（详见总结报告）。

剩下唯一的系统性缺口是 **η2 型 controller**，即最佳动作为全程 η2（F2）的 controller：
- **round17**（`diagnostics/ring_critic_new_recipe_confirmation_v1/`）中，base_u_v3_macflow__arch512 的 F2 成功率为 96%，A3 选了 S300（89%），遗憾 6.36pp。这是 K2 最大遗憾门未通过的唯一原因。
- **训练数据里几乎没有 η2 型样本**（设计者已只读核对，你需要复核）：round15 的 56 个 controller 中，只有 2 个的最佳动作是 F2，领先幅度只有 0 和 +0.6pp。v3、plus_w_v3 配方的新 seed 大多偏好时间表。
- **η2 型的出现很零散**：v3 原 controller（round10）是 η2 型，v3 的宽网络版是，但 v3 的多数新 seed 不是。
- **探测看不到 η2：** R5 只记录 η10、η13 下的行为，不观察 controller 对 η2 的反应。

**本轮检验两种补救，用析因设计判定各自的作用：**
- **覆盖：** 在训练集中加入更多 v3 家族（v3、plus_w_v3 配方）的 controller，包含不同架构与 seed；
- **表示：** 在探测中加入 η2，得到 R5η2 特征。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 A：新 controller（GPU）

**训练设置：** 使用 round15 / round17 已验证的历史训练包 `git_43b5264`、GPU、兼容性门、FP32 修复，以及 round14A 的配方架构修复规则。

**v3 家族（24 个）：**
- 配方：v3、plus_w_v3；
- 架构：256×3、384×3、512×4；
- seed：原 seed + 50001 至 +50004；
- 共 2 × 3 × 4 = 24 个。

**非 v3 对照（12 个）：** 其余 12 个配方，架构 384×3，seed 为原 seed + 50001，每个配方 1 个。

**预注册划分（写入预注册，按以下规则执行，不得根据结果调整）：**
- **TRAIN-new（18 个）：**
  - v3 家族中 seed + 50001、+50002 的 12 个；
  - 非 v3 对照中，按配方名字典序排在奇数位的 6 个。
- **TEST（18 个）：**
  - v3 家族中 seed + 50003、+50004 的 12 个；
  - 其余 6 个非 v3 对照。

**检查：** 每个新 checkpoint 的参数 hash 都必须与所有已有 checkpoint 不同。

## 阶段 B：无标签探测（在任何新成功标签之前完成并冻结）

**对象：**
- round15 的 56 个 controller；
- round17 的 17 个 controller；
- 本轮的 36 个新 controller。

**探测：**
- 与 round15 相同的 R5：16 个 probe state × {η10, η13} × 200 步；已有的直接复用，不重跑。
- 新增 η2 探测：同样的 16 个 state，用 η2 运行 200 步，统计量定义与 R5 相同。

**特征：**
- R5-PCA2（与 round16 的预处理相同）；
- R5η2-PCA2：155 + 80 维的同类特征。z-score 与 PCA 只用训练 controller 拟合；缺失格的处理规则与 round7 / round15 相同。

**冻结：** 特征冻结后留收据。

## 阶段 C：TRAIN-new 的标签（训练数据）

- 18 个 TRAIN-new controller × 32 个全新 state × 8 个动作（F10、F13、F2、S200、S300、S400、S500、S600）× seed 0–3，共 **18,432 条**。
- 先过 32 例一致性门；按 shared_rollout_db 流程执行（design → preflight → 只跑缺失 → journal → 单进程 merge → postflight）。
- **在 critic 冻结之前，不得读取或产生 TEST 的任何标签。**

## 阶段 D：critic（训练、冻结、留收据）

架构与超参数一律沿用 round16 的 A3：64/64/16 MLP，AdamW，lr 0.0008，12,000 步，fit seed 17 / 23 / 41，用训练集内部按配方划分的验证集以 NLL 选 step，JAX x64 = false。

| 代号 | 训练数据 | controller 表示 | 作用 |
|---|---|---|---|
| C0 | round15 | R5 | round16 冻结的 A3，原样使用，作为参照 |
| C1 | round15 + round17 + TRAIN-new | R5 | 只增加覆盖 |
| C3 | round15 + round17 | R5η2 | 只改表示 |
| C2（主） | round15 + round17 + TRAIN-new | R5η2 | 覆盖 + 表示 |
| A0 | 与 C2 相同 | 无（不看 controller） | 基线 |

**选择规则：** 与 round17 的主规则相同，即先在评估 state 上平均预测值，再为每个 controller 选一个固定动作。逐 state argmax 作为次要规则。

**冻结要求：** 对 18 个 TEST controller，冻结每个模型给出的动作并留收据，然后才能进入阶段 E。

## 阶段 E：TEST 标签（前瞻）

- 18 个 TEST controller × 48 个全新 state（与阶段 C 的 state 不同，且与 DB 去重）× 8 个动作 × seed 0–3，共 **27,648 条**。
- 流程同阶段 C。

**本轮总规模：** 约 46,080 条完整 rollout，另加 32 例一致性门与短探测。

# 三、预注册终点与判定

**η2 型的定义（用 TEST 标签判断）：** F2 的成功率比所有非 F2 动作中最好的那个高 ≥ 3pp。

**headroom 门：** TEST 中至少 3 个 η2 型 controller。不满足时，B1 记为 INCONCLUSIVE（没有 headroom），其余判定照常进行。

所有 CI 都按 state 成簇做 bootstrap，使用配对比较。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| B1（主） | η2 缺口是否补上 | 在 η2 型 TEST controller 上，C2 的平均遗憾低于 C0（配对 95% CI 上界 < 0），且 C2 在这些 controller 上的最大遗憾 ≤ 5pp |
| B2 | 是否达到完整标准 | 在全部 18 个 TEST controller 上，C2 的平均遗憾 ≤ 2pp，最大遗憾 ≤ 5pp，且没有任何 controller 比 F10 低 10pp 以上 |
| B3 | 是否没有退化 | 在非 η2 型 TEST controller 上，C2 的平均遗憾 ≤ C0 的平均遗憾 + 0.5pp |
| B4 | controller 信息是否仍有价值 | C2 的平均遗憾低于 A0，配对 CI 上界 < 0 |

**析因分析（描述性，附 CI）：**
- 覆盖的作用：C1 对 C0；
- 表示的作用：C3 对 C0；
- 二者是否互补：C2 对 C1、C3。

**额外报告（描述性）：**
- v3 家族中 η2 型的出现率，按架构与 seed 分组，用来刻画 regime 的脆弱性；
- 各模型给 η2 型 controller 选了什么动作；
- η2 探测特征与“F2 − 最佳非 F2”的相关。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| B1、B2、B3、B4 都通过 | η2 缺口补上；在新架构配方上达到完整的预注册标准。用析因结果说明起作用的是覆盖、表示还是两者 |
| B1 通过，B2 不通过 | η2 型改善了，但仍有其他 controller 超出 5pp。报告是哪些 controller、属于什么类型 |
| headroom 门满足，但 B1 不通过 | 在这两种补救下，η2 型仍学不到。依据析因结果判断哪一种更接近有效，并提出下一步，例如在动作菜单中加入以 η2 收尾的时间表 |
| headroom 门不满足 | η2 型太罕见，无法检验。报告它的出现率；从研究意义上看，它对整体选择的影响有限 |

# 五、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/ring_eta2_gap_v1/` 中写文件（包括新 checkpoint）。不修改、不删除历史资产，不做 git commit。
- **冻结时序：**
  - 特征必须在任何新标签之前冻结；
  - critic 与 TEST 上的选择，必须在任何 TEST 标签产生之前冻结；
  - 每次冻结都留收据。阈值、划分与模型定义不得根据结果调整。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。唯一允许的改变，是在第 t 步替换 η。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **证据等级：** TEST 上的结果是前瞻的。v3 家族的 TEST 与 TRAIN-new 共享配方与架构（只有 seed 不同），报告中必须写明这一点。
- **视频：** 本轮是纯 Q 函数的数据生成与 critic 评估，按 AGENTS.md 的例外不运行视频入口。这一判断在预注册中写定。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。提交前查看实时占用。

# 六、交付

全部交付物放在 `diagnostics/ring_eta2_gap_v1/`：
- PREREGISTRATION.md 及收据；
- 训练、新颖性与兼容性审计；
- 36 个 checkpoint 及其训练日志；
- 探测特征与冻结收据；
- 两个阶段的 state 去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- critic 与 TEST 选择的冻结收据；
- results.json；
- 图表：各模型在 η2 型与非 η2 型 TEST controller 上的遗憾、逐 controller 选择、η2 探测特征的散点图；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：η2 缺口是否补上（B1）、是否达到完整标准（B2），以及起作用的是覆盖还是表示。

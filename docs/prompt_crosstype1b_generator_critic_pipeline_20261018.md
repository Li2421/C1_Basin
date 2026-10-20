你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。本轮是"跨 type"系列的跨type1b，排在跨type2 之前。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`（已有 jax、flax、optax；不安装任何新包）
- 本轮目录：`diagnostics/crosstype1b_generator_critic_pipeline_v1/`

**写作约定：** 本文件中的 controller、做法和终点都先写全称，括号中给出文件里的代号，代号只用来定位文件与身份。你的报告也照此书写：代号第一次出现时给出全称，之后正文用全称。

# 零、本轮要做什么（用户 2026-10-18）

跨type1 用的是一个**学习型选择器**：根据 controller 的探测特征，在几套固定参数或切换规则中挑一套，同一个 controller 在所有 state 上都用同一套。它不是我们论文的方法。

本轮要在跨type1 的同一个学习型动物园上，**把我们真正的 pipeline 跑出来**：
- **生成器（generator）** q(η | h) 对每个 state 给出候选修正；
- **critic** Q(h, C, a) 预测"在 state h、controller C 下执行候选 a 的成功概率"，对每个 state 在候选中取最大者执行。

两者的差别：
- 真正的 pipeline **按 state 选**，候选来自生成器，不限于事先写好的几套参数；
- critic 同时看 state 与 controller。

本轮是**开发轮**，允许迭代 critic 的结构、输入与候选的组成，每次迭代都写进 `ITERATION_LOG.md`；冻结后在全新 state 上只评估一次。

# 一、先读、边界与源码保护

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）** 第一至六节全部适用，不得放宽。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. **pipeline 的定义与冻结部件：**
   - `docs/controller_generalization_audit_20261011.md` 第 1 节（pipeline 各部件与代码位置）；
   - `diagnostics/orthoflow3_generator_critic_v1/`：`REPORT.md`、`ARCHITECTURE_AUDIT.md`、`train_evaluate.py`（`dist_params`、`eta_mean`、`eta_from_noise`、`log_prob`、生成器与 critic 网络）、`normalization.json`、`generator/`（冻结的生成器：seed 41，第 1,900 步，`checkpoint.msgpack`）、`critic/`（冻结的 critic：seed 23，第 2,700 步）；
   - `diagnostics/orthoflow3_generator_critic_frozen_test_v1/run_frozen_test.py`（`FrozenNewRuntime.conditioning`：为新 state 计算 h 与环境描述 c，h 由观测与一个固定的参考 Flow 动作组成，不依赖被测 controller）；
   - `diagnostics/orthoflow3_ring_k16_diagnostic_v1/`（Ring 的 16 候选诊断）、`diagnostics/orthoflow3_c1_generalization_v1/eval_interaction_ring_k16.py`（Ring 上生成器与 critic 的用法示例）；
   - `diagnostics/ring_controller_critic_v1/`（round16：看 controller 的 critic，输入为 physical23、R5-PCA2 与动作；`fit.py`、`freeze.py`、`evaluate.py`、`REPORT.md`）；
   - `docs/controller_generalization_ring_summary_20261014.md` 第 0–2 节（为什么动作要从固定 η 扩展到"先 η13、后 η10"的时间表）。
4. **跨type1（`diagnostics/crosstype1_learned_zoo_selector_v1/`，只读）：** `REPORT.md`、`HANDOFF.md`、`GAPS.md`、`ZOO_MANIFEST.csv`、`features.py`、`assemble.py`、`initial_actions.py`、`selector.py`、`FROZEN_PIPELINE.json`、`DEVELOPMENT_BUNDLE.npz`、`LOCKED_BUNDLE.npz`、`DEVELOPMENT_FEATURES.npz`、`LOCKED_FEATURES.npz`、`DEVELOPMENT_SELECTION_VALUE.json`、`CONTROLLER_SELECTION_DETAILS.csv`、`AXIS_TABLE.csv`，以及它复用 round37/38 执行链的方式。
5. round35 的按进度切换实现 `diagnostics/axis4a_geometry_horizon_v1/progress.py`；round38 的 `diagnostics/blackbox_families_v1/`（`TUNING_STATES.json`、`EVAL_STATES.json`、`episode.py`）。

**源码保护：** 新代码与输出只写在本轮目录。已有源码、`docs/`、`AGENTS.md`、其他全部 diagnostics 目录（包括跨type1 与 `orthoflow3_generator_critic_v1`）、`datasets/`、PoC 目录 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004`（不在 git 中）、MAC-Flow 官方仓库与 venv 一律只读；需要改动的实现按共同协议复制到本轮目录并记录 sha256。冻结的生成器与 critic 权重只读加载，不覆盖。只做只读 git 操作（两个仓库都有用户未提交的工作）。开始与结束各做一次完整性审计。共享 DB 只通过官方设计阶段的 INSERT OR IGNORE 登记、`append_journal` 默认路径、单进程 merger 写入；导入后断言 `shared_rollout_db.__file__` 位于 `/home/zhihan/research/Basin_C1/` 之下（PoC/source 中有旧的同名副本）。

**与并行轮次的边界：**
- 另一个对话的分布式线目录：`diagnostics/decentralized_*`（含以后新建的同前缀目录）与 `diagnostics/teacher_student_*`，不得读写。唯一例外是共同协议第二节引用的 `diagnostics/decentralized_pipeline_v1/`，允许按协议只读。
- 多场景线目录：`diagnostics/multiscene_*`；跨type2 目录：`diagnostics/crosstype2_*`（若已存在）。两者都不得读写。
- 以上禁止目录一律按目录名前缀剪枝，不按轮次代号匹配。对禁止目录和并行任务目录，审计只允许记录路径、大小与修改时间（`os.stat`），不得打开文件、不得计算内容哈希。开工审计时，把最近 24 小时内被其他运行中任务写入的目录列为"并行任务目录"，其变化单独列出，不作为本轮审计失败的理由。
- 若因剪枝错误误读了禁止目录的文件（只读、内容未被使用、未写入），记录为边界偏差：写明涉及的文件、修正剪枝，然后继续。写入禁止目录，或在实验中使用了其中的内容，仍然整体停下。

**场景与安全层：** 只用 Ring 场景（round10 生成器）与集中式安全投影，H700，x64 = False。执行链与跨type1 完全相同：环形交换场景流匹配策略用 FF 位置，其余用 TT 位置（PoC `Runtime.action` 的 `chain = "old"`），只替换 nominal 动作的来源。OrthoFlow3 修正 δa^η = η_goal·B_goal + η_perp·B_flow⊥ + η_rel·B_rel 每步重算，与历史实现相同。

# 二、跨type1 的结果与本轮的出发点

**论文定位（用户确定）：** 修正黑盒 controller 在安全层下出现的死锁，不论死锁成因。黑盒指方法不依赖 controller 的控制律式子，只探查它的行为。死锁是否被解决，用成功率相对 safety baseline（只加安全投影、不加修正）的提升来衡量。timeout 不得直接称为 deadlock。

**跨type1 的结果（按 controller 选臂）：**
- 9 次迭代的全部选择器，对每个学习型 controller 都选了按进度切换时间表；
- 一律按进度切换 95.72%，每个 controller 事后最佳 96.11%，按 controller 选的空间只有 0.39pp。

**本轮要回答的问题：** 按 controller 选没有空间，那么**按 state 选**有没有？历史上 Ring 的同场景研究（`controller_cached_ff_relearn_v1`）中，按 state 匹配的选择达到 97.07%，每个 controller 一个固定 η 为 95.21%，说明 state 层面可能有额外空间。真正的 pipeline 正是按 state 选的。

# 三、动物园与 state

**controller：** 跨type1 `ZOO_MANIFEST.csv` 中全部合格实例，共 9 个类型、62 个 controller：
- 8 个开发类型（56 个）：环形交换场景流匹配策略（Flow）；高斯（`gaussian`）、确定性（`deterministic`）、混合密度（`mixture`）、扩散（`diffusion`）、动作分块（`chunking`）、循环网络（`recurrent`）、条件变分自编码器（`cvae`）行为克隆策略；
- 参数共享独立近端策略优化强化学习策略（`ring_ippo`，6 个）：它的标签在跨type1 中已经打开过，本轮作为普通开发类型。

隐式能量行为克隆策略仍是缺口，本轮不处理。

**state 划分：**
- **critic 训练 state：** 跨type1 的 16 个调参 state（`TUNING_STATES.json`）与 32 个评估 state（`EVAL_STATES.json`），共 48 个。它们在跨type1 中已有锚点臂的标签（见第四节），本轮补上生成器候选的标签。
- **新评估 state（本轮新生成，冻结前不得使用）：** 24 个，用 round10 生成器按开工时写定的种子生成，与 DB 中全部 state 去重，seed 0–1。冻结前不得为它们跑任何闭环 rollout。为它们计算 h 与生成器候选不需要 rollout，可以提前做。

# 四、候选动作

**每个 state 的候选集（共 13 个）：**

| 来源 | 候选 | 说明 |
|---|---|---|
| 冻结的生成器 | 5 个 η 提议：变换后的均值，以及 4 个随机样本 | 完全按 `orthoflow3_generator_critic_v1` 第 6 节的冻结提议规则：输入为该 state 的 h（`FrozenNewRuntime.conditioning`）与环境描述 c，按 `normalization.json` 归一化，随机样本用 SHA256 派生的确定性种子。生成器不看 controller（在 controller 研究中一直保持冻结） |
| 由生成器提议得到的动作 | 每个提议 η_g 给出两个动作：**全程 η_g**（原 pipeline 的整局固定 η），以及**前期 η_g、按进度切换到 η10**（切换规则与按进度切换时间表相同，p* = 0.9681620884830543，实现只读复用 round35 的 `progress.py`） | 共 10 个 |
| 锚点 | safety baseline（η = 0）、按比例切换时间表（`S500`，前 500 步 η13、之后 η10）、按进度切换时间表（`Sprog`） | 3 个；跨type1 已有标签 |

η10 = (1.246, −0.289, 0.186)，η13 = (1.075, 0.058, 0.266)，均在生成器的 η 盒 [0.5, 1.25] × [−0.5, 0.5] × [0, 0.75] 之内。持续右手规则在全部学习型类型上都很差（跨type1：0–7%；强化学习策略 94%，低于按进度切换的 99.5%），本轮不放入候选。

**开发中可以改变候选的组成**（例如提议数、切换目标、是否加入第二种切换规则），但每次改动都要给全部 controller 补齐对应标签，写进迭代日志，并计入追加预算。

# 五、标签

**critic 训练 state 的标签：** 每个 controller × 48 个训练 state × 10 个生成器动作 × seed 0；锚点 3 臂复用跨type1 的标签（调参 state 为 seed 0，评估 state 为 seed 0–1）。已有完全同身份的记录一律通过 DB preflight 复用（例如历史上 Flow 在 Ring 上的固定 η 标签，若 state、η 的三个 float64 字节、controller 与 seed 都相同）。

**新评估 state 的标签（冻结之后才跑）：** 每个 controller × 24 个新评估 state × seed 0–1 × 全部 13 个候选 + nominal only。全部候选都要跑，用于计算按 state 的 oracle 与遗憾。

**规模上限（开工时写定）：** 训练 state 约 62 × 480 ≈ 30,000 条；新评估 state 约 62 × 24 × 2 × 14 ≈ 42,000 条；合计约 72,000 条，扣除精确复用。开发中若需追加标签（改变候选组成等），累计追加不超过 10,000 条，每次写进迭代日志。资源不够时按第九节的顺序先做试点，不得为省预算删去新评估 state 上的任何候选。

# 六、critic 与 pipeline

**critic Q(h, C, a)：**
- **输入：**
  - state：h（Ring 的 100 维 `conditioning.flat`）与环境描述 c，按训练 state 归一化；另可加 round16 的 physical23；
  - controller：C 为跨type1 的无成败机制探测特征（24 维，`DEVELOPMENT_FEATURES.npz` / `LOCKED_FEATURES.npz`，按冻结的 `features.py` 定义），可选加 R5-PCA2；
  - 动作：前期 η（3 维）、后期 η（3 维，全程固定时与前期相同）、切换方式的独热编码（全程固定、按进度切换、第 500 步切换）、是否为 safety baseline。
- **起点结构：** 沿用 `orthoflow3_generator_critic_v1` 的 critic（native adapter 96、上下文与 η 编码 32、共享 128/64、一个 logit），加一个宽 32 的 controller 编码分支，并把 η 编码扩为上面的动作编码。允许用冻结 critic 的 state 与 η 分支权重做 warm start，也允许从头训练，两者都要报告。
- **损失：** 与历史相同，按观测到的成功次数做二项对数似然：−s·log p − f·log(1 − p)，权重为 min(实际 seed 数, 16)；unknown 不计入，不填 0。
- **训练细节：** AdamW，学习率 1e-3，权重衰减 1e-4，梯度范数截断 5；3 个 seed（17、23、41）集成。checkpoint 与超参数只能在源类型内部再做一次留一类型（嵌套）来选。round38 的教训是：留出 controller 的负对数似然从第 500 步起就开始上升，最后只学到平均最好的臂。必须监控并报告留出负对数似然曲线，以及**同一 state 内**不同候选的排序准确率。
- **选择规则：** 对每个 state 在 13 个候选中取 critic 预测最高者。另做一个保守版本：用集成的均值减 1 倍标准差打分；若最高者相对按进度切换时间表的优势小于 0，就执行按进度切换时间表。

**对照方法（同表报告）：**

| 编号 | 全称 | 内容 |
|---|---|---|
| 对照 1 | 一律按进度切换时间表 | 跨type1 的最佳单一修法，本轮的主要对手 |
| 对照 2 | 一律按比例切换时间表 | |
| 对照 3 | 跨type1 冻结的按 controller 选择器 | 直接使用 `FROZEN_PIPELINE.json`，不重训 |
| 对照 4 | 原 pipeline 原样 | 冻结的生成器 + 冻结的 critic（`orthoflow3_generator_critic_v1`，不看 controller），只在 5 个全程固定 η 提议中取最高者。这是这套方法在 controller 研究之前的原始形态 |
| 对照 5 | 只用生成器 | 执行生成器均值的"前期 η_g、按进度切换到 η10" |
| 对照 6 | 不看 controller 的 critic | 与主 critic 相同，但去掉 C 输入；检验 controller 信息是否有用 |
| 对照 7 | 不看 state 的 critic | 与主 critic 相同，但去掉 h 与 c 输入；它等价于按 controller 选，检验 state 信息是否有用 |
| oracle（特权） | 按 state 的事后最佳候选 | 交叉拟合：用 seed 0 选、在 seed 1 上打分，反之亦然，取平均；另报样本内与按 controller 的 oracle |

# 七、开发与冻结

**开发前先报告可选空间（不看任何模型，只用训练 state 的标签）：**
- 按 state 的交叉拟合 oracle − 一律按进度切换；
- 按 controller 的 oracle − 一律按进度切换；
- 生成器覆盖：在多少比例的 (controller, state) 上，生成器的 10 个动作中至少有一个不差于按进度切换；有多少比例严格更好。

类型等权与 controller 等权各报一次。训练 state 上锚点只有 seed 0（调参 state）或 seed 0–1（评估 state），交叉拟合只在有 2 个 seed 的 state 上做，并注明。

**留一类型开发检验（DEVELOPMENT，9 折）：** 每折留出一个类型，其余 8 个类型为源；全部预处理、超参数与 checkpoint 只能用源类型（嵌套）来选；在留出类型的训练 state 上打分。**类型等权**合并为主，另报 controller 等权。

**开发目标（达到就冻结；达不到也要在预算内冻结当前最好的版本，并如实报告）：**
- 类型等权：pipeline − 一律按进度切换时间表的配对 CI 下界 > 0；
- 相对按 state 交叉拟合 oracle 的平均遗憾小于"一律按进度切换"相对它的遗憾的一半；
- 没有任何留出类型比一律按进度切换差 2pp 以上；伤害数（比 safety baseline 低 ≥ 10pp 的 controller 数）为 0。

**冻结：** 把 9 个留一类型模型（每个都没见过它的留出类型）与用全部 9 个类型训练的最终模型，连同代码、输入定义、候选规则、选择规则写入 `FROZEN_PIPELINE.json` 并留 UTC 与 sha256 收据。然后为 24 个新评估 state 计算候选并用冻结模型选择，留选择收据，选择收据的时间必须早于新评估 state 的第一个 rollout 提交。

# 八、新评估 state 上的一次性评估（CONFIRMATORY）

每个 controller 用"没见过它所属类型"的那个留一类型模型选择。因此每个 controller 都处于"未见类型、未见 state"的条件下。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| 终点 1（主） | 真正的 pipeline 是否胜过最佳单一修法 | 类型等权：pipeline − 一律按进度切换时间表的配对 CI 下界 > 0 |
| 终点 2 | 是否接近按 state 的事后最佳 | 相对按 state 交叉拟合 oracle 的平均遗憾 ≤ 2pp，伤害数为 0 |
| 终点 3 | state 与 controller 信息各自是否有用 | pipeline − 对照 6（不看 controller）、pipeline − 对照 7（不看 state）的配对差与 CI |
| 终点 4 | 相对原 pipeline 的进步 | pipeline − 对照 4（原 pipeline 原样）的配对差与 CI |
| 终点 5（描述性） | 生成器的作用 | 新评估 state 上的生成器覆盖；pipeline 选中生成器动作与锚点的比例；按类型列出选中动作的分布；同一 controller 在不同 state 上选中不同动作的比例 |

另报：用全部 9 个类型训练的最终模型在新评估 state 上的结果（各类型都见过，只作参照）。

**CI：** 按共同协议第四节，5000 次配对 bootstrap；类型等权时先在类型层重抽，再在 controller 层，再在 state 层成簇重抽（两个 seed 保留在同簇）；oracle 在每次重抽内重新计算。unknown 按下界计，另报上界。

# 九、执行顺序

1. 完整性审计（开始，含并行任务目录清单与 `squeue`）→ 复现门：跨type1 的 16 条记录逐位一致；冻结的生成器在 `orthoflow3_generator_critic_v1` 的 Ring 验证 state 上重算的提议与其 `frozen_validation_proposals.json` 逐位一致（或在 1e-6 以内，写明原因）；冻结的 critic 在同一批 state 上的分数与原记录一致。不一致就停下报告。
2. 为 48 个训练 state 与 24 个新评估 state 计算 h、c 与 13 个候选，冻结候选清单并留收据。
3. **试点：** 先在 3 个类型（流匹配策略、扩散行为克隆策略、参数共享独立近端策略优化强化学习策略）上跑训练 state 的生成器动作标签，报告这 3 类的可选空间与生成器覆盖。试点结果不改变后续设定，只写进迭代日志；之后补齐其余类型。
4. 可选空间报告 → critic 开发与留一类型检验（迭代）→ 冻结。
5. 新评估 state：冻结模型选择并留收据 → 全部候选的标签 → 终点 1–5。
6. 完整性审计（结束）→ 交付。

任何阶段被挡住时，把受影响的类型记为缺口并写明原因，继续其余部分。只有源码保护、DB 规则或复现门无法满足时才整体停下。

# 十、结论映射

| 观察结果 | 结论 |
|---|---|
| 终点 1 通过 | 真正的 pipeline（生成器 + 看 state 与 controller 的 critic）在没见过的学习型类型与新 state 上胜过最佳单一修法：按 state 选有按 controller 选所没有的价值 |
| 终点 1 不通过，可选空间报告中按 state 的空间也很小 | 在 Ring 的学习型 controller 上，按进度切换已接近上限，按 state 选也没有可赚的；跨type2 应加入最佳修法不同的类型 |
| 可选空间大，但 critic 没能利用 | 瓶颈在 critic 的跨类型泛化；看留出负对数似然曲线、同一 state 内的排序准确率，以及终点 3 中哪种信息有用 |
| 终点 3 显示不看 controller 的 critic 与 pipeline 相当 | 在学习型类型之间，state 信息足够，controller 信息不是瓶颈 |
| 生成器覆盖很低 | 冻结的生成器只学过原 Flow 的成功 η，不足以服务其他 controller；下一步考虑看 controller 的生成器 q(η | h, C) |

# 十一、交付

按共同协议第五节交付，放在本轮目录。另外：
- 复现门报告；候选清单与收据；可选空间报告；试点报告；
- critic 的代码、config、训练曲线（含留出负对数似然与同一 state 内排序准确率）、checkpoint 与收据；`ITERATION_LOG.md`；`FROZEN_PIPELINE.json` 与选择收据；
- AXIS_TABLE：每个类型一行，另加"类型等权合并""controller 等权合并"两行；列为 13 个候选中的 3 个锚点、第六节全部方法与 oracle，分"训练 state 留一类型"和"新评估 state"两部分；单元格为成功率 % [95% CI]；另附碰撞率、超时率、平均与最大遗憾、伤害数、证据等级；
- 每个 (controller, 新评估 state) 的选中候选、各候选的成功数、遗憾的明细表；
- 每个类型一张"选中动作分布"图；每个类型一个典型 state 上 safety baseline、按进度切换时间表与 pipeline 选中动作的轨迹静态图；
- REPORT.md：第一段先写完整性审计是否通过，再给出可选空间（按 state 与按 controller）、生成器覆盖、开发目标是否达到、终点 1–4 的结论与关键数字；
- HANDOFF.md（不依赖本对话也能读懂），写明跨type2 可以直接复用的部件（critic 代码、候选规则、冻结模型）与路径。

**资源：** 用户全部任务合计不超过 18 个 CPU shard、2 个 GPU shard。开工前与每次提交前都要用 `squeue` 查看总占用。**不做 git commit。**

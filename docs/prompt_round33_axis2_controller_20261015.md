你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/axis2_controller_v1/`

# 零、先读

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）。** 其中的源码保护、DB 规则、复用规则、标准方法与指标、标准交付，对本轮全部适用，优先级高于本文件以外的一切。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. 总结报告 `docs/controller_generalization_ring_summary_20261014.md`，重点是第 0、2、14–17 节。
4. round27（`diagnostics/other_controller_classes_v1/`）与 round29（`diagnostics/nominal_safety_baselines_v1/`）的报告与代码；round16（`diagnostics/ring_controller_critic_v1/`）；round26（`diagnostics/method_benchmark_v1/`）。

# 一、要回答的问题：泛化轴 2，Controller

**定义：** 同一场景（Ring），测试没见过的 controller。

| 子项 | 已有证据 | 本轮要补的 |
|---|---|---|
| 2a 新 seed | round14A，14 个，G_E 平均遗憾 0.61pp | nominal only 与 safety baseline |
| 2b 新 checkpoint | round12，8 个同 run 兄弟，1.42pp | 同上 |
| 2c 新架构 | round17 N2（13 个），A3 1.70pp；round18 TEST（18 个） | 同上 |
| 2d 新类型 | round27 与 round29：MPC 与势场在方法层面接近 oracle；但 critic 学到的关系能否跨类型，未得到确认（O3 REJECTED；固定菜单中的信号只是描述性的，而且每类内部答案完全一致） | **4 个新类型**，用来检验学到的关系能否跨类型迁移 |
| 2e 附加：部署方式 | round28（分散式安全层）；round30（teacher–student，若已完成） | 只整理进表 |

**本轮的核心问题（2d）：**
- 在答案确实有分歧的新类型 controller 上，冻结的 A3 是否真的利用了 controller 信息，而不只是选了一个对谁都还行的稳健动作？
- 把更多类型加进训练，能否改善对新类型的零样本选择？

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 与 sha256 收据）

## 2.1 补齐 2a–2c 的基线

- **controller：** round14A 的 14 个、round12 的 8 个、round17 N2 的 13 个、round18 TEST 的 18 个，共 53 个。
- **state：** 各自原评估 state 中，按 state_uid 排序的前 32 个；seed 0–1，seed key 与原轮次一致，以便配对。
- **臂：** nominal only、safety baseline（FF，η = 0）。
- **复现门：** 各轮次的执行代码各复现 16 条历史记录（共同协议第三节）。
- **规模：** 53 × 32 × 2 × 2 = 6,784 条。

## 2.2 新类型 controller（2d）

四个新类型，每类 6 个实例。参数网格写入预注册，不得根据结果调整。全部实现写在本轮目录中。

| 类型 | 内容 | 6 个实例 |
|---|---|---|
| **G：高斯 BC 策略** | 单峰的高斯 MLP 策略：输入为联合观测（4 × 23），输出 8 维均值与对角 log-std。用 BC 在 Ring 专家数据上训练，与 MACFlow 使用相同的数据集。观测的坐标系（local 或 global）、动作的坐标系、归一化与限速，都与该配方的 Flow 相同（参照 `RingFlowAdapter` 的 `observation` 与 `to_world`）。执行时用 episode 的随机键从高斯中采样。它是学习得到的、非 flow、单峰的 controller | 从 14 个配方的数据集中预先选 6 个（写入预注册），每个 1 个 seed。MLP 3×256，Adam 3e-4，步数与对应 Flow 相同，按 dev 损失选 checkpoint，全部设置写入 config（round24 的教训） |
| **DM：分散式 MPC** | 每个机器人只为自己做 random shooting，把其他机器人预测为匀速运动。代价与 round27 的 MPC 相同：到目标的距离、碰撞惩罚、平滑项 | 从 H_mpc ∈ {10, 30}、样本数 ∈ {64, 256}、避碰权重 ∈ {20, 100} 中选 6 组 |
| **VO：速度障碍式反应 controller** | 每个机器人采样候选速度，剔除在时间 τ 内会与邻居（按匀速假设）相撞、或撞上障碍与外环的速度，选最接近期望速度（指向目标、限速）的那一个 | τ ∈ {1, 2, 4} s × 邻居范围 ∈ {1.5 m, 全部} |
| **PT：带切向项的势场** | round27 的势场，加一项环绕中心障碍的切向速度 k_t·t̂（t̂ 为逆时针切向） | (k_t, 吸引增益) ∈ {(−0.3, 1.0), (−0.15, 1.0), (0.15, 1.0), (0.3, 1.0), (0.15, 0.6), (0.15, 1.4)}；排斥增益取 round27 势场 P2 的值 |

**说明：** PT 的切向项与修正中的 B_⊥ 方向相互作用，有望在固定菜单中产生"有的偏好 F10、有的偏好 F13"的分歧。这正是检验学到的关系所需要的。

**执行：** 全部用 TT 位置（PoC `Runtime.action` 的 `chain = "old"`），只替换 nominal 动作的来源，与 round27 相同。复用 round27 的执行代码时，按共同协议第三节的复制规则，并先过复现门（复现 round27 的 16 条记录）。

**能力门：**
- 每个新 controller 在 16 个 pilot state（新生成，种子区间写入预注册，与 DB 及评估 state 去重）上，用 F10 与 F13 各跑 seed 0。
- 两个臂都低于 15%，或都高于 98%，就排除该实例。不替换、不调参。
- 某类型剩下的实例少于 4 个，该类型记为 INCONCLUSIVE。

**评估：**
- 32 个新 state（与 DB 及 pilot 去重），seed 0–1。
- 臂：nominal only、safety baseline、F10、F13、F2、S200、S300、S500（时间表为 η13 → η10）。
- 规模：最多 24 × 32 × 2 × 8 = 12,288 条，另加能力门 768 条。

**探测与冻结选择（在任何评估标签之前完成并留收据）：**
- 对通过能力门的实例运行 R5 探测（TT 位置），预处理与 PCA 用 round16 冻结的版本。
- 冻结以下选择：A3（只在 Flow 上训练）、A0、G_E、一律 S500、一律 F10。完整菜单为上述 6 个修正臂；固定菜单为 {F10, F13, F2}。
- 混合类型 critic（见 2.3）的选择也在此时冻结。

## 2.3 混合类型 critic（在任何新类型标签之前训练并冻结）

- **架构：** 与 A3 相同，输入为 [physical23, R5-PCA2, 动作 one-hot]，另加一个 FF/TT 位置标志。
- **训练数据：** round15 的 Flow 标签（FF）加 round27 的 MPC 与势场标签（TT）。菜单取二者的交集。
- **训练设置：** fit seed 17、23、41；选 step 的规则与 round16 相同。
- **对照（描述性，DEVELOPMENT）：** 在 {Flow, MPC, 势场} 三个已有类型上做留一类型，报告遗憾。

# 三、预注册的终点与判定

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| CT1 | 方法层面：新类型 | 在通过能力门的新类型实例上，A3（完整菜单）相对完整菜单 oracle 的平均遗憾 ≤ 3pp；A3 − safety baseline 的 CI 下界 > 0；伤害数为 0 |
| CT2 | critic 学到的关系能否跨类型（固定菜单 {F10, F13}） | 前提：新类型实例的固定偏好有分歧，即至少 3 个实例 F10 比 F13 高 ≥ 10pp，且至少 3 个实例 F13 比 F10 高 ≥ 10pp；不满足就记为 INCONCLUSIVE。判定：A3 的平均遗憾低于 A0，配对 CI 上界 < 0 |
| CT3 | 训练中加入更多类型是否有帮助 | 在新类型上，混合类型 critic（完整菜单）的平均遗憾低于 A3，配对 CI 上界 < 0 |
| CT4（描述性） | 2a–2c 的基线 | 每个子项上 A3（或该轮原用的规则）相对 safety baseline 与 nominal only 的提升，以及从 nominal only 到 safety baseline 的转移表 |

**证据等级：** CT1–CT3 与 2.1 的新标签为 CONFIRMATORY；已有轮次的结果为原有等级。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| CT1 通过 | 方法在全新类型的 controller 上仍可零样本使用 |
| CT2 通过 | critic 学到的 controller 与 Q 的关系，跨类型迁移得到确认：它能在答案有分歧时选对 |
| CT2 为 INCONCLUSIVE | 新类型之间仍无分歧，跨类型的关系仍不可检验，论文只能声称方法层面的迁移 |
| CT2 不通过 | 学到的关系不跨类型，跨类型时的收益来自稳健动作 |
| CT3 通过 | 增加训练中的 controller 类型能改善泛化：支持"多类型训练"的方法叙事 |
| 类型 G 在 safety baseline 下大量停滞，而修正后成功 | 单峰 BC 的模式平均问题，同样可以由修正弥补；可以作为论文的补充论据 |

# 五、交付

按共同协议第五节交付，放在本轮目录。另外：
- 新 controller 的实现、BC 训练代码、config、checkpoint 与单元测试；
- 能力门报告；
- 探测特征、冻结选择与混合类型 critic 的 checkpoint 及收据。

AXIS_TABLE 的行至少包括：2a 新 seed、2b 新 checkpoint、2c 新架构（round17 N2、round18 TEST 各一行）、2d 新类型（MPC、势场、G、DM、VO、PT 各一行）、2e 部署方式（round28 三条管线、round30 若已完成）。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过；
- CT1、CT2、CT3 的结论与关键数字；
- Controller 轴各子项的一句话总结。

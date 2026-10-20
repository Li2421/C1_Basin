你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，以及总结报告 `docs/controller_generalization_ring_summary_20261014.md`（重点是第 2、14、15 节）。严格遵守 AGENTS.md。

# 一、要回答的问题：结论能否推广到非 Flow 类的 controller？

到目前为止，所有 controller 泛化的证据，都来自同一类学习型策略（MACFlow）：不同的 checkpoint、seed、配方与网络宽度。论文中“跨 controller 泛化”的说法，需要至少一类结构完全不同的 controller 来检验。用户不关心新 controller 的具体形式，只要求它们与 Flow 在类别上不同。

**本轮检验两个层次：**
- **L1（与方法无关的原理）：** 对新类别的 controller，时间表（前期 η13、后期 η10）是否同样优于固定 η，好方向是否同样在 controller 之间共享？这一层不需要 critic。
- **L2（学到的关系能否跨类迁移）：** 只在 Flow controller 上训练的冻结 critic A3，以新 controller 的 R5 探测作为输入，能否零样本给它们选好动作？此外，在新类别内部做 controller 间的留一，同类训练的 critic 能否学到类内的 controller–Q 关系？

# 二、新 controller 的实现（只在本轮目录中实现，不改任何已有代码）

**两个新类别，每类 6 个实例。** 参数网格在预注册中写定，不得根据结果调整。

**类 M：采样式 MPC（random shooting 或 CEM）。**
- 每一步在内部模型中滚动 H_mpc 步，模型为单积分器 x' = x + dt·u，|u| ≤ max_speed。
- 代价包括：到目标的距离、与障碍和外环的碰撞惩罚、机器人之间的碰撞惩罚、动作平滑项。
- 执行第一步动作。
- 6 个实例由 3 个维度组合而成：
  - 预测步长 H_mpc ∈ {10, 30}；
  - 样本数 ∈ {64, 256}；
  - 避碰权重为低或高。

  从组合中选出 6 个写入预注册。
- 中心化或分散式均可，选一种写明。

**类 P：反应式势场 / 社会力 controller。**
- 速度命令 = 目标吸引 + 机器人间排斥 + 障碍排斥，再按 max_speed 截断。
- 6 个实例对应 6 组增益，网格在预注册中写定。

**Flow 参照组：** round10 的 14 个原 Ring Flow 中，取 4 个，覆盖 η13 型与 η10 型，在预注册中写明是哪几个。它们在本轮使用与新类别相同的修正位置（TT，见下），用来区分“controller 类别的影响”与“修正位置的影响”。

**修正位置：**
- 非 Flow 的 controller 没有 flow ODE，FF 修正无从定义，因此新类别一律使用 TT：nominal 动作 → 安全投影 → 加上 OrthoFlow3 修正 → 再次投影。
- 复用已有的 Ring TT 执行代码（例如 `diagnostics/orthoflow3_tt_ff_matched_learnability_v1/` 与 `new_benchmark_common/safety_eta3.py` 中 orthoflow3 链的写法）。
- 只把 nominal 动作的来源换成新 controller；物理、安全投影、事件判定与成功标准都不得修改。

**能力门（预先写定，只用独立的 pilot state）：**
- 每个新 controller 在 16 个 pilot state 上，用 F10、F13 各跑 1 个 seed。
- 两个臂的成功率都低于 15%，或者都高于 98%（没有区分度），就排除该实例并报告。不得替换实例，也不得调参后重试。

**精度：** 全部导入完成之后，显式恢复协议规定的精度（round25 的教训），并记录 x64 状态。

# 三、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**无标签探测（在任何评估标签之前完成并冻结）：** 对所有通过能力门的 controller 运行 R5：16 个 probe state × {η10, η13} × 200 步，关闭终止，实现与 round15 / 17 冻结的探测代码相同，修正位置为 TT。然后按 round16 冻结的预处理和 PCA 投影。

**评估标签：**
- 32 个全新 Ring state，与 DB 中所有 state 去重；
- 臂：F10、F13、F2、S200、S300、S500（时间表为 η13 → η10）；
- seed 0–1；
- 规模：约 (12 + 4) × 32 × 6 × 2 ≈ 6,144 条完整 rollout，另有 pilot 与探测。

**流程：**
- 先过一致性门：用 Flow 参照组在原 TT 实现下逐位复现已有的 TT 记录，若存在；没有可比记录时，就复现两次运行的逐位一致性。
- 按 shared_rollout_db 流程执行：design → preflight（包括探测的 probe preflight）→ 只跑缺失 → journal → 单进程 merge → postflight。新 controller 登记为新的执行身份。

**冻结的跨类选择（必须在评估标签之前冻结并留收据）：**
- 对每个新 controller，用 round16 冻结的 A3 与 A0 做 controller 级选择，完整菜单与固定菜单各一次；
- 同时记录 G_E 的选择，以及全选 S500、全选 F10 的结果。

# 四、预注册的终点与判定

所有 CI 先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次。unknown 按下界计，另报上界。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| O1（L1 主检验） | 时间表在新类别中是否优于固定 η | 在类 M 与类 P 的合并结果上，“最佳时间表 − 最佳固定臂”的平均值 CI 下界 > 0；并且在两类中各有 ≥ 70% 的 controller 满足 S(最佳) ≥ max(F10, F13, F2) − 1pp |
| O2（方向共享） | 好方向是否在类内共享 | 在每个类别内，η13 → η10 这一方向的优势（S500 ≥ F13 且 S500 ≥ F10，各留 1pp 余量）在 ≥ 70% 的 controller 上成立。与固定偏好 sign(F13 − F10) 的一致率对比（描述性） |
| O3（L2，跨类零样本） | Flow 训练的 critic 能否直接用 | A3（完整菜单）在新类别上的平均遗憾低于 A0，配对 CI 上界 < 0；同时报告与全选 S500、G_E 的比较 |
| O4（类内可学性，描述性） | 同类数据训练的 critic 能否学到 | 在每个新类别内部做 controller 间留一，用 A3 的架构与超参数，在该类其余 controller 的标签上训练，再对留出的 controller 选择。报告遗憾，并与全选 S500 比较 |

**Flow 参照组（描述性）：** 报告这 4 个 Flow 在 TT 下的 O1、O2，并与它们在 FF 下的已有结果（round10）比较，以说明修正位置的影响。

# 五、结论映射

| 观察结果 | 结论 |
|---|---|
| O1、O2 都通过 | “阶段时间表优于固定 η、且方向在 controller 间共享”不依赖 Flow 这一类别，在 MPC 与势场 controller 上同样成立（限 Ring、TT） |
| O1 通过，O3 不通过 | 原理可以跨类，但 Flow 上训练的 critic 不能直接跨类使用，需要同类数据（结合 O4 判断） |
| O1 不通过 | 时间表的优势依赖 Flow 类 controller 的行为特征。报告新类别的失败类型与发生时间，并结合 Flow 参照组判断这是类别造成的，还是修正位置（TT）造成的 |
| 能力门之后，某类别剩下的实例少于 4 个 | 该类别的结论记为 INCONCLUSIVE |

# 六、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/other_controller_classes_v1/` 中写文件（包括新 controller 的代码）。不修改、不删除历史资产与已有代码，不做 git commit。
- **不得改动核心组件：** 不修改物理、安全投影、事件判定或成功标准。新增的只有 nominal controller；时间表只在第 t 步替换 η。
- **参数写死：** controller 参数网格、能力门、臂、阈值都已在预注册中写定，不得根据结果调整。
- **冻结时序：** 探测与跨类选择必须在评估标签之前冻结。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **视频：** 本轮属于纯 Q 函数的数据生成与评估，按 AGENTS.md 的例外不运行视频入口。这一判断在预注册中写定。
- **资源：** 本 workstation 上用户所有任务（包括并行的 round26）合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 七、交付

全部交付物放在 `diagnostics/other_controller_classes_v1/`：
- PREREGISTRATION.md 及收据；
- 新 controller 的实现与单元测试；
- 能力门报告；
- 探测特征与冻结的选择及收据；
- 一致性门报告；
- 去重审计；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：每类各臂的成功率、O1 / O2 的分布图、跨类选择的遗憾图；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：时间表现象能否推广到非 Flow 类的 controller（O1、O2），以及 Flow 上训练的 critic 能否跨类零样本使用（O3）。

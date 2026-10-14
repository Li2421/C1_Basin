你是 Basin_C1 / OrthoFlow3 的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

先读 `AGENTS.md`。

# 本文件的性质：round10 的补充预注册分析

- 本文件由实验设计者在 round10（`diagnostics/controller_universal_eta_schedule_v1/`）主实验运行期间写成。写成时，**round10 的任何结果都尚未被读取**。写成时间与 sha256 见同目录的 `prompt_round10_addendum_phase_decomposition_20261013.receipt.json`。
- 本分析只使用 round10 的冻结结果，以及历史的 controller 描述（R3、R5），**不新增任何 rollout**，也不修改 round10 目录。
- 必须在 round10 标记为 COMPLETE 之后才能执行，并且严格按下面写定的定义与预测执行，不得根据结果调整。

# 一、要检验的根因假设 H_compress

在“整局固定一个 η”的设定下，controller 级的 η 偏好不是一个独立的 controller 属性，而是**两个相反阶段效应之差的符号**：

- **E(C)，前期效应：** 前期用 η13 换来的收益，定义为 S500 − F10。
- **L(C)，后期效应：** 后期换回 η10 换来的收益，定义为 S500 − F13。
- **N(C)，固定 η 的净偏好：** N = F13 − F10 = E − L。

这个等式本身是恒等式，不需要检验。可检验的内容是：E 与 L 在多数 controller 上都很大，而且各自的符号一致；而 N 只是这两个大量之差，符号在 controller 之间来回翻转。

若成立，它能统一解释此前的所有失败：
- 静态描述与 200 步探测都识别不出 regime；
- 源 controller 先验不节省标签；
- 配方相近的 v9_local 与 v10_local 偏好相反。

# 二、数据与定义

- **数据：** round10 的全部 14 个真实 Flow、64 个全新 state、seed 0–3。
- **臂：** F10、F13、Freg（仅 3 个 controller 有）、S300、S400、S500。
- **成功率：** 一律按 unknown 保守下界计算，同时报告上界。
- **对每个 controller 计算：**
  - E、L、N；
  - 用 S300、S400 替换 S500 后得到的 E 与 L，作为敏感性分析；
  - CI：按 state 成簇做 bootstrap，5000 次，种子写进本分析自己的预注册收据。

# 三、预注册的预测（设计者写定，不得修改）

**P1：隐藏的反向效应。**
- 在 N < 0（固定偏好为 η10）的 controller 中，至少一半满足 E ≥ 10pp；
- 在全部 14 个 controller 中，至少 5 个同时满足 E ≥ 10pp 与 L ≥ 10pp。

**P2：分量比净值更一致。**
- E > 0 的 controller 数与 L > 0 的 controller 数，都 ≥ 11/14（按点估计）。
- 同时报告 N 的符号分布。

**P3：扩展动作空间使零样本选择良态化。** 在 14 个 controller 上做留一法：对留出的 controller，选“源 controller 平均成功率最高的动作”，不使用任何 controller 信息。比较两个可选动作集：
- 固定集：{F10, F13}；
- 扩展集：{F10, F13, S300, S400, S500}。

遗憾的参照统一取目标 controller 全部 6 个臂中的最佳臂（含 Freg）。

预测：扩展集的平均遗憾不超过固定集的 1/3，且扩展集的最大遗憾 ≤ 5pp。

**P4：分解后的 controller–Q 关系能否学到。** 不做方向性预测，只如实报告。设计者的预期是：E、L 的可预测性不会明显好于 N（与 round7 一致）；若结果相反，这是重要的正面发现。

- **(a) 预测分量：** 用 R3-PCA2、R5-PCA2 作为 controller 描述。PCA 只在源 controller 上拟合；R5 沿用 round7 的 155 维主方案。用 ridge（α = 1，留一）分别预测 E、L、N，报告 Spearman 相关与符号准确率。
- **(b) 用描述选动作：** 在扩展集中，用 1-NN（同样的描述）选择动作。报告它的遗憾是否低于 P3 中的源平均选择。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| P1 与 P2 都成立 | **H_compress SUPPORTED（DEVELOPMENT）**：零样本 controller 泛化失败的根因，是固定 η 的动作把两个 controller 特有的阶段效应压缩成了一个符号 |
| P1 或 P2 只在部分 controller 上成立 | 根因只适用于这部分 controller。列出例外，并给出它们的失败类型与发生步数 |
| P1 与 P2 都不成立 | **H_compress REJECTED** |
| P3 成立 | 在扩展动作空间中，不使用 controller 信息也能实现低遗憾的零样本选择 |
| P4 中 E、L 可预测，或 1-NN 进一步降低了遗憾 | 分解形式下，controller–Q 关系可学，下一轮可以设计结构化 critic |
| P4 中 E、L 都不可预测 | 用现有描述学不到 controller–Q 关系，即使改成分解形式也一样。只能依靠动作空间的良态化，或少量目标标签来估计 E、L |

# 五、规则与交付

- 只在新目录 `diagnostics/controller_phase_decomposition_v1/` 中写文件。不修改 round10 或其他历史资产，不做 git commit。
- 不新增 rollout，不写共享 DB。这是纯分析，不需要视频。
- 先写 PREREGISTRATION.md（引用本文件及其收据），再读取 round10 的结果。

交付：
- 代码；
- 每个 controller 的 E、L、N 表及其 CI；
- P1–P4 的判定；
- E–L 散点图（按固定偏好着色）；
- results.json；
- REPORT.md；
- HANDOFF.md。

REPORT.md 第一段必须直接回答 H_compress 是否成立，以及 controller–Q 关系在分解形式下能否学到。所有结论标为 DEVELOPMENT；事实与解释分开写。

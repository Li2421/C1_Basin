你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md` 和 `shared_rollout_db/README.md`，并严格遵守。

# 一、要回答的问题（只回答这一个）

在 Ring 场景下，η13 能帮助 base_u、v10_local（+26～30pp），却伤害 v9_local（−16pp）。**这个 controller 依赖的 η 效应，是在 episode 的哪个阶段产生的？**

round7–8 的结论（均为 DEVELOPMENT）：
- 静态描述 R3 和 200 步行为探测 R5 都识别不出 η 的 regime。
- 200 步内，η13 对所有 Flow 都推进得更快，但这种推进与最终成功呈负相关。
- 少样本适应只有在预设候选 η 菜单的条件下，才在 128 条标签时成立。源先验和 critic 都没有节省标签。

因此，当前最可疑的根因是：**η 的效应由后期的闭环交互决定（例如会车或停滞），短时或静态的 controller 描述看不到它。**

本轮用最直接的因果干预检验这一点：在 episode 中途切换 η，测量成功率随切换时刻的变化。

# 二、阶段 0：零 rollout 的结局分析（先做）

只读已有的 raw 记录：`diagnostics/controller_loco_real_flow_v1/raw/`，约 86k 个 JSON。

- 每条记录包含：outcome（success / timeout / collision / deadlock）、episode_length、first_success_step、各类碰撞的首次步数。
- 记录中没有轨迹。

**对象：** 7 个面板 Flow × {η10, η13}，使用 loco_test 的 64 个 state、seed 0–7。

**需要报告：**
- 各结局类型的比例。
- 成功步数的分布。
- 失败的类型与发生步数。
- 重点比较以下三组，看 η13 在每组里改变了哪一种结局、发生在第几步：
  - base_u / v10_local 在 η10 与 η13 下；
  - v9_local 在 η10 与 η13 下；
  - v9_local 与 v10_local 之间（两者静态描述相近）。

**本阶段只做描述，不作结论。** 写一页摘要后，再进入阶段 1。阶段 1 的设计不因阶段 0 的结果而改变。

# 三、阶段 1：η 切换时刻的干预实验（主实验）

## 设计（在执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**controller（4 个）：**
- η13 型：base_u、v10_local；
- η10 型：v9_local（受 η13 伤害、且与 v10_local 静态相近）、plus_w_v4_wide。

**state 与 seed：**
- state：round4 的 64 个 loco_test state。
- seed：0–3，seed key 与已有记录完全相同。这样切换前的轨迹与纯 η 基线逐步一致，可以配对比较。

**干预：**
- 前 t 步用 η_a，从第 t 步起改用 η_b，直到第 700 步。
- OrthoFlow3 修正仍然每步重算；唯一的改变是从第 t 步起，所用的 η 参数换掉。
- 两个方向：η10→η13，η13→η10。
- t ∈ {100, 200, 300, 400, 500}。
- t = 0 与 t = 700 就是纯 η 基线，直接复用已有标签，不重跑。

**规模：** 4 个 controller × 64 个 state × 2 个方向 × 5 个 t × 4 个 seed = **10,240 条完整 rollout**。

## 执行要求

- **一致性门（必须先通过）：** 修改后的 runtime 在 t = 0 和 t = 700 时，必须与已有记录逐位一致：outcome、episode_length、first_action。在 32 个 case 上检查。不通过就停下报告，不得进入主实验。
- **共享 DB：** 切换 η 是一种新的执行语义，必须登记为新的 controller / η 身份（在 identity 中写入 η_a、η_b、t），并按 shared_rollout_db 的流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。
- **轨迹记录：** 每条 rollout 每 10 步保存一次位置、速度和安全投影激活情况，供机制分析使用。只写到本实验目录。
- **其他设置：** Ring、FF、H700、K16 中的这两个 η、安全投影、成功判定、JAX x64 = false，全部不变。
- **资源：** 本 workstation 上用户所有任务合计不超过 18 个 CPU shard 和 3 个 GPU shard。
- **视频：** 本实验只涉及单个场景，属于机制诊断。按 AGENTS.md 判断是否适用视频要求；若适用，只对 ring_exchange 运行统一入口，并在报告中说明。

## 分析

**1. 成功率曲线 S(t)。** 对每个 controller × 方向，画成功率随 t 变化的曲线：
- 按 state 成组做 bootstrap 95% CI；
- 与两个纯 η 基线配对比较。

**2. 定位“决定时刻”。** 对 η13 型 controller，定义 η13 的总收益 D = S(全程 η13) − S(全程 η10)。
- t_half：η10→η13 方向上，保留至少 50% 收益的最晚切换时刻；
- 对 η10 型 controller，用同样的方法定位 η13 的伤害在何时产生。

**3. 机制。** 用轨迹描述切换前后的状态，例如停滞的机器人数、最小间距、会车所在位置，并说明 η 改变的是哪一种失败（timeout / collision / deadlock，以及发生步数）。

**4. 当前状态能否预测切换的效果（描述性）。** 以切换时刻 t 的物理状态为特征（不使用 controller 身份），用预注册的固定 logistic 回归（C = 1，按 state 留一），预测“切换到 η13 是否会成功”。比较两种情况：
- 加入 controller 身份 one-hot；
- 不加 controller 身份。

如果当前状态已经足以预测，而 controller 身份几乎不增加信息，就说明 controller 的作用主要通过它诱导出的状态分布体现。

## 预注册的判定

| 观察结果 | 结论 |
|---|---|
| η13 型 controller 的 t_half ≥ 300，且 η10 型的伤害也主要在 200 步之后产生 | **H_late SUPPORTED**：η 效应由后期交互决定。它解释了 R5（200 步）为什么失败；下一步的表示应基于后期状态，或改为 episode 中途重新选择 η |
| t_half ≤ 100 | **H_early**：效应在早期就已注定。R5 失败不是 horizon 的问题，而是统计量没有选对 |
| 介于两者之间 | 报告曲线形状，不强行二分 |
| 分析 4 中，状态预测充分，而 controller 身份增益很小 | **H_state_mediated SUPPORTED（DEVELOPMENT）**：可以设计一个与 controller 无关、在中途重新选择 η 的 critic，作为下一轮的零样本候选 |

# 四、硬性规则

- 只在新目录 `diagnostics/controller_eta_switch_time_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- 不修改控制器、物理、安全投影、事件判定或成功标准。唯一允许的改变，是在第 t 步替换 η。
- unknown 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- 所有结论标为 DEVELOPMENT。阶段 0 只做描述，不得用它来修改阶段 1 的设计。

# 五、交付

全部交付物放在 `diagnostics/controller_eta_switch_time_v1/`：
- PREREGISTRATION.md 及收据；
- 代码；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- 阶段 0 摘要；
- S(t) 曲线与 t_half；
- 机制图；
- 分析 4 的结果；
- results.json；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：η 效应在哪个阶段产生，以及这对“零样本 controller 泛化”的根本原因意味着什么。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE；事实与解释分开写，不夸大。

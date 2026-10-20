你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/axis4a_geometry_horizon_v1/`

# 零、先读

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）。** 其中的源码保护、DB 规则、复用规则、标准方法与指标、标准交付，对本轮全部适用，优先级高于本文件以外的一切。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. 总结报告 `docs/controller_generalization_ring_summary_20261014.md`，重点是第 0、2、12、14 节。
4. round10（`diagnostics/controller_universal_eta_schedule_v1/`）的预注册、执行代码与保存的完整轨迹；round20（`diagnostics/cross_scene_probe_selection_v1/`）中改变 Ring 障碍半径的做法；round16（`diagnostics/ring_controller_critic_v1/`）。
5. 若 round31（`diagnostics/gain_attribution_v1/`）已跑出 η = 0 的 FF 记录，它们会通过 preflight 自动复用。

# 一、要回答的问题：泛化轴 4，Cross-Factor 中的几何与时间预算

**定义：** 场景与 controller 不变，改变环境几何与任务的时间预算（horizon，即 max_steps）。机器人数量另见 round36。

| 子项 | 内容 | 已有证据 |
|---|---|---|
| 4a-H 时间预算 | H ∈ {500, 700, 1000} | 无。全部 Ring 实验都在 H700 下做 |
| 4a-G 几何 | 中心障碍半径 r ∈ {0.80, 0.85, 0.95, 1.00}（基准为 0.90） | round20：r = 0.75 或 1.05 时，Flow 本身 0/16 失败，测不了修正 |

**为什么先做 horizon：** 我们的时间表按绝对步数切换（第 500 步）。horizon 一变，这就是方法最可能失效的地方。本轮同时比较三种切换规则：
- 按绝对步数（S-abs）；
- 按 horizon 比例（S-rel）；
- 按进度触发（S-prog）。

如果按进度触发更稳，它会直接成为方法的改进。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 与 sha256 收据）

**controller：** round10 的 14 个原 Ring Flow（best.pkl）。FF 位置（PoC `Runtime.action` 的 `chain = "field"`）。复现门：复现 round10 的 16 条记录。

**state：** round10 的 64 个 state 中，按 state_uid 排序的前 32 个（与 round31 第二部分相同）；seed 0–1，`key_for` 与 round10 一致。这样，H700 与 r = 0.90 下已有的 F10、F13、S300、S500 标签可以直接复用。

**环境参数的改变：** horizon 与障碍半径都通过环境 config 的构造参数传入（在本轮目录中写新函数或子类）。原代码不改，物理与事件判定也不改。

## 2.1 时间预算（先做）

**切换规则：**
- **S-abs：** 第 500 步切换。H700 下即 S500；H500 下永远不会切换，等同 F13（不重复跑，按 F13 记）。
- **S-rel：** 第 round(5/7·H) 步切换。H500 下为第 357 步，H700 下为第 500 步，H1000 下为第 714 步。
- **S-prog：**
  - 记 p(t) = 各机器人 clip(1 − d_i(t)/d_i(0), 0, 1) 的平均，其中 d_i 为机器人 i 到目标的距离。
  - 在 p(t) 第一次 ≥ p* 时，从 η13 切到 η10；若到第 round(6/7·H) 步仍未达到，就在该步切换。
  - **p* 的确定：** 用 round10 已保存的 H700 下 S500 的完整轨迹（14 个 controller × 这 32 个 state × seed 0–1），取第 500 步 p(500) 的中位数。离线计算，在任何新 rollout 之前写入预注册并冻结。
  - S-prog 写成新函数并配单元测试：切换时刻必须与离线按同一轨迹算出的结果一致。

**各 H 下的臂：**

| H | 臂 |
|---|---|
| 500 | nominal only、safety baseline、F10、F13（= S-abs）、S-rel、S-prog |
| 700 | nominal only、safety baseline、S-prog 为新跑；F10、F13、S500（= S-abs = S-rel）复用 round10 |
| 1000 | nominal only、safety baseline、F10、F13、S-abs、S-rel、S-prog |

**规模：** 新跑约 16 个 (H, 臂) 组合 × 14 × 32 × 2 ≈ 14,336 条，减去可复用的部分。

**A3 零样本：** 冻结 A3 在 {F10, F13, S500} 中的 controller 级选择，再按"相对规则"映射到各 H：F10 → F10，F13 → F13，S500 → S-rel。在评估标签之前冻结并留收据。另报"绝对规则"映射（S500 → S-abs）作为描述性对照。

**专家（特权参照）：** 每个 H 下，对 32 个 state 各规划一次。

## 2.2 几何（在 2.1 之后做）

**能力门：**
- 每个 r 下，在 16 个新 pilot state（种子区间写入预注册，与 DB 去重）上，14 个 controller 用 F10 与 S500 各跑 seed 0。
- 若各 controller 的 max(F10, S500) 成功率的中位数 ≥ 20%，该 r 进入主评估。
- 否则记为"controller 在该几何下本身失败"，是 controller 的能力局限，不是修正的失败，不计入方法结论。

**主评估（每个通过能力门的 r，H700）：**
- 臂：nominal only、safety baseline、F10、F13、S500，共 5 个。
- 规模：每个 r 为 14 × 32 × 2 × 5 = 4,480 条，最多 4 个 r，共 17,920 条。

**A3 零样本：** 用各 r 下计算的 physical23 特征做 controller 级选择，在评估标签之前冻结。如果 physical23 不含几何量，A3 就看不到几何变化，如实记录。

**专家：** 每个 r 下，对 32 个 state 各规划一次。

# 三、预注册的终点与判定

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| HZ1（主） | 按比例切换的时间表在新的 H 下是否仍有优势 | 在 H500 与 H1000 下分别检验：S-rel − 每个 controller 的最佳固定 η（交叉拟合）的平均值 CI 下界 > 0 |
| HZ2 | 按绝对步数切换会不会失效 | H1000 下 S-rel − S-abs，H500 下 S-rel − F13，各报平均值与 CI |
| HZ3 | 按进度触发是否更好 | 三个 H 合并，S-prog − S-rel：CI 下界 > 0 记为"更好"；CI 上界 < 0 记为"更差"；其余记为"无差别" |
| HZ4 | 方法在新的 H 下是否成立 | 每个 H 下，A3（相对映射）− safety baseline 的 CI 下界 > 0，且 A3 相对该 H 下完整菜单 oracle 的平均遗憾 ≤ 3pp |
| GM1 | 方法在新几何下是否成立 | 每个通过能力门的 r 下：S500 − safety baseline 与 A3 − safety baseline 的 CI 下界都 > 0；A3 平均遗憾 ≤ 3pp；伤害数为 0 |
| GM2（描述性） | 几何改变时偏好是否转移 | 每个 r 下的最佳固定臂、S500 − 最佳固定 η；能力门未通过的 r 单独列出 |

**证据等级：** 新标签与预注册终点为 CONFIRMATORY；复用的 round10 标签为原有等级。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| HZ1 通过，HZ2 显示绝对切换在 H ≠ 700 时损失大 | 时间表的本质是阶段，而不是绝对时刻；论文应写成按比例或按进度切换 |
| HZ3 为"更好" | 按进度触发是更好的方法形式，可以作为方法的改进；同时说明按 state 的信息（进度）有用 |
| HZ1 不通过 | 阶段结构依赖 H700 这一特定预算，是方法的局限 |
| GM1 在所有通过能力门的 r 下都通过 | 在 controller 能工作的几何范围内，方法稳健 |
| 多数 r 未通过能力门 | 几何泛化的瓶颈在 controller 本身，而不在修正 |

# 五、交付

按共同协议第五节交付，放在本轮目录。另外：
- p* 的离线计算与冻结收据；
- S-prog 的实现与单元测试；
- 能力门报告。

AXIS_TABLE 的行为 H500、H700、H1000，以及 r = 0.80、0.85、0.90（基准）、0.95、1.00 各一行；未通过能力门的 r 也保留一行，标为缺口。列为共同协议的标准方法，H 的各行再加 S-abs、S-rel、S-prog 三列。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过；
- HZ1–HZ4 与 GM1 的结论及关键数字；
- 时间表应当按绝对时刻、按比例还是按进度切换。

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/axis3_cross_scene_v1/`

# 零、先读

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）。** 其中的源码保护、DB 规则、复用规则、标准方法与指标、标准交付，对本轮全部适用，优先级高于本文件以外的一切。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. 总结报告 `docs/controller_generalization_ring_summary_20261014.md`，重点是第 6、12、15 节。
4. 已有跨场景轮次的报告与代码：
   - round13：`diagnostics/cross_scene_eta_schedule_v1/`；
   - round19b（Four）：`diagnostics/four_controller_axis_v2/`；
   - round20：`diagnostics/cross_scene_probe_selection_v1/`；
   - round23：`diagnostics/cross_scene_fewshot_v1/`；
   - 早期四场景 LOSO 审计：`docs/controller_generalization_audit_20261011.md`。

# 一、要回答的问题：泛化轴 3，Cross-Scene（LOSO）

**定义：** 训练时完全排除目标场景，再分别在 Toy（toy_giveway）、DB（double_bottleneck）、Four（four_way_intersection）、Ring（ring_exchange）上测试。

**已有证据（全部为负结果）：**
- 四场景 LOSO 审计：η 偏好由场景决定，Toy 与 Ring 偏好 η10，DB 与 Four 偏好 η15。
- round13：同一个时间表统一不了各场景。
- round20：探测规则的零样本平均遗憾 30.7pp，最大 76.7pp。
- round23：少样本每臂需要 16 条以上标签。

**缺的部分：** 还没有做过正式的 LOSO 检验，即用场景无关的特征与动作表示，在三个场景上训练一个 critic，再到第四个场景上零样本选择。round20 用的是手工规则，不是训练出来的 critic。本轮补上，作为论文 Cross-Scene 轴的正式结果，无论成败都如实报告。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 与 sha256 收据）

**各场景的 controller 与执行：**
- 在历史审计中，从 round13、round19b、round20、round23 用过的、有兼容 runtime 的 checkpoint 里，为每个场景选最多 6 个 controller，写入预注册。
- 修正位置沿用各场景在 round13 中的设定，写入预注册。
- 各场景的执行代码分别过复现门：各复现 16 条历史记录（共同协议第三节）。
- 某个场景兼容的 controller 少于 2 个，或复现门不通过，就把该场景记为缺口，继续其余场景。
- **能力门：** 每个 controller 在 16 个 pilot state 上，用 F10 与该场景已知的偏好臂各跑 seed 0。两个臂都低于 15% 或都高于 98% 的 controller 排除，不替换。

**场景无关的动作菜单：** 各场景的 horizon H 沿用原设定，切换时刻按 H 的比例定义。

| 臂 | 内容 |
|---|---|
| nominal only、safety baseline | 基线（safety baseline 即 η = 0） |
| F10、F13、F15 | 全程固定 η |
| S13→10@f | 前 round(f·H) 步用 η13，之后用 η10；f ∈ {3/7, 5/7} |
| S15→10@f | 前 round(f·H) 步用 η15，之后用 η10；f ∈ {3/7, 5/7} |

共 9 个臂。η10、η13、η15 的精确 float64 值从共享 DB 的 eta 表读取。Ring 在 H700 下的 S13→10@5/7 就是 S500。

**state：** 每个场景 32 个新 state，用各场景原有的生成器，种子区间写入预注册，并与 DB 去重。seed 0–1。

**规模：** 最多 4 × 6 × 32 × 2 × 9 = 13,824 条。已有同身份的记录通过 preflight 复用。

**探测：**
- 对每个通过能力门的 controller，在各自场景中做 R5 式探测。
- 定义沿用 round20 的跨场景探测；round20 未覆盖的场景，按同样的定义补齐，写入预注册。

**场景无关的 critic（LOSO）：**
- **特征**（全部写入预注册）：
  - **state 特征：** 只用每个场景都能算的量，例如机器人数、到目标距离的统计量（按场景尺度归一化）、最小机器人间距与最小障碍间隙（按机器人半径归一化）、初速度统计量（按 max_speed 归一化）。
  - **controller 特征：** R5 探测的 5 个统计量，在训练场景内标准化。
  - **动作特征：** 前期 η 的 3 个分量、后期 η 的 3 个分量、切换比例 f。固定臂的前期与后期 η 相同；η = 0 的三个分量都为 0。
- **模型：** 与 A3 相同的 MLP 64/64/16，fit seed 17、23、41，超参数与选 step 的规则在预注册中写死。
- **LOSO：** 对每个目标场景，只用另外三个场景的标签训练。可以并入另外三个场景中与本轮动作身份兼容的已有标签，写入预注册。
- **选择：** 在目标场景上，对每个 controller 在 7 个修正臂中做 controller 级选择。

**对照：**
- 源场景平均动作：在三个训练场景上平均成功率最高的臂；
- Ring 的默认时间表：S13→10@5/7；
- 一律 F10；
- 目标场景的 oracle（交叉拟合与样本内两种）；
- 目标场景内的少样本：k ∈ {1, 2, 4, 8, 16} 条每臂，按 round23 的方法，随机种子写入预注册。

# 三、预注册的终点与判定

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| XS1 | LOSO critic 是否优于源场景平均动作 | 在四个目标场景合并的结果上，LOSO critic 的平均遗憾低于源场景平均动作，配对 CI 上界 < 0 |
| XS2 | 实用标准 | LOSO critic 的平均遗憾 ≤ 5pp 的目标场景数 ≥ 3 |
| XS3（描述性） | 阶段结构是否各场景共有 | 每个场景中，最佳时间表 − 最佳固定 η（交叉拟合），以及哪一族臂胜出 |
| XS4（描述性） | 少样本代价 | 达到遗憾 ≤ 3pp 所需的目标场景标签数 |
| XS5（描述性） | 修正本身是否在每个场景都有用 | 每个场景中，最佳修正臂 − safety baseline，以及从 nominal only 到 safety baseline 的转移表 |

**证据等级：** 本轮的新标签与预先写死的 LOSO 流程为 CONFIRMATORY；已有轮次的结果为原有等级。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| XS1、XS2 都通过 | 用场景无关的特征与动作表示，可以零样本跨场景选择修正 |
| XS1 通过，XS2 不通过 | 跨场景信息有用，但不够实用；论文报告所需的少样本代价（XS4） |
| XS1 不通过 | 跨场景零样本不可行，是方法的局限。结合 XS3 与 XS5 说明原因：η 偏好由场景决定，阶段结构不共有 |
| XS5 中修正在每个场景都有用 | 即使选择不能跨场景，"安全层 + 修正"这一结构本身在各场景都成立 |

# 五、交付

按共同协议第五节交付，放在本轮目录。另外：
- 各场景 controller 清单、复现门与能力门报告；
- 探测特征、LOSO critic 的 checkpoint 与收据；
- 少样本曲线。

AXIS_TABLE 的行为四个目标场景各一行（另加合并一行），列为共同协议的标准方法，再加"LOSO critic""源场景平均动作""Ring 默认时间表"三列。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过；
- XS1、XS2 的结论与关键数字；
- 各场景的一句话总结。

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/axis4b_agent_count_v1/`

# 零、先读

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）。** 其中的源码保护、DB 规则、复用规则、标准方法与指标、标准交付，对本轮全部适用，优先级高于本文件以外的一切。
2. `AGENTS.md`，特别是"可扩展场景母版"一节；`shared_rollout_db/README.md`。
3. `docs/scalable_scene_audit_20261007.md`、`bottleneck_family/README.md`。
4. 机器人数量相关的已有资产（只读）：
   - `bottleneck_family/` 下的环境、专家、互惠避碰基线（`evaluate_reciprocal_baseline.py`）、Flow 评估与安全审计脚本；
   - `diagnostics/gap_flow_*` 下的能力与安全审计；
   - 冻结的 Gap1 Flow，例如 `diagnostics/gap_flow_scale_20261007/frozen/`；
   - `datasets/gap1_eta_scaling_v1` 至 `v4`：N = 2、10、20 的 η 标签，均未完成。N = 10 时 1,343 条有效 rollout 中只有 46 条成功，其余几乎全部超时。

# 一、要回答的问题：泛化轴 4，Cross-Factor 中的机器人数量

**定义：** 场景族与任务不变，改变机器人数量 N。

Ring 环境写死为 4 个机器人，所以本轮使用 AGENTS.md 指定的可扩展母版 `bottleneck_family`（Gap 双向交换）。

**本轮问题：**
1. N 增大时，"安全层 + 修正"是否仍然有用？
2. 在小 N 上选出的修正动作，能否零样本用到更大的 N？
3. 最佳的修正动作族会不会随 N 改变？

**AGENTS.md 的硬性要求（本轮严格遵守）：**
- 固定地图的数量扫描保持机器人半径、动力学与任务不变；新增的机器人必须参与对共享资源的竞争。
- 容量不足或通道封死，如实报告，不能靠缩小机器人来掩盖，也不能把几何上不可达标成 deadlock。
- 区分五层证据：几何有效、单机器人有路、联合可解、controller 成功、死锁解决。
- `stalled_groups` 只是诊断候选，正常排队也可能满足，不得称为 deadlock。
- 现有固定数量的 checkpoint 不能宣称直接支持其他 N。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 与 sha256 收据）

## 2.1 第 0 步：资产审计（先做，写入预注册）

对 N ∈ {2, 10, 20}，以及有冻结的、经审计有能力的 N = 50 Flow 时的 N = 50，逐一列出：
- 各 N 的环境设定与原 horizon H_N；
- 有没有能用的专家；
- 各 N 专用的冻结 Flow 及其能力证据；
- **与 N 无关的 controller：** 同一个实现可以原样在多个 N 上运行的 controller，例如互惠避碰基线，或者一个 checkpoint 能在多个 N 上运行的置换等变 Flow；
- 已有标签与本轮动作身份的兼容性。

找不到的项记为缺口，并写明原因。

## 2.2 controller

- **N 专用 Flow：** 每个 N 用该 N 的冻结 Flow。不同 N 之间 controller 不同，所以只能分别报告各 N 的结果。
- **与 N 无关的 controller：** 同一个 controller 跑所有 N，这是机器人数量泛化的干净检验。
  - 至少用互惠避碰基线（若审计确认它可以作为 nominal controller）；
  - 另外写一个通用势场 controller：目标吸引、机器人间排斥、墙与障碍排斥，增益沿用 round27 势场 P2 的值，几何从环境读取。写在本轮目录中并配单元测试。
- 若一个与 N 无关的 controller 都没有，transfer 检验（AC2）记为缺口，其余照做。

## 2.3 动作菜单（与 round34 相同，便于跨轴比较）

nominal only、safety baseline（η = 0）、F10、F13、F15、S13→10@f 与 S15→10@f（f ∈ {3/7, 5/7}，切换时刻为 round(f·H_N)），共 9 个臂。η 的精确值从共享 DB 读取。修正位置按 bottleneck_family 已有的执行方式，写入预注册。

## 2.4 state、门与规模

- **state：** 每个 N 用 bottleneck_family 原有的生成器，生成 32 个新 state（种子区间写入预注册，与 DB 去重）；seed 0–1。
- **复现门：** 使用的每段历史执行代码，各复现 16 条历史记录（共同协议第三节）。不一致就把依赖它的子项记为缺口。
- **能力门（每个 (N, controller)）：**
  - 16 个 pilot state，safety baseline、F10、S13→10@5/7 三个臂，seed 0；
  - 三个臂中最高的成功率 ≥ 20% 才进入主评估；
  - 否则记为第 4 层（controller 成功）的缺口，不得称为修正失败或 deadlock。
- **专家：** 每个 N、每个 state 规划一次（若有专家），用于第 3 层"联合可解"。
- **规模：** 最多 3 个 N × 3 个 controller × 32 × 2 × 9 = 5,184 条，另加能力门与专家。N = 50 若纳入，单独列出规模。

## 2.5 transfer 检验

- **源 N：** 通过能力门的最小 N，写入预注册，预期为 N = 2。
- 对每个与 N 无关的 controller，在源 N 上选出它的最佳修正臂（交叉拟合），原样用到更大的目标 N。
- 另做"不看 controller"的版本：取源 N 上所有 controller 平均最好的臂。
- 用目标 N 上的标签评估遗憾（相对目标 N 的完整菜单 oracle，交叉拟合）。

# 三、预注册的终点与判定

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| AC1 | N 增大时修正是否仍有用 | 在每个通过能力门的目标 N 上，从源 N 迁移来的臂 − safety baseline 的 CI 下界 > 0 |
| AC2 | 选择能否跨 N 迁移 | 对与 N 无关的 controller，迁移选择的平均遗憾 ≤ 5pp |
| AC3（描述性） | 最佳动作族是否随 N 改变 | 每个 N 下胜出的臂族（固定 η10 / η13 / η15、两种时间表），以及最佳时间表 − 最佳固定 η |
| AC4（描述性） | 五层证据 | 每个 N 一行：几何有效、单机器人有路、联合可解（专家成功率）、controller 成功（nominal only 与 safety baseline）、修正效果 |

**证据等级：** 新标签与预注册终点为 CONFIRMATORY；已有资产为原有等级。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| AC1、AC2 都通过 | 修正与选择可以跨机器人数量迁移（限 bottleneck_family 与所测的 N） |
| AC1 通过，AC2 不通过 | 修正在大 N 下仍有用，但最佳动作随 N 改变，需要按 N 重新选择 |
| 多数 (N, controller) 未通过能力门 | 机器人数量泛化的瓶颈在 controller 本身（第 4 层），不在修正；论文写为局限 |
| 第 3 层不成立（专家也失败） | 该 N 下任务本身不可解，不能计入方法结论 |

# 五、交付

按共同协议第五节交付，放在本轮目录。另外：
- 资产审计；
- 五层证据表；
- 通用势场 controller 的实现与单元测试；
- 能力门报告；
- transfer 检验表。

AXIS_TABLE 的行为每个 N × controller 类型一行；未通过能力门或没有资产的保留一行，标为缺口。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过；
- AC1、AC2 的结论与关键数字；
- 五层证据中在哪一层、哪个 N 上第一次失败。

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、要回答的问题

**已有结论：**
- **Ring 上，跨 controller 的 η 泛化失败已经定位并修复。**
  - 根因是 H_compress：整局固定 η 把前期效应 E 与后期效应 L 压缩成了一个符号。
  - 修复是“时间表动作 + 由 200 步无标签探测（R5）预测 E”。
  - 已在同 run 兄弟 controller 上前瞻确认（`diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/`）。
- **跨场景的失败不是同一个根因**（`diagnostics/cross_scene_eta_schedule_v1/`）。

**本轮问题：** 在第二个场景 four_way_intersection 中，不同真实 Flow controller 之间是否也存在 η regime 差异（即跨 controller 泛化问题）？如果存在，它是否也是阶段压缩？E 是否同样能从无标签探测中学到？

**可用资产：**
- `diagnostics/four_way_intersection_stage1/` 下有 23 个 Flow 训练 run，每个都有 best.pkl 和 final.pkl。
- 共享 DB 中只有 3 个 best.pkl 有记录（设计者已只读核对，你需要复核）：
  - v13_broad_global_source_balanced：约 28 万条，也是 E8 四场景实验所用的 Four Flow；
  - v12_..._canonical：1,360 条；
  - v10_..._wide：1,024 条。
- 其余约 20 个从未评估过。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## runtime 与切换

- **runtime：** 使用 E8（`diagnostics/controller_ff_factorial_repair_v1/`）的四场景 FF runtime，以及 round13 的 η 切换包装层（`diagnostics/cross_scene_eta_schedule_v1/`）。
- **controller：** 在 Flow-reference 层使用 flow_isotropic 算子，只替换 Flow checkpoint。
- **兼容性门：** 每个 checkpoint 都要能用同一个 adapter 加载，网络结构与归一化器一致，前向计算可复现。不兼容的记为不可用，不得强行运行。
- **一致性门：** 用 v13_broad best.pkl 的已有记录逐位复现 8 例。

## controller

- 23 个 best.pkl 中所有通过兼容性门的。
- final.pkl 只纳入参数与对应 best 不同的那些，单独列为第二组。

## 阶段 0：普查（headroom 门）

- **state：** 32 个全新的 Four state，按 round13 的流程生成并去重。
- **臂：** F10、F13、F15，seed 0–1。
- **门（满足任一条才进入阶段 1）：**
  - **(a) regime 差异：** 至少 3 个 controller 的最佳固定 η 不同于多数 controller 的最佳固定 η，且差值 ≥ 10pp；
  - **(b) 有提升空间：** 至少 3 个 controller 的最佳固定 η 成功率 ≤ 85%。
- **两条都不满足：** 结论为“在 Four 中，跨 controller 的 η 选择不存在泛化问题”（所有 controller 共享同一 regime，或都已接近满分），报告后停止。

## 阶段 1：时间表与 E / L 分解（门通过后才执行）

**时间表臂（与阶段 0 的 state 和 seed 相同）：**
- S15→10、S10→15，各在 t1 = 471、t2 = 786 切换；
- S13→10，在 786 切换（Ring 上已确认的时间表按比例换算）。

**对每个 controller、每个 η 对计算：**
- E = S − F_late；
- L = S − F_early；
- N = F_early − F_late。

**零样本选择：** 在 controller 间做留一，比较两种动作集上“源 controller 平均最好”的选择：固定集 {F10, F13, F15}，扩展集为全部臂。报告平均遗憾与最大遗憾。

## 阶段 2：E 的可学习性（与阶段 1 并行采集；在读取阶段 1 标签之前冻结特征）

- **探测：** 对每个 controller，在 16 个新 probe state 上，分别用 η10、η13、η15 各运行 200 步，无标签。统计量与 R5 相同（`diagnostics/controller_flow_blend_v1/probe.py`），使用 FP32（参见 round12 的修复记录）。
- **E 的选定：** 用阶段 1 中“扩展集源平均选择”最常选中的时间表族来定义 E。这个选定规则写死在预注册中。
- **预测：** 在 controller 间做留一，用 ridge（α = 1）从探测特征的 PCA-2 预测 E。
- **判定：** Spearman ≥ 0.6，且 RMSE ≤ 源均值基线的 0.7 倍，即判定 E 可学（与 Ring 的门槛相同）。

## 规模

- 阶段 0：约 23 × 32 × 3 × 2 = 4,416 条。
- 阶段 1：约 23 × 32 × 5 × 2 = 7,360 条。
- 阶段 2：约 23 × 48 次短探测。

Four 的 horizon 是 1100 步。

# 三、结论映射

| 观察结果 | 结论 |
|---|---|
| 阶段 0 的门不通过 | 跨 controller 的 η 泛化问题在 Four 中不存在。Ring 上的问题有场景特异性，不必推广 |
| 门通过，扩展集明显降低留一遗憾（平均遗憾 ≤ 固定集的 1/3），且 E 可学 | H_compress 与“由探测学到的 E”在第二个场景中也成立，controller 轴的结论可以推广 |
| 门通过，但时间表不能降低遗憾 | Four 中 controller 间的 regime 差异另有机制。报告失败类型与发生时间 |

# 四、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/four_controller_census_v1/` 中写文件。不修改、不删除历史资产（包括 checkpoint 与 PoC 目录），不做 git commit。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。唯一允许的改变，是在第 t 步替换 η。
- **参数写死：** 门、阈值与时间表参数已在本文件中写定，不得根据结果调整。
- **unknown：** 一律保留，不重跑，不算作失败。Four 没有 deadlock 判据，timeout 只能称为 timeout。
- **执行流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。
- **资源：** 本 workstation 上用户所有任务（包括并行的 round14A）合计不超过用户当前授权的总上限。最近一次授权为 24 个 CPU shard、3 个 GPU shard；如有更新，以用户最新指示为准。
- **视频：** 按 AGENTS.md 交付 four_way_intersection 的完整仿真视频。Four 没有 deadlock 判据，缺失的死锁角色如实报告。

# 五、交付

全部交付物放在 `diagnostics/four_controller_census_v1/`：
- PREREGISTRATION.md 及收据；
- 兼容性审计与一致性门报告；
- 普查表：每个 controller × 3 个固定 η；
- 阶段 1、阶段 2 的结果（若执行）；
- results.json；
- 图表；
- 视频及 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：Four 中是否存在跨 controller 的 η 泛化问题；若存在，它是否也是阶段压缩，以及 E 能否学到。

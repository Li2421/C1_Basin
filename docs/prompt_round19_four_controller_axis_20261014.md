你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，以及总结报告 `docs/controller_generalization_ring_summary_20261014.md`，并严格遵守 AGENTS.md。

# 一、要回答的问题

Ring 上 controller 泛化的结论已经确立（见总结报告）：
- **根因 H_compress：** 整局固定 η 会把前期与后期效应压缩成一个符号。
- **方法：** 时间表动作 + 200 步无标签行为探测 + 通用 critic。
- **前瞻确认：** 已三次通过。

**本轮问题：这个结论能否推广到第二个场景 four_way_intersection？** 具体分三层：
1. Four 中不同真实 Flow controller 之间，是否也存在 η regime 差异（即 controller 泛化问题）？
2. 若存在，它是否也是阶段压缩（时间表能否消除差异）？
3. 前期效应 E 能否从无标签探测中学到？

# 二、为什么现在能做（上一次 round14B 失败的原因）

round14B（`diagnostics/four_controller_census_v1/`）只找到 1 个与 E8 四场景 runtime 兼容的 Four Flow，因而判为 INCONCLUSIVE。设计者只读核对后，发现其中有可修正的部分（你需要独立复核）：

- **可用的 Flow：** `diagnostics/four_way_intersection_stage1/` 的 22 个 run 中，有 13 个 best.pkl 的 environment_fingerprint 与参考 Flow v13_broad 相同（前缀 `73910125d2`）。
  - 其中 12 个与 v13 的网络结构、config 完全一致，唯一差别是 4 个归一化数组（obs_mean、obs_scale、act_mean、act_scale）。这些数组本来就由各自的训练数据决定，属于每个 Flow 的正常部署参数。
  - v10_..._wide 另有不同的 actor_hidden_dims。
- **逐 checkpoint 加载有先例：** 其中两个 Flow 已在历史实验中，以“逐 checkpoint 加载”的方式运行过，DB 里有记录：
  - v12_..._canonical：1,360 条；
  - v10_..._wide：1,024 条；
  - 来源为 `exp_orthoflow3_controller_intervention_*`，相关代码在 `diagnostics/orthoflow3_controller_intervention_generalization_v1/` 等目录。
- **排除的 Flow：** 另外 9 个 run 是为不同的环境版本训练的：7 个前缀为 `849d4b1f68`，2 个 lane_local 前缀为 `32732ec568`。本轮不纳入，以免把“controller 差异”和“环境版本差异”混在一起。

# 三、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 0：逐 checkpoint 适配层与等价门（任一门不通过，就停下报告）

**适配层：**
- 在 E8 的四场景 FF runtime（`diagnostics/controller_ff_factorial_repair_v1/`；Flow-reference 层，flow_isotropic 算子）中，改为从每个 checkpoint 读取它自己的归一化参数与架构。
- 加载方式参照历史 intervention runtime 中“替代 Flow”的加载路径。
- η 切换沿用 round13 的包装层（`diagnostics/cross_scene_eta_schedule_v1/`）。
- 不得修改 PoC 目录或任何历史代码，只在本轮目录中实现。

**等价门：**
- **G0：** 用 v13 best 逐位复现 E8 的已有记录，8 例：终止原因、步数、首动作的字节。对 v13，适配层应当与原 runtime 完全相同。
- **G1：** 用 v12 best 与 v10_wide best，在历史 intervention 实验原有的 conditioning 下，各逐位复现 8 条已有记录。这一项验证“逐 checkpoint 归一化”的加载方式与历史一致。
- **G2：** 对其余 Flow，检查结构与 config：除 4 个归一化数组（以及 v10_wide 的宽度）外必须完全一致；前向计算同 key 重复时必须字节一致；导入后显式恢复 FP32（参考 round12 的修复）。

## controller

- 13 个环境一致的 best.pkl，加上其中参数与 best 不同的 final.pkl，作为第二组（按 round14B 的审计约为 11 个）。
- 在预注册中逐个列出。

## 阶段 1：无标签探测（在任何成功标签之前完成并冻结）

- 生成 16 个新的 Four probe state，与 DB 去重。
- 对每个 controller，分别用 η10、η13、η15 各运行 200 步，记录与 R5 相同的统计量。
- 冻结特征，并留收据。

## 阶段 2：普查与 headroom 门

- **数据：** 32 个全新的 Four state，与 DB 去重；臂为 F10、F13、F15；seed 0–1。
- **门（满足任一条才进入阶段 3）：**
  - **(a) regime 差异：** 至少 3 个 controller 的最佳固定 η 不同于多数 controller 的最佳固定 η，且差值 ≥ 10pp；
  - **(b) 有提升空间：** 至少 3 个 controller 的最佳固定 η 成功率 ≤ 85%。
- **两条都不满足：** 结论为“在 Four 的这批 controller 中，不存在跨 controller 的 η 泛化问题（它们共享同一 regime，或已接近满分）”。报告后停止。这本身是有意义的结论：说明 Ring 上的问题有场景特异性。

## 阶段 3：时间表与 E / L 分解（门通过后才执行）

**时间表臂（与阶段 2 的 state 和 seed 相同）：**
- S15→10、S10→15，各在 t1 = 471、t2 = 786 切换；
- S13→10，在 786 切换。

**对每个 controller、每个 η 对计算：**
- E = S − F_late；
- L = S − F_early；
- N = F_early − F_late。

**零样本选择：** 在 controller 间做留一，比较两种动作集上“源 controller 平均最好”的选择：固定集 {F10, F13, F15}，扩展集为全部臂。报告平均遗憾与最大遗憾。

**判定 H_compress_Four：** 满足以下两条即判定成立：
- 扩展集的平均遗憾 ≤ 固定集的 1/3；
- 至少一半 controller 在某个 η 对上同时满足 E ≥ 10pp 与 L ≥ 10pp。

## 阶段 4：E 的可学习性（与阶段 3 一起分析，只用冻结的特征）

- **E 的选定：** 用阶段 3 中“扩展集源平均选择”最常选中的时间表族来定义 E。这个规则写死在预注册中。
- **预测：** 在 controller 间做留一，用 ridge（α = 1）从探测特征的 PCA-2 预测 E。
- **判定：** Spearman ≥ 0.6，且 RMSE ≤ 源均值基线的 0.7 倍，即判定 E 可学（与 Ring 的门槛相同）。
- **兄弟泄漏：** 同 run 的 best 与 final 必须同时留出。

## 规模

| 阶段 | 规模 |
|---|---|
| 阶段 2 | 约 24 × 32 × 3 × 2 ≈ 4,608 条 |
| 阶段 3 | 约 24 × 32 × 5 × 2 ≈ 7,680 条 |
| 阶段 1 | 约 24 × 48 次短探测 |

Four 的 horizon 是 1100 步。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| 阶段 0 的任一门不通过 | 无法可靠地部署其他 Four Flow，C 方向 INCONCLUSIVE。报告具体是哪一项不一致 |
| 阶段 2 的门不通过 | Four 中不存在跨 controller 的 η 泛化问题。controller 轴的问题是 Ring 特有的，或依赖场景 |
| 门通过，H_compress_Four 成立，E 可学 | Ring 的根因与方法推广到第二个场景。下一步可以在 Four 上训练 critic 并做前瞻确认 |
| 门通过，但 H_compress_Four 不成立 | Four 中 controller 间的 regime 差异另有机制。报告失败类型与发生时间（Four 没有 deadlock 判据，只有 timeout） |

# 五、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/four_controller_axis_v1/` 中写文件。不修改、不删除历史资产（包括 checkpoint、PoC 目录与历史 runtime），不做 git commit。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。唯一允许的改变，是读取各 checkpoint 自身的归一化参数，以及在第 t 步替换 η。
- **参数写死：** 门、阈值、时间表参数与 controller 列表已写定，不得根据结果调整。
- **冻结时序：** 探测特征必须在任何成功标签之前冻结。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **执行流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。适配层下的新 controller 登记为新的执行身份。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。提交前查看实时占用。
- **视频：** 这是 Four 场景的 controller / 时间表评价 batch，按 AGENTS.md 交付 four_way_intersection 的完整仿真视频。Four 没有 deadlock 判据，缺失的死锁角色如实报告缺口，不得用 timeout 冒充。

# 六、交付

全部交付物放在 `diagnostics/four_controller_axis_v1/`：
- PREREGISTRATION.md 及收据；
- 适配层代码与 G0 / G1 / G2 等价门报告；
- 探测特征与冻结收据；
- 新 state 的去重审计；
- 普查表；
- 阶段 3 与阶段 4 的结果（若执行）；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表；
- 视频及 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答三件事：
1. 等价门是否通过；
2. Four 中是否存在跨 controller 的 η 泛化问题；
3. 若存在，H_compress 与“由探测学到的 E”是否在 Four 上成立。

每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE，事实与解释分开写。

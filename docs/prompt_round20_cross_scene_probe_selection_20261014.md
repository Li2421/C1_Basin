你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，以及总结报告 `docs/controller_generalization_ring_summary_20261014.md`（重点是第 6 节与第 11 节）。严格遵守 AGENTS.md。

# 一、要回答的问题：跨场景的 η 选择能否零样本完成？

最初的四场景 LOSO 失败，其根因不是动作压缩，而是**各场景在同一阶段需要相反的 η**（round13，`diagnostics/cross_scene_eta_schedule_v1/`）：
- Toy：前期用 η15 或 η13 会让行死锁；
- DB：前期用 η13 或 η10 会堵在瓶颈。

源场景平均选择的 LOSO 遗憾即使加入时间表也有 44pp（固定 η 时为 68pp）。

设计者对 round13 已有标签的事后观察（DEVELOPMENT；成功率为同一批 3 个 Flow 算子 × 40 state × Q4 的结果，你需要复核）：

| 场景 | S13→10@3H/7（成功率 / 遗憾） | S15→10@5H/7（成功率 / 遗憾） | 场景 oracle |
|---|---|---|---|
| Toy | 75.2%（= oracle） | 61.0%（−14pp） | S13→10@3H/7 |
| DB | 26.0%（−74pp） | 98.75%（−1pp） | F15 |
| Four | 99.0%（−0.8pp） | 99.6%（−0.2pp） | F15 |
| Ring | 95.4%（−1.5pp） | 94.4%（−2.5pp） | S15→10@3H/7 |

**这张表说明：**
- 用一个只有两个时间表的菜单，每个场景都能做到 ≤ 2.5pp 的遗憾，前提是选对；
- 真正需要的场景信息只有一点：这个场景是“前期弱 goal 会让行死锁”（Toy 型），还是“前期强 goal 会堵塞”（DB 型）；
- 这两种失效都发生在前期，而且表现为机器人停滞。

**由此提出要检验的方法（与 Ring 上 controller 轴的思路一致）：** 在新场景中，用无标签的前期探测观察 η13 与 η15 下是否出现停滞，再从两个时间表中选一个。这条规则不学习任何参数。

由于这条规则是看过四个原场景的数据之后设计的，原场景上的结果只能算 DEVELOPMENT。**本轮的主检验，是在新的场景几何变体上做前瞻评估。**

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 2.1 runtime 与 controller

- **runtime：** E8 的四场景 FF runtime（`diagnostics/controller_ff_factorial_repair_v1/`），加上 round13 的 η 切换包装层。
- **controller：** 每个场景使用自己的 Flow，取 Flow-reference 层的 3 个算子：flow_isotropic、flow_anisotropic、flow_coupled。
- **几何变体：** 只改变环境中的几何参数。round13 的 runtime 已经可配置这些参数；round 早期的 Toy 宽度族（`diagnostics/controller_state_transfer_v1/`）已经用过 Flow 算子跑不同的 corridor_width，可作先例。
- **一致性门：** 先用原几何复现 round13 的已有记录 8 例（逐位）。

## 2.2 新场景变体（预注册清单）

| 族 | 参数 | 原值 | 变体 |
|---|---|---|---|
| Toy | corridor_width | 0.40 | 0.44、0.50、0.56 |
| DB | bottleneck_width | 0.50 | 0.60、0.75、0.90 |
| Ring | obstacle_radius | 0.90 | 0.75、1.05 |

共 8 个变体。

**有效性门（预先写定，只用独立的 pilot state）：**
- 每个变体取 16 个 pilot state，与评估 state 不同，在 F10、F15 两个臂上用 flow_isotropic 各跑 1 个 seed。
- 若两个臂的成功率都低于 20%，判定该变体中 Flow 无法完成任务，排除并报告。不得替换成其他变体。
- 另外检查几何有效性：机器人能放得下，通道不封死。容量不足要报告，不得缩小机器人。

## 2.3 无标签前期探测（在任何评估标签之前完成并冻结）

**对象：** 每个有效变体，以及 4 个原场景（原场景的结果作为 DEVELOPMENT 参照）。

**探测：**
- 16 个新 probe state；
- 用 flow_isotropic 算子，分别以 η13、η15 运行前期 round(3H/7) 步；
- 关闭成功、碰撞与死锁的终止，不读取任何终止或成功标签。

**统计量（逐 state 计算，再取平均）：** 停滞机器人比例（速度 < 0.01、持续 ≥ 2 秒，即连续 40 步）、进展比例、最小两两距离。

**选择规则 R（预注册，不学习）：**
- 若 stall(η15) > stall(η13) + 0.05，选 S13→10@3H/7。这类场景是 Toy 型：前期弱 goal 会停滞。
- 否则选 S15→10@5H/7（默认项，round13 中它在 3/4 个场景里接近最优）。

特征与每个变体的选择都要冻结并留收据。

## 2.4 评估标签（前瞻）

- **规模：** 每个有效变体 × 3 个算子 × 40 个全新 state × 7 个臂（F10、F13、F15、S13→10@{3H/7, 5H/7}、S15→10@{3H/7, 5H/7}）× seed 0–1。8 个变体全部有效时约为 **13,440 条**。
- **切换步：** 按各族的 horizon 计算：Toy、DB 为 364 / 607；Ring 为 300 / 500。
- **流程：** 按 shared_rollout_db 流程执行：design → preflight（含探测的 probe preflight 收据，round19b 漏掉过这一项）→ 只跑缺失 → journal → 单进程 merge → postflight。

# 三、预注册终点与判定

oracle 为每个变体在 7 个臂中的最佳臂，场景成功率取 3 个算子的平均。所有 CI 按 state 成簇做 bootstrap。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| D1（主） | 规则 R 能否零样本选好 | 在全部有效变体上，规则 R 的平均遗憾 ≤ 3pp，且最大遗憾 ≤ 10pp |
| D2 | 场景探测是否带来价值 | 与“源场景平均选择”相比（用 round13 四个原场景的平均成功率在 7 个臂中取 argmax，在预注册中写明是哪一个），规则 R 的平均遗憾更低，配对 CI 上界 < 0 |
| D3 | 两个时间表的菜单是否足够（描述性） | 每个变体中，菜单里较好的那个时间表与 oracle 相差 ≤ 3pp 的比例 |
| D4 | 原场景上的参照（DEVELOPMENT） | 规则 R 在 4 个原场景上的选择是否与 round13 的事后最优一致。只作描述，不参与判定 |

**额外报告：** 每个变体的 stall(η13)、stall(η15)、规则的选择、7 个臂的成功率，以及失败类型与发生时间。Toy、DB 有真实死锁判据。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| D1、D2 都通过 | 至少在这三个几何族内，跨场景的 η 选择可以零样本完成：场景级的“regime”可以归结为一个可在前期观察到的失效模式（让行停滞 vs 瓶颈堵塞），用两个时间表即可覆盖。下一步是推广到新的拓扑（例如 bottleneck_family） |
| D3 不满足（菜单不够） | 新变体需要别的时间表。写明是哪些变体、缺的是什么动作 |
| 菜单足够，但 D1 不通过 | 短时探测识别不出正确的时间表，场景信息在前期不可观察。报告规则选错的变体，以及它们的停滞统计 |
| 有效变体少于 4 个 | 证据不足，记为 INCONCLUSIVE |

**结论边界：** 这里检验的是同一拓扑族内的几何变化（改宽度、半径），不是全新的拓扑。不得把结论表述为“任意新场景”。

# 五、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/cross_scene_probe_selection_v1/` 中写文件。不修改、不删除历史资产（包括 PoC 目录），不做 git commit。
- **不得改动核心组件：** 不修改控制器、物理、安全投影、事件判定或成功标准。允许的改变只有两项：环境的几何参数，以及在第 t 步替换 η。
- **参数写死：** 变体清单、有效性门、探测定义、规则 R、阈值与菜单都已在本文件中写定，不得根据结果调整。
- **冻结时序：** 探测与规则的选择，必须在任何评估标签之前冻结并留收据。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock；只有 Toy、DB 的原判据触发时，才能称为 deadlock。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。
- **视频：** 这是跨场景的评价 batch，按 AGENTS.md 与 `docs/batch_videos.md` 交付 toy_giveway、double_bottleneck、ring_exchange 三个场景的视频。
  - Toy 与 DB 有真实死锁判据，应能给出正式的 `deadlock_show` 与 `deadlock_resolution`：同一初态、同一 seed，固定 η 的死锁基线对比时间表的成功轨迹。
  - 每个场景另需一个 `safety_success`。
  - Ring 没有死锁判据，如实报告缺口。
  - launcher 最后一步接入视频入口，缺视频时以非零退出码结束。

# 六、交付

全部交付物放在 `diagnostics/cross_scene_probe_selection_v1/`：
- PREREGISTRATION.md 及收据；
- 一致性门与有效性门报告；
- 探测特征、规则选择与冻结收据；
- state 去重审计；
- DB 的 preflight（包括 probe preflight）与 postflight 记录；
- results.json；
- 图表：每个变体 7 个臂的成功率与规则的选择，以及 stall(η13) 对 stall(η15) 的散点图；
- 视频及 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：在新的几何变体上，无标签前期探测加两个时间表的规则，能否零样本选好 η（D1、D2），以及两个时间表的菜单是否足够（D3）。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE，事实与解释分开写。

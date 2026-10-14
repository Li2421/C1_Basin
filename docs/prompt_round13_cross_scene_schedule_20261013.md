你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、要回答的问题

在 Ring 上已经得到前瞻确认（round12，`diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/`）的根因是 **H_compress**：
- “整局固定一个 η”把前期效应 E 和后期效应 L 压缩成了一个符号；
- 改用“前期一个 η、后期换 η10”的时间表，就能消除 controller 之间的 regime 差异。

**本轮问题：最初四场景 LOSO 泛化失败的根因，是不是同一个？**

最初的失败见 `docs/controller_generalization_audit_20261011.md` §4。场景按最优 η 分成两种 regime：

| 场景 | 最优固定 η | 各 η 成功率 | 主要失效方式 |
|---|---|---|---|
| Toy | η10 | η10 0.61 | 让行死锁 |
| Ring | η10 | η10 0.79，η15 0.18 | η15 太慢 |
| DB | η15 | η15 0.99，η10 0.17 | η10 堵瓶颈 |
| Four | η15 | η15 1.00，η10 0.86 | — |

注：Ring 一行中 η15 的成功率 0.18，以及 Ring、DB 两行的失效方式（η15 太慢，中位成功步 496 vs η10 的 408；η10 有 63% 的 rollout 触发候选死锁，η15 能过但慢，中位成功步 773/850），引自审计文档。

LOSO 中 η-only 选择等于“源场景平均后取 argmax”，遗憾为 0.2–0.86。

**要检验的是：** 前期用一个疏解冲突的 η（η13 或 η15），后期换成 η10，这样的时间表能否在四个场景都不劣于该场景的最佳固定 η，从而消除场景级 regime。

# 二、必读材料（亲自核对）

- `docs/controller_generalization_audit_20261011.md`：§3、§4，以及失效机制部分。
- `diagnostics/controller_ff_factorial_repair_v1/`（E8）：
  - protocol.json：四场景环境配置、Flow 算子、runtime / FF operator 的 sha、state_selection；
  - raw_labels.npz、dataset.npz。
- `diagnostics/controller_eta_switch_time_v1/`、`diagnostics/controller_universal_eta_schedule_v1/`：Ring 上的切换实现（`core.py`）与一致性门做法。
- `diagnostics/controller_phase_decomposition_v1/`：E、L、N 的定义。

需要核对的事实：
- horizon：Ring 700、Toy 850、DB 850、Four 1100 步。
- deadlock 判据：Toy 与 DB 有（window_progress_v2）；Ring 与 Four 没有。
- 每个场景的 Flow-reference 层使用 E8 的 Flow checkpoint 和 FF 算子。
- η 的取值（与 Ring 的 K16 相同）：
  - η10 = (1.246, −0.289, 0.186)；
  - η13 = (1.075, 0.058, 0.266)；
  - η15 = (0.529, 0.308, 0.005)。

# 三、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## runtime

- 使用 E8 的四场景 FF runtime。
- η 切换以包装层实现：从第 t 步起把 η_a 换成 η_b；OrthoFlow3 修正仍每步重算。
- 不修改 PoC 目录或任何历史代码。

## 一致性门（必须先通过）

- 每个场景 8 例，共 32 例。
- 用 E8 已有的 F10 / F15 记录（seed 0–1）逐位复现：终止类型、episode 长度、首动作的字节。
- 不通过就停下报告。

## controller

每个场景取 E8 Flow-reference 层的 3 个算子：flow_isotropic、flow_anisotropic、flow_coupled。

## state

- 每个场景 40 个**全新** state。
- 用与 E8 / 原 benchmark 相同的初态生成流程，换一个预先登记的新种子。
- 与 DB 中的所有 state 去重。

## 臂（每个 controller × state，seed 0–3）

**切换时刻：** 按 horizon H 的固定比例取，t1 = round(3H/7)，t2 = round(5H/7)。

| 场景 | t1 | t2 |
|---|---|---|
| Ring | 300 | 500（与 Ring 上已确认的 S300 / S500 一致） |
| Toy、DB | 364 | 607 |
| Four | 471 | 786 |

**臂：**
- 固定：F10、F13、F15。
- 时间表：
  - S13→10@t1、S13→10@t2；
  - S15→10@t1、S15→10@t2。

**规模：** 4 场景 × 3 controller × 40 state × 7 臂 × 4 seed = **13,440 条完整 rollout**。

**执行流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。切换臂登记为新的执行身份（写入 η_a、η_b、t）。

## 预注册的终点

所有 CI 按 state 成簇做 bootstrap。场景成功率取 3 个 controller 的平均。

**X1（主检验）：Ring 上已确认的时间表能否原样迁移到其他场景。**
- 对象：S13→10@t2。
- 判定：在每个场景中，它不低于该场景最佳固定 η（F10、F13、F15 中取最好）超过 5pp。四个场景都满足才算通过。
- 同时报告每个 controller 的结果与 CI。

**X2（直接对应最初的失败）：LOSO 零样本选择。**
- 对每个留出场景，在 3 个源场景的平均成功率上取 argmax 选动作；遗憾以留出场景 7 个臂中的最佳臂为参照。
- 比较两个动作集：固定集 {F10, F13, F15}，扩展集为全部 7 个臂。
- 判定：扩展集在四个留出场景上的最大遗憾 ≤ 5pp，且平均遗憾 ≤ 固定集平均遗憾的 1/3。

**X3（逐场景 H_compress，描述性）。** 对每个场景 × controller，分别以 η_a ∈ {η13, η15} 作为前期 η，在 t2 下计算：
- E_a = S − F10；
- L_a = S − F_a；
- N_a = F_a − F10。

报告：
- 同时满足 E_a ≥ 10pp 与 L_a ≥ 10pp 的数目；
- 失败类型与发生步数（Toy、DB 有真实的 deadlock 判据）；
- 每个场景里时间表改变的是哪一种失败。

# 四、结论映射（预注册）

| 观察结果 | 结论 |
|---|---|
| X1 或 X2 通过 | **跨场景失败与跨 controller 失败同源（H_compress）**：固定 η 的动作空间压缩了阶段效应，换成时间表动作即可消除场景级 regime。下一步：训练一个覆盖四场景、以时间表为动作的 critic，并在新几何上做 LOSO 确认（例如 `bottleneck_family/`） |
| X1 不通过，X2 通过 | 能消除场景级 regime 的时间表不是 Ring 的那一个。写明是哪一个，下一轮做前瞻确认 |
| X1、X2 都不通过，但 X3 显示各场景都有阶段结构 | H_compress 在各场景内部成立，但最优的阶段 η 因场景而异。场景信息仍然需要，但问题缩小为“在少数时间表中选一个”，报告剩余遗憾 |
| 在 DB、Four、Toy 上时间表都不优于最佳固定 η | H_compress 只在 Ring 成立，跨场景失败另有根因（参见审计中“主导失效机制不同”） |

# 五、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/cross_scene_eta_schedule_v1/` 中写文件。不修改、不删除历史资产（包括 PoC 目录），不做 git commit。
- **不得改动核心组件：** 不修改控制器、物理、安全投影、事件判定或成功标准。唯一允许的改变，是在第 t 步替换 η。
- **参数写死：** 时间表参数与判定阈值都已在本文件中写定，不得根据结果调整。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock；只有 Toy、DB 的原判据触发时，才能称为 deadlock。
- **证据等级：** 新 state 上的结果是前瞻的；其余结论为 DEVELOPMENT。
- **资源：** 本 workstation 上用户所有任务（包括可能并行的 round13B）合计不超过 18 个 CPU shard 和 3 个 GPU shard。提交前查看实时占用。
- **视频：** 这是四场景的跨场景评价 batch，按 AGENTS.md 与 `docs/batch_videos.md` 每个场景都要交付：
  - 两个不同的死锁解决案例（`deadlock_show`、`deadlock_resolution`）：同一初态、同一 seed，一个是固定 η 的死锁基线，一个是时间表的成功；
  - 一个独立的 `safety_success` 案例。

  launcher 的最后一步接入视频入口，缺视频时以非零退出码结束。Ring 与 Four 没有 deadlock 判据，如果没有合格案例，就如实报告缺口，不得用 timeout 冒充。

# 六、交付

全部交付物放在 `diagnostics/cross_scene_eta_schedule_v1/`：
- PREREGISTRATION.md 及收据；
- 切换包装层代码；
- 一致性门报告；
- 新 state 的生成与去重审计；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：四场景 × 7 臂的柱状图、X2 的遗憾对比图、X3 的 E–L 散点图；
- 视频及 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：四场景 LOSO 失败是否与 controller 级失败同源，即 X1、X2 是否通过。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE，事实与解释分开写。

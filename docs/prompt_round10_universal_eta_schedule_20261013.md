你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、要检验的命题（只检验这一个）

**H_schedule：在 Ring 场景中，“先用 η13、到第 t 步换成 η10”这个与 controller 无关的 η 时间表，在所有真实 Flow controller 上都不劣于该 controller 自己的最佳固定 η。** 如果成立，之前各轮看到的“controller 依赖的 η regime”主要是“整个 episode 只能用一个固定 η”这一动作空间限制造成的。

## 依据

依据来自 round9（`diagnostics/controller_eta_switch_time_v1/`，DEVELOPMENT）。这些数据所用的 state 与 seed 0–3 都已被打开过；t = 500 也是看过数据后才挑出的：

| controller | 纯 η10 | 纯 η13 | η13 → η10，t = 500 | 相对最佳固定 η |
|---|---|---|---|---|
| base_u | 23.4% | 53.1% | 84.0% | +30.9pp |
| v10_local | 57.8% | 92.2% | 94.5% | +2.3pp |
| v9_local | 96.1% | 79.3% | 93.8% | −2.3pp |
| plus_w_v4_wide | 63.3% | 26.2% | 96.9% | +33.6pp |

一个可能的机制解释：前期推进需要 η13，后期收敛需要 η10。固定 η 偏好哪一个，取决于该 controller 的失败主要集中在哪个阶段。

本轮的任务，是在新 state 和全部 14 个真实 Flow 上做前瞻检验，其中 10 个 Flow 从未参与时间表的选择。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## controller

全部 14 个真实 Ring Flow。分组如下：
- **schedule-dev 组：** base_u、v10_local、v9_local、plus_w_v4_wide。它们是 round9 中用于选择时间表的 4 个。
- **held-out 组：** 其余 10 个。

## state

- 按 round4 loco_test 的生成分布与流程，用新的、预先登记的随机种子抽取 **64 个全新 state**。
- 抽取后，确认它们与历史 DB 中所有 state 都不重叠。
- 在本轮终点评价之前，不读取这些 state 的任何结果。

## 臂（每个 controller × state，seed 0–3，即 Q4）

| 臂 | 定义 |
|---|---|
| F10 | 全程 η10 |
| F13 | 全程 η13 |
| Freg | 全程使用该 controller 的 Stage 0 regime η，仅对 regime 不是 η10 / η13 的 controller 执行：plus_w_v3 与 v3 用 η2，v8_local 用 η4 |
| S500（主） | 前 500 步 η13，之后 η10 |
| S300、S400（次） | 同上，切换步分别为 300、400 |

- **切换的实现：** 复用 round9 冻结的切换实现（`core.py`）。唯一的干预是从第 t 步起替换 η；OrthoFlow3 修正仍然每步重算。
- **一致性门：** 先在 32 个 case 上复现 round9 的一致性门，结果须逐位一致。
- **规模：** 14 × 64 × 5 × 4 = 17,920，加上 Freg 的 3 × 64 × 4 = 768，**共 18,688 条完整 rollout**。
- **共享 DB：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。切换臂沿用 round9 已登记的身份格式。

## 不变项

Ring、FF、H700、安全投影、成功判定、JAX x64 = false。不修改控制器、物理或事件判定。

## 终点与判定（全部预注册）

对每个 controller，定义 **最佳固定 η = max(F10, F13, Freg)**。它是在同一批数据上事后选出的 oracle，对时间表不利，因此是保守的比较对象。

**主判定 G_universal 以 S500 为准，须同时满足：**
1. **整体收益：** 14 个 controller 上，S500 − 最佳固定 η 的平均值，其 95% 下界 > 0。bootstrap 时按 state 成簇，并在 controller 层重抽。
2. **逐 controller 不劣：** 没有任何 controller 的 S500 比其最佳固定 η 低 5pp 以上（按点估计；同时报告逐 controller 的 CI）。
3. **相对源先验不出灾难：** 没有任何 controller 的 S500 比 F10 低 10pp 以上（E3 = 0）。

**泛化检查：** 对 held-out 组（10 个）单独重复第 1 条和第 2 条。这是对“时间表不是在 4 个 controller 上过拟合”的直接检验。

**次要分析：**
- S300、S400 的同样指标。
- 在 14 个 controller 上，比较时间表的增益与 F10 − F13 的关系，用来检验“固定 η 的 regime 由失败阶段决定”。
- 失败类型与发生步数。

# 三、预注册的结论映射

| 观察结果 | 结论 |
|---|---|
| G_universal 通过，且 held-out 组也通过 | **H_schedule SUPPORTED。** 零样本 controller 泛化失败的根因，是动作空间设定错误：用单个固定 η 压缩了随阶段变化的最优 η，于是“哪个 η 好”变成了一个看不见的 controller 属性。在 Ring 上，一个与 controller 无关的时间表即可实现泛化。下一步是让 critic 学习 Q(s, t, η) 或在中途重新选择 η，并检验其他场景 |
| schedule-dev 组通过，held-out 组不通过 | 时间表部分过拟合，controller 依赖部分是真实的。报告哪些 controller 失败，以及它们的失败阶段 |
| 两组都不通过 | round9 的现象不具普遍性，H_schedule REJECTED |

# 四、硬性规则

- 只在新目录 `diagnostics/controller_universal_eta_schedule_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- 时间表的参数（η13→η10，t ∈ {300, 400, 500}，主 t = 500）已写死，不得根据本轮结果调整。
- unknown 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **证据等级：** 新 state 首次打开前的预注册部分是前瞻的；其余全部标为 DEVELOPMENT。
- **资源：** 本 workstation 上用户所有任务合计不超过 18 个 CPU shard 和 3 个 GPU shard。
- **视频：** 这是 ring_exchange 的大型评价 batch，按 AGENTS.md 和 `docs/batch_videos.md` 交付 ring_exchange 的完整仿真视频。Ring 的原 runtime 没有 deadlock 判据；如果找不到合格的死锁解决案例，就如实报告缺口，不得用 timeout 冒充。

# 五、交付

全部交付物放在 `diagnostics/controller_universal_eta_schedule_v1/`：
- PREREGISTRATION.md 及收据；
- 新 state 的抽取与去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：每个 controller 的 F10 / F13 / Freg / S300 / S400 / S500 柱状图，以及 held-out 组单独一张图；
- 视频及其 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：H_schedule 是否成立，以及在 held-out 组上是否成立。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE；事实与解释分开写，不夸大。结论边界：Ring、FF、H700、2 个 η、14 个真实 Flow。

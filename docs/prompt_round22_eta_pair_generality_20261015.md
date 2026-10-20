你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，以及总结报告 `docs/controller_generalization_ring_summary_20261014.md`（重点是第 14 节）。严格遵守 AGENTS.md。

# 一、要回答的问题：时间表现象是否只限于这两对 η？

目前的证据只来自两个 η 对：
- Ring：η13 → η10；
- Four / DB：η15 → η10。

总结报告第 14.4 节提出了“阶段排序不变性”的解释：
- 若成功率近似可以按阶段分解，那么对一对 η（a, b）而言，**只要方向选对，时间表就会同时优于两个端点的固定 η**；
- “哪个方向对”由各阶段的优劣排序决定，而这个排序在同一场景的 controller 之间是共享的。

**本轮检验：** 对 K16 中**随机抽取**的 η 对，上述预测是否仍然成立。如果成立，“阶段时间表优于固定 η”就可以作为 Ring 上的一般性质写进论文；如果不成立，就说明这一现象依赖于特定的 η 对。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**场景与 controller：**
- 场景：Ring，FF，H700。
- controller：14 个原 Ring Flow（`diagnostics/ring_exchange_stage1/*_macflow/best.pkl`，即 round10 的 controller）。它们覆盖了全部 14 个配方。

**state：**
- 32 个全新 state，按 round10 的流程抽取，并与 DB 中所有 state 去重。
- 切换实现复用 round10 / 15 冻结的 `core.py`；先过 32 例一致性门。

**η 对的抽取（预注册，不看任何结果）：**
- 锚点对：(η13, η10)，用来复现已知结论。
- 随机对：以种子 20261015、`numpy.random.default_rng` 从 K16 的 16 个 η 中抽 6 个不同的无序对，排除锚点对。在预注册中写明抽取代码与结果。
- K16 的 η 取值见 `diagnostics/controller_loco_real_flow_v1/protocol.json`。

**臂（每个 η 对 {a, b}）：**
- 固定：F_a、F_b；
- 两个方向的时间表：S(a→b) 与 S(b→a)，切换步一律为 t = 500（Ring 的默认值）。
- 所有 η 对共用的固定臂只跑一次。

**seed：** 0–1（Q2）。

**规模：** 14 × 32 × (去重后的固定臂数 + 14 个时间表臂) × 2，约 2–2.5 万条完整 rollout。在预注册中写出确切数字。

**流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。

# 三、预注册的预测与判定

所有 CI 都做分层 bootstrap：先在 controller 层重抽，再在 state 层成簇重抽，共 5000 次。成功率把 unknown 当失败（下界），另报上界。

**对每个 η 对与每个 controller，定义：**
- 好方向 = 两个时间表中成功率较高者；
- 支配 = S(好方向) ≥ max(F_a, F_b) − 1pp。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| P1 | 支配 | 在 7 个 η 对中至少 5 个，≥ 70% 的 controller 满足支配 |
| P2 | 方向共享（排序不变性） | 在 7 个 η 对中至少 5 个，好方向在 ≥ 80% 的 controller 上相同 |
| P3 | 收益 | 在 7 个 η 对中至少 5 个，“共享方向的时间表 − max(F_a, F_b)”在 controller 间的平均值，其 CI 下界 > 0 |

“共享方向”指该 η 对中多数 controller 的好方向。

**对照预测（理论的另一面）：** 对每个 η 对，统计固定 η 的偏好符号 sign(F_a − F_b) 在 controller 之间的一致率。理论预期：这个一致率明显低于 P2 的方向一致率，即固定 η 的偏好随 controller 翻转，而时间表的方向不翻转。描述性报告，并给出两者之差的 CI。

**锚点复现：** (η13, η10) 应当复现 round10 的结论：S500 平均优于两个固定臂，且好方向为 η13 → η10。若不复现，先报告再解释。

**额外报告（描述性）：**
- 各随机对中时间表的最大收益；
- 是否有随机 η 对的时间表超过锚点 S500；
- 失败类型与发生时间。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| P1、P2、P3 都通过 | “方向选对的阶段时间表优于固定 η，而且方向在 controller 间共享”是 Ring 上的一般性质，不依赖特定 η 对。论文可以写成一般原理（限 Ring） |
| P1 通过，P2 不通过 | 时间表普遍更好，但方向因 controller 而异，排序不变性只在部分 η 对上成立。选择方向仍需要 controller 信息 |
| P1 不通过 | 支配现象依赖特定的 η 对。论文只能就已验证的 η 对立论，并报告是哪些 η 对、为什么失败 |

# 五、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/ring_eta_pair_generality_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **不得改动核心组件：** 不修改控制器、物理、安全投影、事件判定或成功标准。唯一允许的改变，是在第 t 步替换 η。
- **参数写死：** η 对的抽取种子、切换步、阈值都已写定，不得根据结果调整。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **视频：** 本轮属于纯 Q 函数的数据生成与分析，按 AGENTS.md 的例外不运行视频入口。这一判断在预注册中写定。
- **资源：** 本 workstation 上用户所有任务（包括同时进行的其他分析任务）合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 六、交付

全部交付物放在 `diagnostics/ring_eta_pair_generality_v1/`：
- PREREGISTRATION.md 及收据；
- η 对抽取记录；
- 一致性门与去重审计；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：每个 η 对的四臂成功率，以及“方向一致率 vs 固定偏好一致率”的对比图；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答 P1、P2、P3 是否通过，以及时间表现象是否不依赖特定的 η 对。

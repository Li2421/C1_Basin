你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、本轮的性质：在新的真实 controller 上做前瞻确认

到目前为止的结论都只是 DEVELOPMENT 证据。本轮要在**从未评估过的真实 controller**和**全新 state**上，冻结完整流程后做前瞻确认。

## 新 controller 的来源

`diagnostics/ring_exchange_stage1/*_macflow/` 下的 14 个训练 run，每个都有 best.pkl 和 final.pkl 两个 checkpoint，二者的 sha256 都不同。

- 之前所有轮次用的都是 best.pkl。
- 共享 DB 的 `controller_config.flow_checkpoint_sha256` 中，final.pkl 的出现次数为 0（设计者已只读核对）。

因此，**14 个 final.pkl 是从未评估过的真实 controller**。你要独立复核这一点，并写进预注册。

## 要确认的已有结论（全部为 DEVELOPMENT）

1. **H_compress：** 整局固定 η 的偏好，是两个阶段效应之差的符号：
   - E = S500 − F10（前期 η13 的收益）；
   - L = S500 − F13（后期换回 η10 的收益）。

   来源：`diagnostics/controller_phase_decomposition_v1/`。
2. **时间表动作：** 加入 S300、S500 等时间表动作后，零样本遗憾大幅下降。来源：`diagnostics/controller_universal_eta_schedule_v1/`。
3. **E 可以从 200 步无标签探测中学到：** R5-PCA2 ridge 的 LOO Spearman 为 0.84。规则 G_E 在真实 Flow 的 LOO 中把平均遗憾从 1.03pp 降到 0.64pp。来源：`diagnostics/controller_late_phase_probe_v1/`。

# 二、冻结的流程（在读取任何 final.pkl 的结果之前写死）

**动作菜单：** {F10, F13, S300, S500}。时间表与 round10 相同，切换实现复用 round10 冻结的 `core.py`。

**controller 描述 R5：**
- 对每个新 controller，在 R5 的 16 个 probe state 上分别运行 η10 和 η13 各 200 步。
- 无标签，计算方式与 `diagnostics/controller_flow_blend_v1/probe.py` 完全相同。
- 预处理沿用 round7 / round11 的主方案：删除缺失格得到 155 维；z-score 与 PCA 只在训练 controller 上拟合；新出现的缺失格用训练 controller 的列均值填补。
- **探测在任何成功标签产生之前完成并冻结。**

**Ê 模型：**
- ridge（α = 1）。输入为 R5-PCA2，标签为 round10 中 14 个 best.pkl controller 的 E。
- 主方案：排除同 run 的兄弟 controller。对 final_X，用除 best_X 以外的 13 个 best controller 训练。
- 次方案：用全部 14 个训练。

**选择规则 G_E：** Ê < 5pp 时选 F10；否则选 S500。

**对照：**
- 源平均选择（S500）；
- 旧的源先验 F10；
- 每个 controller 的 oracle。

# 三、实验设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**controller：** 14 个 final.pkl。
- 先检查运行时的兼容性：用同一个 adapter 加载；网络结构与归一化器一致；前向计算可复现。不兼容的 checkpoint 记为不可用，不得强行运行。
- 描述性地报告每个 final 与其兄弟 best 之间的差异：参数距离，以及 R5 特征距离。

**state：**
- 按 round10 的流程，用新的、预先登记的种子抽取 64 个全新 state。
- 确认它们与 DB 中所有 state 都不重叠。

**臂（每个 controller × state，seed 0–3）：**
- 动作菜单中的 F10、F13、S300、S500；
- 另加 F2 与 F4，只作 oracle 参照，用来检测 η2 / η4 型 controller。

**规模：** 14 × 64 × 6 × 4 = **21,504 条完整 rollout**，另有 14 × 32 次短探测。

**执行：**
- 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。
- 先过 32 例一致性门，用 best.pkl 的已有记录逐位复现。

# 四、预注册的终点与判定

所有 CI 都按 state 成簇做 bootstrap。oracle 取 6 个臂中的最佳臂。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| C1 | 完整流程的效果 | G_E 相对 F10 的平均收益，95% 下界 > 0；G_E 的平均遗憾 ≤ 2pp；没有任何 controller 的 G_E 比 F10 低 10pp 以上 |
| C1b | 最大遗憾（次要） | 最大遗憾 ≤ 5pp。若不满足，逐个报告超标的 controller，并根据新标签描述它的类型（例如 η2 型） |
| C2 | controller 信息的价值 | G_E 的遗憾 < 源平均选择的遗憾，用配对 CI 判断 |
| C3 | 学到的 controller–Q 关系能否前瞻迁移 | Ê 与真实 E（来自本轮标签）在 14 个新 controller 上的 Spearman ≥ 0.6，且 RMSE ≤ 源均值基线 RMSE 的 0.7 倍 |
| C4 | H_compress 在新 controller 上是否仍成立 | E > 0 与 L > 0 的 controller 数，以及同时满足 E ≥ 10pp 和 L ≥ 10pp 的数目，与 round10 比较（描述性） |

**C2 的前置条件（headroom 门）：**
- C2 只有在新 controller 中至少 2 个满足“真实 E < 5pp”时才可判定。
- 不满足时记为 INCONCLUSIVE（没有 headroom），不得写成 SUPPORTED 或 REJECTED。
- 这条规则是为了避免重复 round11 融合族检验的问题：那批测试对象没有 headroom，检验失去了区分力。

# 五、结论映射

| 观察结果 | 结论 |
|---|---|
| C1 与 C3 都通过 | 在新的真实 controller 上前瞻确认：时间表动作加上“由行为探测学到的前期分量”能实现零样本 controller 泛化，并且学到的 controller–Q 关系（R5 → E）可迁移。这是本研究线第一个非 DEVELOPMENT 的结论，但局限于 Ring 场景，且新 controller 是同 run 的兄弟 |
| C1 通过，C3 不通过 | 时间表的泛化得到确认，但 R5 → E 的关系没有前瞻迁移。controller 信息的价值仍未确认 |
| C1 不通过 | 报告失败的 controller 及其失败类型与时间，前瞻确认失败 |
| C2 不满足 headroom 门 | 只报告 C1、C3，不对 controller 信息的选择价值下结论 |

# 六、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/controller_new_checkpoint_confirmation_v1/` 中写文件。不修改、不删除历史资产（包括 checkpoint），不做 git commit。
- **冻结时序：** 第二节的全部内容都必须在读取任何 final.pkl 的成功标签之前冻结（带收据）。冻结之后不得修改阈值、菜单或模型。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **资源：** 本 workstation 上用户所有任务合计不超过 18 个 CPU shard 和 3 个 GPU shard。
- **视频：** 本轮评估的是新的控制动作（η 时间表）在新 controller 上的表现，不是纯 Q 实验。按 AGENTS.md 与 `docs/batch_videos.md` 交付 ring_exchange 的完整仿真视频。Ring 的 runtime 没有 deadlock 判据；如果找不到合格的死锁解决案例，如实报告缺口，不得用 timeout 冒充。

# 七、交付

全部交付物放在 `diagnostics/controller_new_checkpoint_confirmation_v1/`：
- PREREGISTRATION.md 及收据；
- 兼容性与兄弟差异审计；
- R5 探测与冻结收据；
- Ê 与 G_E 的冻结选择；
- 新 state 的去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：逐 controller 的 6 个臂，以及 Ê 对真实 E 的散点图；
- 视频及其 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答 C1 与 C3 是否通过，以及 C2 是否满足 headroom 门。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。只有冻结后才产生的新数据可以称为“前瞻确认”。

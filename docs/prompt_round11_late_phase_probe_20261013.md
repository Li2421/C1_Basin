你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，并严格遵守。

# 一、背景与本轮问题

研究目标：让 critic 学会 controller 与 Q 的关系，在新 controller 上选好动作。

已确立的结论（全部为 DEVELOPMENT 证据）：
- **round10**（`diagnostics/controller_universal_eta_schedule_v1/`）：时间表 S500（前 500 步 η13，之后 η10）比各 controller 的最佳固定 η 平均高 26pp，在 held-out 组上高 30pp。
- **补充分析**（`diagnostics/controller_phase_decomposition_v1/`）：H_compress SUPPORTED。固定 η 的偏好是两个阶段效应之差的符号：
  - 前期效应 E = S500 − F10；
  - 后期效应 L = S500 − F13；
  - 净偏好 N = E − L。

  其中 E > 0 的有 11/14，L > 0 的有 14/14。
- **E 可以学到，L 还学不到：**
  - 200 步行为探测 R5（只覆盖前期）能预测 E：LOO Spearman 0.842，符号准确率 13/14，RMSE 8.6pp（源均值基线为 24.9pp）。
  - R3 和 R5 都无法预测 L（Spearman ≤ 0.21）。
  - 这与机制一致：R5 只看到前期，所以它也预测不了由 L 主导的 N。
- **扩展动作集下的零样本选择：** 只用源平均选择，平均遗憾 1.03pp，最大遗憾 5.86pp。剩余的损失来自两类 controller：
  - E ≈ 0 的 controller（v7、v8、v9）：它们在 η10 下已接近满分，前期的 η13 没有用处；
  - η2 型的 v3。

**本轮只回答一个问题：** 用一个专门覆盖后期阶段的无标签行为探测，能否学到 L？进而，基于 Ê 和 L̂ 的零样本动作选择，能否在新 controller 上消除剩余损失？

如果能，controller–Q 关系就在分解形式下被完整学到了：前期分量由前期探测得到，后期分量由后期探测得到。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 A：后期探测 R6（无标签，不读成功标签）

**对象：** 26 个 controller，即 14 个真实 Flow 和 round5 的 12 个融合 controller。

**探测 state：** 沿用 R5 的 16 个 probe state（`diagnostics/controller_flow_blend_v1/stage_2/probes/protocol.json`）。

**流程（每个 controller × probe state）：**
1. 以 η13 运行 400 步，到达后期状态。
2. 从这个状态分支，各运行 200 步：
   - 分支一：继续用 η13；
   - 分支二：换成 η10。
3. 期间关闭“成功即终止”与“碰撞即终止”，与 R5 的 probe.py 做法一致。
4. 只记录无终局标签的统计量：
   - 与 R5 相同的 5 个统计量；
   - 另加“200 步内到达目标的机器人数的变化”和“最终停滞的机器人数”。

   不读取、也不写入任何 success 标签。
5. 第 400 步之前出现数值失败的探测记为 unknown，不得以 0 填充。

**特征：** 对每个 controller，在 probe state 上取两个分支的统计量之差（η10 分支 − η13 分支）的均值，得到 R6-Δ；另外取原始统计量的 PCA-2，得到 R6-PCA2。z-score 与 PCA 只在训练 controller 上拟合。

**规模：** 26 × 16 × (400 + 2 × 200) = 332,800 个模拟步。

**DB 登记：** 按 R5 的先例，以 UNLABELED_SHORT_PROBE 的方式只登记 source_file 与独立 UID，不制造 success 记录。

## 阶段 B：L 的可学习性（不新增 rollout）

- **标签：** 用 round10 的 E、L、N 作为 14 个真实 Flow 的标签，从 `controller_phase_decomposition_v1/results.json` 读取。
- **方法：** 14-Flow LOO，固定 ridge（α = 1）。描述分别用 R5-PCA2、R6-Δ、R6-PCA2、R5-PCA2 ⊕ R6-Δ。
- **报告：** 对 E、L、N 分别报告 Spearman、符号准确率和 RMSE，并与源均值基线比较。
- **预注册判定：**
  - L 可学：R6 系列中至少一种描述对 L 的 LOO Spearman ≥ 0.6，且 RMSE 不高于源均值基线的 0.7 倍。
  - 不满足即判定 L 不可学。

## 阶段 C：零样本动作选择规则（预注册，不调参）

**动作集：** {F10, F13, S300, S500}。

**规则 G（不使用任何目标标签）：**
- 若 Ê < 5pp，选 F10。这类 controller 没有前期困难。
- 否则，若 L̂ < 5pp，选 F13。这类 controller 没有后期代价。
- 否则，选 S500。

Ê 用 R5-PCA2 的 ridge 预测；L̂ 用阶段 B 中预注册的主描述 R6-Δ 的 ridge 预测。

**对照：**
- 源平均选择（不使用 controller 信息）；
- 只用 Ê 的规则 G_E：若 Ê < 5pp 选 F10，否则选 S500；
- 真实 Flow 上的 1-NN。

**评估一：14 个真实 Flow 的 LOO（DEVELOPMENT）。** 遗憾以目标全部 6 个臂中的最佳臂为参照，与补充分析一致。

**评估二：12 个融合 controller（新的 controller，主检验）。**
- 规则在 14 个真实 Flow 上冻结后，直接用于融合 controller。
- 融合 controller 已有 round5 的 F10 与 F13 标签：32 个 test state × seed 0–3，直接复用。
- 只需补跑 S300 与 S500：12 × 32 × 2 × 4 = **3,072 条完整 rollout**。
  - 按 shared_rollout_db 流程执行；切换实现复用 round10 冻结的 `core.py`；先过 32 例一致性门。
- 遗憾以 {F10, F13, S300, S500} 中的最佳臂为参照。

**预注册判定（融合族）：**
- G 的平均遗憾低于源平均选择，且最大遗憾 ≤ 5pp，则判定 controller 信息带来了可迁移的选择收益。
- 同时报告 G_E 的结果，用来区分收益来自 Ê 还是 L̂。

# 三、结论映射

| 观察结果 | 结论 |
|---|---|
| L 可学，且 G 在融合族上过门 | controller–Q 关系在分解形式下被学到并能迁移：前期分量来自前期探测，后期分量来自后期探测。下一步在新状态与新的真实 controller 上做前瞻确认 |
| L 不可学，但 G_E 已过门 | 可迁移的部分是前期分量。后期分量要么不需要（L > 0 普遍成立，时间表即可覆盖），要么还需要另一种表示 |
| L 不可学，且 G、G_E 都未过门 | controller 信息只能解释 E；剩余几 pp 的损失暂时不能靠零样本消除，记录为已知限制 |

# 四、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/controller_late_phase_probe_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。
- **探测不读成功标签：** 阶段 A 不读取任何成功标签。阶段 B、C 只在冻结 R6 特征之后才读取标签。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **证据等级：** 14 个真实 Flow 的结果为 DEVELOPMENT。融合族的 S300 / S500 标签是新的，但融合族本身来自已打开过的锚点，只能称为“新 controller 上的检验”，不能称为独立确认。
- **资源：** 本 workstation 上用户所有任务合计不超过 18 个 CPU shard 和 3 个 GPU shard。
- **视频：** 按 AGENTS.md 自行判断阶段 C 的 3,072 条 rollout 是否属于需要视频的大型评价 batch。若适用，只对 ring_exchange 运行统一入口，并如实报告缺口。

# 五、交付

全部交付物放在 `diagnostics/controller_late_phase_probe_v1/`：
- PREREGISTRATION.md 及收据；
- 探测代码与 R6 特征；
- 探测审计（unknown 与分布）；
- 阶段 B 的 E / L / N 预测表；
- 阶段 C 中每个 controller 的动作与遗憾；
- 融合族的 DB preflight / postflight 记录；
- results.json；
- 图表；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答两件事：L 是否可学；controller 信息是否在新 controller 上带来了可迁移的选择收益。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE，事实与解释分开写。

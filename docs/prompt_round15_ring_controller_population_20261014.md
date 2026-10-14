你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md` 和 `shared_rollout_db/README.md`，并严格遵守。

**时间均为 workstation 本地时间（UTC+8）。**

# 一、为什么做这个实验（不是为了跑而跑）

**研究目标：** 让 critic 学会 controller 与 Q 的关系，从而在新 controller 上零样本选好动作。

**Ring 上已确立的结论**（DEVELOPMENT 证据，加上 8 个同 run 兄弟上的前瞻确认，见 `diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/`）：
- **根因 H_compress：** 整局固定 η 时，controller 的 η 偏好 N = F13 − F10 是两个阶段效应之差的符号：
  - 前期效应 E = S500 − F10；
  - 后期效应 L = S500 − F13。
- **修复：** 换成时间表动作，再用 200 步无标签探测 R5 预测 E（规则 G_E）。

**至今没有解决、而且都受“只有 14 个训练 controller”限制的三个问题：**
1. **后期效应 L 学不准**（round11，`diagnostics/controller_late_phase_probe_v1/`）：后期探测 R6 对 L 的排序相关为 0.79，但 RMSE 只降到基线的 0.77 倍，门槛是 0.7。
2. **切换时刻因 controller 而异，但规则还不会选：** round12 的 C1b 未通过，v10 型 controller 需要更早切换。η2 型 controller（v3、plus_w_v3）也没有覆盖。
3. **覆盖假设从未被检验：** 更多 controller 能否让关系学全？配方相同、seed 不同的 controller，regime 是否一致？（round7 中“独立 η13 型 controller 覆盖”仍为 INCONCLUSIVE。）

**现在可以检验了，原因有二：**
- round14A（`diagnostics/ring_independent_flow_confirmation_v2/`）恢复了历史训练代码 `git_43b5264`，在 GPU 上通过了预注册的备用复现门，并修复了 v4_wide 配方的架构元数据。
- 训练一个 Ring Flow 只需几秒到约 3 分钟。

**本轮任务：** 训练 56 个独立 seed 的 Ring Flow，在共同的新 state 上，采集一个包含 8 个动作的 Q 数据集；然后预注册地分析 controller–Q 关系的学习曲线。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 A：controller 总体（GPU）

- **配方：** 14 个 Ring 配方（`diagnostics/ring_exchange_stage1/*_macflow/config.json`）。
- **seed：** 每个配方训练 4 个新 seed，分别为原 seed + 30001、+30002、+30003、+30004，共 56 个 controller。这些 seed 不得与 round14A 用过的 +10000、+20001 至 +20003 重叠。
- **代码与设备：** 训练代码直接复用 round14A 目录中的历史包 `git_43b5264`（先核对字节一致），设备为 GPU，与 round14A 通过复现门时相同。
- **配方修复：** 照搬 round14A 的 PREREGISTRATION_RECIPE_REPAIR。
  - v4_wide 使用原 checkpoint 的架构（512×4）；
  - 其余配方逐项核对架构与归一化器，必须与原 checkpoint 一致。
- **新颖性：** 每个新 checkpoint 的参数 hash 必须与所有已有 checkpoint 都不同（包括 round12 与 round14A 的）。
- **兼容性门：** 用同一个 adapter 加载，前向计算可复现，使用 FP32（含 round12 的导入修复）。

## 阶段 B：无标签探测（CPU；在任何成功标签产生之前完成并冻结）

- **R5：** 16 个 probe state × {η10, η13} × 200 步。实现与 round12 / round14A 冻结的探测代码完全相同。
- **R6：** 先用 η13 运行 400 步，再分出 η13、η10 两个分支，各运行 200 步。实现与 round11 完全相同。
- **冻结：** 两种特征都冻结并留收据。

## 阶段 C：Q 数据集（CPU，主要的 rollout 量）

- **state：** 32 个全新 state，按 round10 的流程抽取，并与 DB 中所有 state 去重。全部 56 个 controller 共用这 32 个 state。
- **动作（8 个）：**
  - 固定：F10、F13、F2；
  - 时间表（前期 η13，第 t 步换 η10）：S200、S300、S400、S500、S600。
- **seed：** 0–3（Q4）。
- **规模：** 56 × 32 × 8 × 4 = **57,344 条完整 rollout**，另加 32 例一致性门。
- **实现与流程：**
  - 切换实现复用 round10 / round12 的 `core.py`；
  - 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。
- **分波执行（保证截止时数据仍可用）：**
  - 第 k 波 = 14 个配方各自的第 k 个新 seed，每波 14,336 条。按 1 → 4 的顺序执行。
  - 一波全部完成后才开始下一波。
  - 07:30 之后不再开始新的一波。
  - 主分析只使用完整的波。未完成的波保留，但不进入主分析。

## 阶段 D：预注册分析（只用冻结的特征与完整的波）

**标签（每个 controller）：**
- 8 个动作各自的成功率，按 unknown 保守下界计，同时报告上界；
- E、L、N（定义同上）；
- t* = S200–S600 中成功率最高的切换步（并列时取较晚者）；
- a* = 8 个动作中的最佳动作。
- 对每种标签，计算 split-half 信度（seed 0–1 对 seed 2–3），作为可学习性的噪声上限。

**1. 方差分解（描述性）：**
- N、E、L、t* 的方差中，配方之间与同配方 seed 之间各占多少（ICC，附 CI）。
- 有多少个配方，其 4 个 seed 对 N 的符号判断不一致。
- 预注册的预期：至少 3/14 个配方不一致，这与 H_compress 的“符号脆弱”一致。只作描述，不作门。

**2. 学习曲线：** 特征分别为 R5-PCA2、R6-Δ、R5-PCA2 ⊕ R6-Δ；模型为 ridge（α = 1）；z-score 与 PCA 只在训练 controller 上拟合。

- **(a) 新配方泛化：** 按配方分组。7 折，每折留出 2 个配方的全部 8 个 controller。训练集为其余 12 个配方各取 k 个 seed，k = 1、2、3、4，对应 n = 12、24、36、48。
- **(b) 新 seed 泛化：** 留出第 s 个 seed 的 14 个 controller（s = 1..4 轮流），用其余 seed 中的 k 个（k = 1、2、3）训练。

**3. 指标：**
- 对 E、L、N：Spearman、符号准确率、RMSE，以及 RMSE 与源均值基线 RMSE 之比。
- 对 t*：误差在 ±100 步以内的比例。
- **选择遗憾**（相对 8 个动作中的 oracle）：
  - 源平均动作；
  - G_E：Ê < 5pp 选 F10，否则选 S500；
  - G_EL：Ê < 5pp 选 F10；否则选 S{t̂*}，t̂* 由同一特征的 ridge 回归预测，并取整到最近的网格点；
  - 8 个动作的多输出 ridge 取 argmax。

**4. 预注册的判定：**
- **H_cov_L：** 在 (a) 中 k = 4 时，用 R6-Δ 或 R5 ⊕ R6，L 的 Spearman ≥ 0.6，且 RMSE 比 ≤ 0.7 → 判定“增加 controller 后 L 可学”。否则判 REJECTED。
- **H_cov_E：** 在 (a) 中，E 的 RMSE 随 k 下降。判定方法：k = 4 相对 k = 1 的 RMSE 差，按折 bootstrap 的 95% CI 上界 < 0。
- **H_switch：** 在 (a) 中 k = 4 时，G_EL 的平均遗憾低于 G_E，配对 CI 上界 < 0。
- **报告要求：** 每个结果都要与 split-half 信度给出的噪声上限并列报告。

**5. 外部验证（描述性）：** 用 (a) 中 k = 4 的模型（全部 56 个训练），预测以下 controller 的 E，与它们已有的标签比较：
- round10 的 14 个原 best controller；
- round12 的 8 个兄弟 controller；
- round14A 的 14 个 controller（若 round14A 届时已完成）。

# 三、结论映射

| 观察结果 | 结论 |
|---|---|
| H_cov_L 与 H_switch 都 SUPPORTED | 增加 controller 后，controller–Q 关系的两个分量都能从无标签探测中学到，选择规则进一步减少遗憾。下一步：用这个数据集训练完整的 critic Q(s, probe, a)，并在新 state、新 controller 上做前瞻确认 |
| H_cov_E SUPPORTED，H_cov_L REJECTED | 覆盖能改善前期分量，但后期分量受表示限制（或受标签噪声限制，对照噪声上限判断） |
| 同配方不同 seed 的 regime 经常不一致 | controller 身份主要在 seed 层面，而不在配方层面。这解释了为什么配方级的静态描述（R3）无效，也说明行为探测是必要的 |

# 四、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/ring_controller_population_v1/` 中写文件（包括新 checkpoint）。不修改、不删除历史资产（包括 round14A 的目录），不做 git commit。
- **冻结时序：** 探测特征与全部分析定义都必须在任何新成功标签产生之前冻结并留收据。阈值、动作菜单与分析定义一律不得根据结果调整。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。唯一允许的改变，是在第 t 步替换 η。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **证据等级：** 新 controller、新 state 上的数据是前瞻采集的。所有学习曲线结论都标为 DEVELOPMENT，因为分析是在同一数据集内做交叉验证。
- **视频：** 本轮是纯 Q 函数的数据生成与可学习性评估。按 AGENTS.md 的例外，不运行视频入口。这一判断写入预注册，并且必须在出现任何结果之前写定。

# 五、资源与时间（硬约束）

**总上限：** 用户本 workstation 上所有任务合计（包括正在进行的 round14A）：

| 时段 | CPU shard | GPU shard |
|---|---|---|
| 08:30 之前 | 24 | 6 |
| 08:30 起 | 18 | 2 |

08:30 时会释放 6 个 CPU 和 4 个 GPU。

**执行要求：**
- 提交前和运行中都要查看实时占用（`squeue`，以及本机常驻进程），任何时刻都不得超过上限。不要抢占 round14A 已在运行的作业。
- **按“可中断的 case 队列”设计 worker：** 每个 case 完成后立即写 raw 与 journal。
- 超出 18 CPU / 2 GPU 的那部分 worker，必须在 08:25 前停止领取新 case，并在 08:30 前退出。未完成的 case 回到队列，由剩余的 worker 继续。被中断的 case 视为“尚未执行”，不是 unknown。
- **时间节点：**
  - 07:30 之后不再开始新的一波；
  - 所有 rollout 与探测必须在 10:00 前结束；
  - 分析、REPORT 与 HANDOFF 在 10:30 前完成；
  - 10:30 之后不得再占用计算资源。
- 若预计第 4 波无法在 10:00 前完成，就不要开始第 4 波，用前 3 波（42 个 controller）做主分析。这个规则已在此预先写定。

# 六、交付

全部交付物放在 `diagnostics/ring_controller_population_v1/`：
- PREREGISTRATION.md 及收据；
- 训练与新颖性审计；
- 56 个 checkpoint 及其训练日志；
- R5 / R6 特征与冻结收据；
- 新 state 的去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- 资源使用记录（每个时段的实际 CPU / GPU 占用）；
- 数据集文件：每个 controller × state × 动作 × seed 的结果，含 unknown 标记；
- results.json；
- 图表：学习曲线、方差分解、E–L 散点图；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答三件事：
1. 增加 controller 后，L 和切换时刻能否学到（H_cov_L、H_switch）；
2. E 的预测是否随 controller 数量改善（H_cov_E）；
3. controller 的 regime 主要由配方决定，还是由 seed 决定。

每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE，事实与解释分开写。

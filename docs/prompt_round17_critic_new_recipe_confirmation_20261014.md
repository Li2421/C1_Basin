你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，并严格遵守。

# 一、要回答的问题

round16（`diagnostics/ring_controller_critic_v1/`）训练了一个以 controller 探测特征为输入的通用 critic Q(s, z_C, a)，结果如下：
- **预注册的主模型 A1**（在 A0 上加有界残差）没有通过 Q1、Q2。原因是有界残差太保守，无法表达较大的 controller 依赖。
- **作为描述性对照的无约束 critic A3**（64/64/16 MLP，z_C = R5-PCA2）在所有测试集上都好得多：

| 测试集 | A0（不看 controller） | A3，controller 级选择 | G_E 规则 |
|---|---|---|---|
| round15 按配方交叉验证 | 2.96 | 1.37 | 1.27 |
| T1：14 个独立 seed | 1.79 | 0.56 | 0.89 |
| T2：14 个原 controller | 1.20 | 0.75 | 0.64 |
| T3：8 个兄弟 controller | 3.91 | 0.54 | 1.42 |

（数值为平均遗憾，单位 pp。）在 T1–T3 上，A3 的灾难数都是 0，最大遗憾为 4.7、5.9、3.1pp。

**但 A3 不是预注册的主模型，所以这个结论目前只是描述性的。** 另外，round15 显示，E、L 的配方 ICC 约为 0.9：已测过的新 controller（兄弟、新 seed）与训练配方高度相似，真正困难的是**新配方**。

**本轮问题：** 冻结的 A3 critic，在**新训练配方**产生的 controller 和全新 state 上，能否零样本选好动作？也就是检验它是否真正学到了可迁移的 controller–Q 关系。

# 二、阶段 A：构造新配方的 controller（GPU）

**训练代码与设备：** 一律使用 round14A / round15 已验证的历史训练包 `git_43b5264`（先核对字节一致）、GPU，以及与 round15 相同的兼容性门和 FP32 修复。

**N1（新数据集）：**
- `diagnostics/ring_exchange_stage1/base_u_v7_local_plus_drift_dataset/` 从未用来训练过 Flow。先核对它的 manifest 与文件 hash 完整。
- 用 base_u_v7_local_macflow 的 config 训练，只替换数据集：100,000 步，256×3。
- seed 为 3124 + 40001 至 40004，共 4 个 controller。
- 若数据集不完整，N1 记为不可用并报告，不得替换为其他数据。

**N2（新架构）：**
- 13 个原配方（除 plus_w_v4_wide 外）的 config 原样使用，只把 actor_hidden_dims 改为 (512, 512, 512, 512)。参照 round14A 对 v4_wide 架构的修复。
- seed 为原 seed + 40001，每个配方 1 个，共 13 个 controller。

**检查：**
- 新颖性：每个 checkpoint 的参数 hash 必须与所有已有 checkpoint 都不同。
- 兼容性：用同一个 adapter 加载，前向计算可复现，使用 FP32。

# 三、阶段 B：冻结（必须在任何新成功标签产生之前完成并留收据）

1. **R5 探测：** 对 17 个新 controller，用 round15 冻结的探测代码，在 16 个 probe state × {η10, η13} × 200 步上运行，无标签。然后用 round16 冻结的预处理和 PCA 投影。
2. **冻结以下选择，逐 controller 写出所选动作并留收据：**
   - **A3（主）：** round16 的 A3 checkpoint，3 个 fit，原样使用，不重训。
     - 主规则：controller 级选择，即先在评估 state 上平均预测值，再取 argmax 作为固定动作。
     - 次规则：逐 state 取 argmax。
   - **A0：** round16 的 checkpoint，同样两种规则。
   - **G_E：** round16 冻结的版本。
   - **源平均动作。**
   - **A3-all（次要）：** 用 round16 的 A3 超参数，在全部已有标签的 controller 上重训：round15 的 56 个，加上 T1、T2、T3 的 36 个，同一套 8 动作或子集标签。必须在新标签产生之前完成训练与冻结。

# 四、阶段 C：标签采集

- **state：** 48 个全新 state，按 round10 的流程抽取，并与 DB 中所有 state 去重。
- **动作：** 与 round15 相同的 8 个：F10、F13、F2、S200、S300、S400、S500、S600。
- **seed：** 0–3。
- **规模：** 17 × 48 × 8 × 4 = **26,112 条完整 rollout**，另加 32 例一致性门。
- **流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。切换实现复用 round15 的冻结代码。

# 五、预注册终点与判定

oracle 为每个 controller 在 8 个动作中的最佳者。CI 一律按 state 成簇做 bootstrap，使用配对比较。

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| K1 | critic 是否学到可迁移的 controller–Q 关系 | A3（controller 级）的平均遗憾比 A0（controller 级）低，配对 95% CI 上界 < 0 |
| K2 | 是否不劣于手工规则 | A3（controller 级）的平均遗憾 ≤ G_E 的平均遗憾 + 0.5pp，且最大遗憾 ≤ 5pp |
| K3 | 是否安全 | 没有任何新 controller 的 A3 选择比 F10 低 10pp 以上 |

**分层报告：**
- N1（新数据集）与 N2（新架构）分开报告；
- N1 只有 4 个 controller，只作描述。

**另外报告：**
- A3 逐 state 选择与 controller 级选择之比较；
- A3-all 与 A3 的比较；
- 由 critic 推出的 E、L 与真实 E、L 的相关。

# 六、结论映射

| 观察结果 | 结论 |
|---|---|
| K1、K2、K3 都通过 | 前瞻确认：通用 critic 学到了 controller–Q 关系，并能零样本迁移到新训练配方的 controller。研究目标的原始表述在 Ring 上实现 |
| K1 通过，K2 不通过 | critic 学到了 controller 依赖，但不如结构化规则。分解形式（探测 → E → 规则）仍是更好的表达 |
| K1 不通过 | 在新配方上，controller 输入没有带来选择收益。round16 的 A3 优势只适用于与训练配方相似的 controller |

# 七、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/ring_critic_new_recipe_confirmation_v1/` 中写文件（包括新 checkpoint）。不修改、不删除历史资产，不做 git commit。
- **冻结时序：** 第三节的全部内容都必须在任何新成功标签产生之前冻结并留收据。不得根据新标签修改模型、规则或阈值。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。唯一允许的改变，是在第 t 步替换 η。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **证据等级：** 新 controller 与新 state 上的结果是前瞻的。N2 与训练配方共享数据集，只能称为“新架构配方”，不得称为完全独立的新数据配方。
- **视频：** 本轮是纯 Q 函数的数据生成与 critic 评估，按 AGENTS.md 的例外不运行视频入口。这一判断在预注册中写定。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。提交前查看实时占用。

# 八、交付

全部交付物放在 `diagnostics/ring_critic_new_recipe_confirmation_v1/`：
- PREREGISTRATION.md 及收据；
- 训练、新颖性与兼容性审计；
- 17 个 checkpoint 及其训练日志；
- R5 特征与冻结收据；
- 冻结的逐 controller 选择；
- 新 state 的去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：各方法遗憾、逐 controller 选择、推出的 E / L 与真实值；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：冻结的通用 critic 能否在新训练配方的 controller 上零样本选好动作，即 K1、K2、K3 是否通过。

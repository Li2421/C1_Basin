你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、要回答的问题

round12（`diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/`）在 8 个新 controller 上做了前瞻确认：
- C1、C2、C3 均通过；
- 时间表动作 + “由 R5 探测预测前期效应 Ê”的规则 G_E 能实现零样本 controller 泛化；
- 学到的 R5 → E 关系能迁移。

但那 8 个是已用 controller 的**同 run 兄弟**（同一次训练的 final.pkl）。

**本轮问题：** 对**独立训练**的新 Ring Flow（同配方、新随机种子），同一个冻结流程是否仍然成立？

# 二、必读材料（亲自核对）

- round12 目录：REPORT.md、PREREGISTRATION*.md、所有冻结收据、Ê 模型与 G_E 的定义、FP32 修复记录。
- `diagnostics/ring_exchange_stage1/*_macflow/`：每个 run 的 config.json（含 seed、dataset_manifest_sha256、训练步数）与 training_summary.json。训练耗时 6–193 秒。
- 训练代码：`new_benchmark_common/training.py`，以及对应的 `*_dataset/` 目录。

# 三、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## 阶段 A：训练 14 个独立的新 Flow

**配方与种子：**
- 对 14 个配方中的每一个，都使用原 config。
- 只把 seed 改为“原 seed + 10000”，在预注册中逐一列出。
- 其余全部不变：数据集 manifest（先核对文件与 hash）、步数、best checkpoint 的选取规则。

**训练复现门（先做）：**
- 用原 seed 重训 base_u_v2_macflow（12k 步，约 16 秒），与原 run 比较：
  - 参数 hash；
  - dev loss 曲线。
- 不逐位一致也可以，但 best_fixed_dev_loss 与 final_fixed_dev_loss 的相对误差必须都 ≤ 1%。
- 不满足就停下报告：训练代码或环境可能已经变化，不能声称这些 controller 和原配方是“同配方”。

**新 checkpoint 的检查：**
- 每个新 checkpoint 的参数 hash 必须与所有已有 checkpoint 都不同。
- 描述性报告它与同配方原 controller 之间的参数距离。

**写入位置：** checkpoint 只写入本轮目录。

## 阶段 B：在新 Flow 上运行冻结流程

所有组件都**原样复用 round12 的冻结定义**：
- **R5 探测：** 16 个 probe state × {η10, η13} × 200 步，无标签，FP32（含 round12 的修复）。在任何成功标签产生之前完成并冻结。
- **Ê 模型：** ridge（α = 1）。输入为 R5-PCA2，标签为 round10 中 14 个 best controller 的 E。
  - 主方案：排除同配方的原 controller，只用其余 13 个训练。
  - 次方案：用全部 14 个训练。
- **规则 G_E：** Ê < 5pp 时选 F10，否则选 S500。
- **动作菜单与 oracle：** 菜单为 {F10, F13, S300, S500}；F2、F4 只作 oracle 参照。

**state：**
- 64 个全新 state，按 round10 的流程抽取。
- 与 DB 中所有 state（包括 round12 的）去重。

**运行要求：**
- 规模：14 × 64 × 6 × 4 = **21,504 条完整 rollout**，另加 14 × 32 次短探测。
- 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。
- 先过 32 例一致性门：用 best.pkl 的已有记录逐位复现。

## 终点

C1–C4 与 round12 完全相同，包括 C2 的 headroom 门（至少 2 个新 controller 的真实 E < 5pp）。阈值一律不改：

| 编号 | 检验内容 | 判定标准 |
|---|---|---|
| C1 | 整套流程的效果 | G_E 相对 F10 的平均收益，95% 下界 > 0；平均遗憾 ≤ 2pp；没有任何 controller 比 F10 低 10pp 以上 |
| C1b | 最大遗憾（次要） | 最大遗憾 ≤ 5pp |
| C2 | controller 信息的价值 | G_E 相对“全部选 S500”的配对改善，CI 下界 > 0 |
| C3 | 学到的关系能否迁移 | Ê 与真实 E 的 Spearman ≥ 0.6，且 RMSE ≤ 源均值基线的 0.7 倍 |
| C4 | E、L 的符号统计 | 描述性 |

**额外的描述性报告（不参与判定）：**
- 对每个 controller，比较 S300 与 S500 的遗憾，检查 round12 中 v10_final 的“需要更早切换”问题是否普遍。
- 报告最优臂为 F2 或 F4 的 controller，用来判断 η2 / η4 型的出现频率。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| C1 与 C3 都通过 | 在独立训练的新 controller 上确认：时间表动作 + 学到的 R5 → E 关系，能实现零样本 controller 泛化。限 Ring，且为同配方、不同种子 |
| C1 通过，C3 不通过 | 时间表的泛化得到确认，但 R5 → E 只在同 run 兄弟上成立，不能推广到独立训练的 controller |
| C1 不通过 | round12 的确认只适用于同 run 兄弟。报告失败的 controller，以及它们的失败类型与时间 |

# 五、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/ring_independent_flow_confirmation_v1/` 中写文件（包括新 checkpoint）。不修改、不删除历史资产，不做 git commit。
- **冻结时序：** 阶段 B 的探测、模型与规则，都必须在任何新 Flow 的成功标签产生之前冻结并留收据。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **资源：** 本 workstation 上用户所有任务（包括可能并行的 round13 跨场景实验）合计不超过 18 个 CPU shard 和 3 个 GPU shard。提交前查看实时占用。
- **视频：** 按 round12 的做法交付 ring_exchange 视频，缺失的 deadlock 角色如实报告缺口。

# 六、交付

全部交付物放在 `diagnostics/ring_independent_flow_confirmation_v1/`：
- PREREGISTRATION.md 及收据；
- 训练复现门报告；
- 14 个新 checkpoint 及其训练日志与 hash；
- R5 探测与各冻结收据；
- 新 state 的去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表；
- 视频及 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答 C1、C3 是否通过，以及 C2 是否满足 headroom 门。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE。

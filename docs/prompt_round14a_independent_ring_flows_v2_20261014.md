你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、背景：round13B 为什么停下，以及本轮怎么修正

round13B（`diagnostics/ring_independent_flow_confirmation_v1/`）用原 seed 重训 base_u_v2_macflow 时，dev loss 与原 run 相差 6.7%（best）和 4.0%（final），没有通过复现门，于是按规定停止。这个停止是正确的。

设计者只读核查后，找到一个具体原因（你需要独立复核）：

- **原 Ring Flow 的训练时间是 2026-09-30**（见 `diagnostics/ring_exchange_stage1/*_macflow/` 的文件时间）。
- **`new_benchmark_common/training.py` 在此之后被大幅修改过。** git 中最早的版本在 commit `43b5264`（2026-10-07），之后又在 10-07 到 10-09 之间多次修改：
  - 新增了大量采样选项、归一化选项、架构选项，以及 load_checkpoint；
  - 文件从 170 行增加到 449 行。
- **commit `43b5264` 中的 training.py 与 PoC 目录里的副本完全相同**（sha256 前缀均为 `4106e0003393`）。PoC 副本位于 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/new_benchmark_common/training.py`，复制于 10-04，早于上述修改。
- **round13B 用的是当前 HEAD 的训练代码。** 即使默认参数相同，采样流程或随机数消耗的差异也足以造成这种偏差。

本轮问题不变：在独立训练的新 Ring Flow（同配方、新 seed）上，round12 的冻结流程是否仍然成立？

# 二、阶段 0：恢复历史训练代码并重做复现门

**1. 构建隔离的历史训练环境。**
- 把 commit `43b5264` 中 `new_benchmark_common/` 的完整包导出到本轮目录（例如用 `git archive` 或 `git show`）。不要修改仓库中的任何文件。
- 逐文件与 PoC 目录 `source/new_benchmark_common/` 比对 sha256。重点是 training.py、macflow.py，以及 training.py 导入的其他模块。
- 报告每个文件是否一致。若有不一致，以 PoC 副本为准；PoC 副本更接近原训练日期。在预注册中写明你选用了哪个版本。

**2. 用历史代码重训 base_u_v2_macflow（原 seed 3104，原 config）。**
- **主判定：** best_fixed_dev_loss 与 final_fixed_dev_loss 的相对误差都 ≤ 1%（与 round13B 相同的阈值）。
  - 同时报告参数 hash 是否逐位一致，以及 best step 是否相同。
  - 先在 CPU 上跑；不通过再在 GPU 上跑（经 Slurm）。两次都要记录设备与 JAX 版本。
- **备用判定（主判定在两种设备上都不通过时才使用；本文件已预先写定）：**
  - 对 base_u_v2 与 base_u_v9_local（后者已收敛，best step 97,750 / 100,000）两个配方，各用 3 个新 seed（原 seed + 20001、+20002、+20003）以历史代码训练。
  - 若两个配方都满足“|原 seed 重训 − 原 run| ≤ 3 个新 seed 的 dev loss 极差（max − min）”，就判定差异在 seed 间随机波动范围内，训练流程一致。
- **两种判定都不通过：** 停下，报告 C1–C4 为 INCONCLUSIVE，并写明还可能的原因（数据集文件、依赖版本等）。

# 三、阶段 1–2（只在阶段 0 通过后执行）

**沿用 round13B 的方法：** 除下面列出的改动外，其余内容全部照 `docs/prompt_round13b_independent_ring_flows_20261013.md` 执行：
- 训练 14 个新 Flow（原 seed + 10000）；
- 原样复用 round12 冻结的 R5 探测、Ê 模型（主方案排除同配方原 controller）与 G_E 规则；
- 动作菜单 {F10, F13, S300, S500}，F2、F4 只作 oracle 参照；
- 64 个全新 state × 6 个臂 × seed 0–3，共 21,504 条完整 rollout；
- 终点 C1–C4 与判定门槛完全不变，包括 C2 的 headroom 门。

**本轮的改动：**
- 训练一律使用阶段 0 通过时所用的历史代码版本与设备。
- 所有新 checkpoint、日志与结果写入新目录 `diagnostics/ring_independent_flow_confirmation_v2/`。不修改 v1 目录。
- 先写补充预注册（带 UTC 与 sha256 收据），说明 round13B 的失败原因与本轮的修正。必须在训练任何新 seed 之前完成。

# 四、硬性规则

- **不修改仓库代码与历史资产：** 历史训练代码只导出到本轮目录中使用。不做 git commit。
- **冻结时序：** 探测、模型与规则必须在任何新 Flow 的成功标签产生之前冻结，并留收据。
- **不得改动核心组件：** 不修改控制器、物理、安全投影或成功判定。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **资源：** 本 workstation 上用户所有任务（包括并行的 round14B）合计不超过用户当前授权的总上限。最近一次授权为 24 个 CPU shard、3 个 GPU shard；如有更新，以用户最新指示为准。提交前查看实时占用。
- **视频：** 同 round12：交付 ring_exchange 视频；缺失的 deadlock 角色如实报告。

# 五、交付

REPORT.md 第一段必须直接回答两件事：
1. 训练代码版本漂移是否就是 round13B 复现失败的原因；
2. C1、C3 是否通过，C2 是否满足 headroom 门。

另交付 HANDOFF.md（不依赖本对话也能读懂）。

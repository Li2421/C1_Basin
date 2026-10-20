你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`、`docs/controller_generalization_ring_summary_20261014.md`，以及本任务的原始设计 `docs/prompt_round19_four_controller_axis_20261014.md`。严格遵守 AGENTS.md。

# 一、背景：round19 为什么停下

round19（`diagnostics/four_controller_axis_v1/`）在阶段 0 按规定硬停止。

**停止原因：** G1 要求逐位复现历史记录的“首动作字节”，但 v12 与 v10_wide 的历史 intervention 记录（共 2,384 条）都没有保存首动作或轨迹，比较无法进行。这是设计者在设计 G1 时没有先核对历史字段造成的，不是执行问题。

**当时的状态：** 停止时没有产生任何新探测、state、成功标签或模型；G0 与 G2 都没有运行（见 v1 的 STOP.receipt.json）。

**可用的指纹字段：** 设计者只读核对过，这两批历史记录在 DB 与原始 journal 中都保存了以下字段，可用作完整轨迹的指纹（你需要独立复核）：
- outcome（success、timeout、numerical_failure）；
- episode_length；
- min_agent_distance（float64，完整精度，例如 0.4359493386929989）；
- min_wall_distance（float64，完整精度）。

一局 200–1100 步的 rollout，只有轨迹逐步一致，这四项才可能全部逐位相等，所以它们足以替代首动作。

# 二、本轮任务

按 `docs/prompt_round19_four_controller_axis_20261014.md` 的设计**完整执行**，只做下面三处修改：

1. **新目录：** 所有文件写入新目录 `diagnostics/four_controller_axis_v2/`。v1 目录保持原样，只作为背景引用。
2. **修订 G1。** 在任何等价门运行之前，写好补充预注册（UTC 与 sha256 收据），写明修订原因，以及“v1 停止时没有产生任何新结果”这一事实。G1 的新定义：
   - **G1a 指纹门（必须通过）：**
     - 对 v12 best 与 v10_wide best，各从历史记录中选 16 条：按 outcome 分层，success 与 timeout 各半；不选 numerical_failure。选取规则在预注册中写定（例如按 rollout_uid 排序后取固定位置），不得根据复现结果挑选。
     - 用新的逐 checkpoint 适配层，在历史实验原有的 conditioning 下重跑这些记录。原 conditioning 为 `training_dev_intermediate_state_episode_eta_v1 + committed_base_t0_then_alt_future_v1`，包括同一 state、同一个由 v13 承诺的 t0 动作，以及同一 seed key 与 continuation seed。
     - 要求 outcome、episode_length、min_agent_distance、min_wall_distance 四项全部逐位相等（float64 比特一致）。32 条中有任何一条不一致，即判 G1a 不通过。不得放宽为容差比较。
     - JAX x64 等精度设置必须与历史实验的记录一致，先核对。
   - **G1b 差分门（条件执行）：** 若历史 intervention runtime 的代码在本机仍可运行，就取 G1a 中的 4 条，用历史 runtime 与新适配层分别运行，逐步比较首动作字节与完整轨迹，要求逐位相等。若历史 runtime 无法运行，记为 NOT_AVAILABLE 并说明原因；G1a 通过时，这一项不阻断后续阶段。
   - **G0 与 G2 不变：** 定义同 round19，必须重新在新适配层上运行。不得把 round14B 中旧适配器的结果当作本轮的 G0。
3. **硬停止规则不变：** G0、G1a、G2 任一不通过，就停下报告。若是因为历史字段缺失等原因导致门无法检验，同样停下，并写明还缺什么证据。

其余内容全部照 round19 的设计执行：controller 列表、阶段 1–4、门槛与阈值、结论映射、硬性规则、视频要求与交付格式。

# 三、资源与证据

- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。
- **证据：** G1a / G1b 的重跑只用于验证适配层，属于一致性门。重跑结果单独存放，标为 gate；不写入、也不替换任何历史 DB 记录。按 shared_rollout_db 的规则，gate 用的记录登记为独立的 gate 身份，或者只写入本目录（二者选一，在预注册中写明）。

REPORT.md 第一段必须先回答两件事：等价门（G0、G1a、G1b、G2）是否通过；若通过，再按 round19 的要求回答 Four 的三个问题。

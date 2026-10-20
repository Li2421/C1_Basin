你是接手 Basin_C1 的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不新增任何正式 rollout，只做小规模的复现排查。** 不写共享 DB，不需要视频。

# 一、要回答的问题

round19b（`diagnostics/four_controller_axis_v2/`）的等价门中：
- **G1a：** 新的逐 checkpoint 适配层逐位复现了 32/32 条历史 intervention 记录的指纹（结局、步数、两项 float64 最小距离）。
- **G1b：** 用历史 intervention runtime 与新适配层各跑 4 例，比较首动作和完整轨迹。只有 3/4 逐位一致，**1 例从第一步动作起就分叉**（见 G1B_REPORT.json）。

**矛盾之处：** 新适配层与历史 DB 记录一致，但今天运行的历史 runtime 却与它在这一例上不一致。这说明“今天运行的历史 runtime”可能已经不同于“当时产生这些记录的 runtime”。

**重点怀疑方向：** round12 发现过，历史控制模块在导入时会把 JAX 切到 FP64（见 `diagnostics/controller_new_checkpoint_confirmation_v1/` 的精度事故记录）。G1b 中历史 runtime 的导入顺序，可能触发了同样的问题。

# 二、排查步骤

1. **复现分叉：** 在本轮目录中重跑 G1b 那一例，记录两条路径在第一步时的 JAX x64 状态（`jax.config.read('jax_enable_x64')`）、输入观测的字节、归一化参数，以及 Flow 输出的字节。
2. **精度假设：**
   - 在历史 runtime 导入之后显式恢复 FP32（做法同 round12 的修复），再重跑这一例，看是否与新适配层逐位一致；
   - 反过来，强制 FP64 运行新适配层，看是否与历史 runtime 一致。
3. **若精度不是原因：** 逐项比较两条路径的差异：checkpoint 加载、归一化、conditioning（v13 承诺的 t0 动作）、RNG key 的派生、安全投影的参数。找出第一个出现差异的中间量。
4. **影响评估：** 检查使用该历史 runtime 的其他实验，是否也可能受同样的导入顺序影响。例如 `exp_orthoflow3_controller_intervention_*` 系列：它们记录的 x64 设置是否与指纹一致。说明 round19b 的结论（以 G1a 为准）是否受影响。

# 三、交付

全部交付物放在新目录 `diagnostics/g1b_divergence_audit_v1/`：
- 排查记录；
- 第一处出现差异的中间量；
- 结论：ROOT_CAUSE_FOUND / NARROWED / UNRESOLVED；
- 对历史实验的影响评估；
- REPORT.md；
- HANDOFF.md。

# 四、硬性规则

- 只运行排查所需的少量单例。结果只写本轮目录，不写共享 DB，不替换任何历史记录。
- 不修改、不删除历史资产与历史 runtime 代码，不做 git commit。
- 事实、解释和推测分开写。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

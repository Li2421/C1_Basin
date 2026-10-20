你是接手 Basin_C1 的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`，并严格遵守。

**本任务不新增任何 rollout，只排查训练的可复现性。** 只做小规模的 Flow 训练，不需要视频。

# 一、要回答的问题：原 Ring Flow 为什么无法复现？

**已知事实**（你需要独立复核）：
- 原 Ring Flow 训练于 2026-09-30，位于 `diagnostics/ring_exchange_stage1/*_macflow/`，目录中有 config.json、metrics.jsonl、training_summary.json、best.pkl、final.pkl。
- 以 base_u_v2_macflow 为例：原 seed 3104，12,000 步，best step 12000，best_fixed_dev_loss = 0.129786。
- 重训的结果：
  - round13B（当前 HEAD 代码，CPU）：dev loss 0.121099，相对误差 6.69%，best step 11750；
  - round14A（历史包 `git_43b5264`，与 PoC 副本逐字节相同，CPU 与 GPU）：得到几乎完全相同的偏差，6.69% / 6.70%。

  见 `diagnostics/ring_independent_flow_confirmation_v1/` 与 `_v2/` 中的 PRIMARY_TRAINING_GATE.json。
- 由此可知，训练代码版本和设备都不是原因。仍可能的原因：数据集文件的实际内容、依赖库版本（jax、jaxlib、flax、optax、numpy）、随机数状态，或归一化参数的计算。

# 二、排查步骤（按顺序；每一步先记录事实，再进入下一步）

1. **第 0 步的损失（最关键）：**
   - 原 run 的 metrics.jsonl 中，第 0 步的 fixed_dev_loss 为 2.05638325214386，fixed_train_loss 为 2.0488712787628174（请复核）。
   - 用历史包、原 seed，只做初始化和第 0 步评估，不训练，比较这两个数。
   - **相同：** 初始化、数据与归一化一致，偏差来自训练过程（批次顺序、优化器、数值）。接着比较第 1000 步，定位第一个出现差异的记录点。
   - **不同：** 问题出在训练开始之前（数据、归一化或初始化）。逐项比较归一化数组（obs / act 的 mean 与 scale）、数据集文件的 hash 与样本数、初始参数的 hash。
2. **数据集：** 核对 base_u_v2_dataset 中每个文件的 sha256 与 manifest 记录的是否一致，以及文件修改时间是否晚于 2026-09-30。
3. **依赖版本：** 记录当前环境的 jax、jaxlib、flax、optax、numpy 版本，并查找原 run 时的版本线索（run 目录、日志、git 历史、venv 记录）。若能确定原版本，在一个隔离的新 venv（放在本轮目录中）里装相同版本，复现第 0 步与第 1000 步。
4. **随机数：** 检查训练中所有随机源（初始化 key、批采样、数据打乱）是否都由 config 中的 seed 决定，以及是否有环境相关的默认值。
5. **再取一个已收敛的配方验证：** 对 base_u_v9_local_macflow（100k 步，已收敛）重复第 1 步的比较，确认结论是否一致。

# 三、交付与判定

全部交付物放在新目录 `diagnostics/ring_training_repro_audit_v1/`：
- 逐步排查记录；
- 比较表；
- 结论；
- REPORT.md；
- HANDOFF.md。

**结论必须是以下三者之一：**
- **ROOT_CAUSE_FOUND：** 指明原因，并能在修正后复现原 run（第 0 步与最终 dev loss 都在 0.1% 以内，或逐位一致）。
- **NARROWED：** 把原因缩小到具体环节，并说明还缺什么证据。
- **UNRESOLVED：** 列出已排除的原因。

**最后评估影响：** round14A 与 round15 的结论依赖“备用复现门”（差异落在 seed 间的随机波动范围内）。说明本轮的发现是否影响这些结论的有效性。

# 四、硬性规则

- 不新增 rollout，不写共享 DB。只训练排查所需的少量 Flow，checkpoint 只写入本轮目录。
- 不修改、不删除历史资产（包括原 run 目录、数据集与 PoC 目录），不做 git commit。新 venv 只建在本轮目录中，不改动现有环境。
- 事实、解释和推测分开写。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

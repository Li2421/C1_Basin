# 历史证据审计（实验未启动）

- `diagnostics/decentralized_pipeline_v1/main_protocol.json`：14 个冻结 Ring Flow 教师；逐一读取 checkpoint 文件并核对 SHA256，14/14 一致，见 TEACHER_CHECKPOINT_AUDIT.json。未反序列化、未推理。
- round28 `IMPLEMENTATION_FROZEN.receipt.json`：逐文件校验结果见 ROUND28_FROZEN_AUDIT.json；D-local 安全层可作为只读组件。其 F10/F13/S300/S500 三管线标签及专家结果应在后续按身份复用。此次未重新运行。
- `docs/controller_generalization_ring_summary_20261014.md` 第 14–17 节、round28 REPORT：已有时间表、critic、分散式安全层与 nominal/safety 证据；round28 的 nominal 仍为联合策略，不能当作本轮分散式学生标签。
- `diagnostics/other_controller_classes_v1/probes.py`、`freeze_choices.py`：R5、A3/A0、G_E 的参考流程存在；原脚本包括全局变量替换与其他目录写入，不能直接运行。后续需在新目录独立实现合规流程。
- `toy_giveway/flowbc.py` 是旧 GiveWayFlowBCAgent 的兼容导出；其原实现 `02_C1_Toy_GiveWay/flowbc/giveway_flowbc_agent.py` 明确仅 Stage-I，use_distillation=False。
- `double_bottleneck/flowbc_4a_agent.py` 为四机器人展平联合 Stage-I，use_distillation=False，不是分散式一步学生。
- `new_benchmark_common/macflow.py` 仅构造 actor_bc_flow，use_distillation=False；未提供显式噪声采样入口。
- 官方 `agents/macflow.py` 有 actor_onestep_flow 与 compute_flow_actions(observations,noises)，可参考网络与损失；该类并非 round28 冻结教师使用的 JointMACFlowAgent。
- 检索范围：上述相关源码/训练配置，以及 diagnostics 下路径名含 student/distill/onestep 的文件；未发现可直接复用的本轮兼容学生。此结论限该检索范围，不声称所有二进制 checkpoint 均已语义审查。

缺失证据：学生训练、dev 选择、能力门、R5、冻结选择与学生闭环标签均未产生。教师补充臂的 DB 精确缺口尚未检索，不推断其 genuinely_missing 数量。阻塞在接口核验阶段，未创建执行身份、未预注册、未做 preflight、未提交作业、未写 DB。没有将未执行阶段标为通过。

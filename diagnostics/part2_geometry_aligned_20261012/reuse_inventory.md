# 复用与成本审计（计算前）

|证据|路径|可复用内容|缺口|
|---|---|---|---|
|真实控制链|`/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py`、`poc/field_controller.py`|TT/TF/FT/FF、原始 Flow 后处理、CBF/投影、10 子步|无代码改动|
|49,152 条完整评估|`/home/zhihan/research/Basin_C1_hard_field_20261006/{confirm_q16_tasks.json,results/}`|固定16 seed、位置、动作、噪声、success/事件；可重新计算方向性几何量|既有 B15 标签不能视作前瞻 held-out|
|16-key 动作/Jacobian 缓存|`/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/{results/,state_eta_chain_features.csv}`|首步 `pre_all`,`exec_all`,`M_exec`,`basis` 及投影活跃签名，覆盖全部 Q16 候选|旧 NPZ 未保存修正/零修正的完整逐子步 `h_k`|
|上一轮机制报告|`diagnostics/part2_gain_loss_mechanism_20261012/`|B15 Gain/Loss、事件重放、轨迹行为、四例 η4 子步 trace、按初态 split 与 η-only 基线|尚无统一有符号几何方向与全 Q16 微分—有限动作对应|
|H1–H4/Part2|`diagnostics/placement_boundary_mechanism_audit_v1/`、`diagnostics/placement_strong_hypotheses_v1/`、`diagnostics/part2_closed_loop_sensitivity_validation_20261012/`|正规局部 M/G/K/S/ρ 与 Toy 非光滑、Ring 局部边界、H1 TT/FF donor 交换|H1 不是本轮 TT/TF 或 FT/FF 独立中介介入|

本轮先重算旧缓存。窄缺口仅是不同 placement 的逐子步路径差，依 `protocol.json` 固定 24 个初态/η/seed action-only cases，共 48 次冻结 Flow 积分，加原动作缓存核对；不调用 `env.step`，新增完整 continuation `0/320`，GPU `0`。若该探针不能识别独立中介，报告不足而不扩大 rollout。

用户随后要求不进行任何新的实验；该要求到达时上述 24 个 action-only cases 已完成，此后仅作缓存算术与文档整理。`flow_substeps.stderr.log` 中 JAX 的可选 CUDA 插件发现警告是非致命的；进程显式使用 `JAX_PLATFORMS=cpu`、正常退出，24 例新旧首动作与原始 rollout 缓存逐数一致。

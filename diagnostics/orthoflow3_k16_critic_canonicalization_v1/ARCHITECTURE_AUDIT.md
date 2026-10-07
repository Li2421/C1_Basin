# 冻结实现与阶段边界

本任务先完成 A（固定表示/生成器，修复 critic 数据覆盖），冻结并报告后才启动 B（物理 conditioning 规范化）。A 的训练与采集不读取已检查测试集。

## 可执行实现

- 模型、损失、域变换：`diagnostics/orthoflow3_generator_critic_v1/train_evaluate.py`。生成器为原生输入适配器 96 + context 32 + 共享 128/64，输出对角 squashed Gaussian。sigma = 0.025 + 0.275 sigmoid。critic 保持相同适配器与共享主干，eta 编码 32，输出经验成功概率 sigmoid(logit)。
- 部署候选继承 `orthoflow3_ring_k16_diagnostic_v1/run_k16.py`：确定性 mean 候选 + **16 个随机样本**，不是总共 16 个；没有 mean-only 部署，也不加入 eta=0 或固定 eta。随机数规则保持 `SHA256(generator-v1-proposals|41|scenario|state_uid)`。
- eta 域为 `[0.5,-0.5,0]` 至 `[1.25,0.5,0.75]`；网络标准化是减中心/除半宽，执行仍使用原始三维系数。
- A 保留原 normalization 与生成器。当前表示为原生观测展平 + 原始 Flow 动作 8 维；Double/Four 为 80 维，Ring 为 100 维。context 是环境数值字段并集 30 维 + 3 维 scenario 编码，按原统计变换。B 才修正表示及统计。
- rollout 使用已有 `TrainingRuntime` / `DoubleTrainingRuntime`；训练/验证 state ID、控制器 identity、seed semantics 均保持精确兼容。Ring 使用当前已修正 outer-boundary 的安全实现。

## Critic 配方选择依据

保留 canonical binomial BCE（经验 Q，证据权重上限 16）、AdamW 1e-3/1e-4、clip 5、至多 4000 步、每场景 96 个状态、种子 17/23/41。只有采样分布对齐部署：等场景、均匀状态、80% 冻结提案 /20% v2 完整 Q 标签。

不复活 superseded ranking-loss：`orthoflow3_ranking_aware_critic_v1/final_report.md` 为 `OBJECTIVE_MISMATCH_NOT_MAIN_BOTTLENECK`；`orthoflow3_nll_weighting_ablation_v1/final_report.md` 不支持换 NLL 权重为主要修复。也不重复 LCB 实验。

全部提案在查对应 rollout outcome 前冻结。Full Q16 用于 Q/regret，不以二次失败提前退出；数值失败保留 unknown，不填成失败。eta=0 仅是独立 rescue/break 参考。

## 预先说明的 B 限制

Four-Way 冻结 MACFlow 本身并非等变。被动旋转同一个 Flow 向量后编码相等，不等于主动旋转场景后重新调用 MACFlow 也得到相同 Flow。B 必须分别报告这两种检验，并保留真实闭环偏差；不能通过改变 MACFlow 或标签来制造等变结论。

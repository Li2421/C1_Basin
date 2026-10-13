# Frozen physical hypotheses (before validation and held-out scoring)

冻结时间：2026-10-12。只使用每场景 UID 01–08 的 discovery 组确定阈值；UID 09–16 和 17–24 尚未用于本轮阈值选择。特征名称引用旧 `/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/state_eta_chain_features.csv` 的 `phase=initial`，其值为同一初态/η 的 16 个固定 Flow key 上的首动作平均。该旧审计已看过全 Q16 标签，所以这里的 held-out 是**本轮规则选择的独立检验**，不是历史上从未看过的全新实验。

## Condition B：Toy 的目标向动作受抑

令每机器人目标单位方向为 `e_i=(g_i-p_i)/||g_i-p_i||`、`P_goal=diag(e_i e_iᵀ)`，安全 nominal 为 `a_safe`。旧审计的 `goal_signed_nominal_motion_p = (P_goal u_p)ᵀ(P_goal a_safe)/||P_goal a_safe||`；`D_goal` 是 FF 减 FT 的 16-key 平均。它比较首步执行动作沿安全 nominal 的目标向分量，分母近零时解释受限。**B0（初态动作级）**：`D_goal < −0.14`。计算需要冻结控制器在初态对 16 个预定 key 各前向一次；不需要 rollout 或成功标签，但真实部署若只知道一个即时 key，则这项 16-key 期望是离线估计。预期 B0 与 FT→FF 的 B15 Loss 相关，因为 FF 把目标向推进转为较弱/横向的动作；FT 的终端注入仍可维持原推进。它是平均动作差，不保证任何单一 seed 成败。

**B1（H20 早期响应）**：`min_i progress_i^FF(20) − min_i progress_i^FT(20) < −0.12 m`。这需要两条配对的前 20 步模拟轨迹，属于 early simulation predictor，部署时不可从当前观测直接取得。其机制候选是重复的推进削弱延迟让行/通行，随后进入 Toy 的 window-progress/speed guard 或 timeout。B0 与 B1 的关系由相关和反例检验，不把任一条件写成充分因果律。

可证伪：若 held-out 上 B0/B1 的 precision、recall 与多数类/动作差幅度基线相近，或相同初态下动作削弱与 FT 成功关键行为无关，则预测条件不通过。跨场景性：goal direction 可跨场景定义；Toy 的 let-pass 和 deadlock guard 是场景特有的。

## Condition A：Ring 的环向转移

令 `P_tan(x)` 为各机器人相对于 Ring 圆心的切向正交投影；旧审计的 `tangential_exec_motion_norm_p=||P_tan u_p||`。`D_tan` 是 TF 减 TT 的 16-key 首步平均差。**A0（初态动作级）**：`D_tan > 0`。它比较 field/terminal 的**总执行动作**环向大小（含基准偏移），不能误写成纯 correction 增量；不使用 success、最终裕度或完整轨迹。该量是绝对环向运动，**不保证方向有利**；会保留这一可证伪缺点。原 transfer 数据的 signed tangent 在 discovery 中区分较弱，因此不能把 A0 伪称为已确定的正确绕行方向。

**A1（H20 早期响应）**：`A0` 且 `min_i progress_i^TF(20) − min_i progress_i^TT(20) < 0`。短期目标进展下降可能表示安全绕行/退让而非失败；若随后环向遍历、共同目标到达改善，则支持“先绕行再收敛”的行为路径。这个条件需要两条 H20 配对轨迹和固定未来噪声，是 early simulation predictor。它也可能在双方都失败时触发；必须报告误报。

可证伪：若 held-out Ring gain 的捕获率与多数类/动作幅度基线无区别、误报过多，或 gain 主要为 late convergence 而非环向遍历，则 A0/A1 不能作为一般 Gain 条件。Ring 只有一个 B15 Loss，不训练或宣称 Loss 分类器。环向投影依赖 Ring 几何；其可迁移抽象可能是“沿通行资源切向的有效修正”，但未经其他场景检验。

## 固定评估与基线

- 主标签：B15（固定 16 seeds，≥15 成功），每个 scene/state/η 一条，unknown 保留。四类 Gain、Loss、Both-success、Both-failure 均报告；二分类只把指定 Gain 或 Loss 作为 positive，其余已知类为 negative。
- Splits：01–08 discovery，09–16 validation，17–24 held-out；初态内所有 η/seed 同组。阈值此刻冻结，validation 和 held-out 只评估，不再调参。
- 比较基线：全预测 negative；训练组多数类常数；初始最大目标距离高于 discovery 中位数（Toy `1.743316 m`、Ring `4.688165 m`）；seed0 首动作差大于 discovery 中位数（Toy `0.233135`、Ring `0.070644`）；另报 eta-only 规则的描述性强度，不从 ETA 指数直接造物理机制。
- 报告 TP/FP/FN/TN、precision/recall/balanced accuracy、PR-AUC（非退化分数时）、按初态重采样的不确定性与至少两类反例。零预测 positive 时 precision 按 0 报告。Ring Loss 只列实际个数。
- A0/B0 的 16-key 动作平均、A1/B1 的 H20 双轨迹均不需要最终标签，但属于离线可比较两 placement 的计算；当前真实部署并不预知全部未来 key。若另用 seed0 当前 key 的首动作，单列为条件于该 key 的即时预测。

# C1：三个问题的直接检验

只检验：controller 行为是否包含可迁移规律、模型能否预测未见 controller 的 Q、预测是否提高 η 选择成功率。复用 P1 **14 个真实 Ring FF Flow checkpoint ×24 states×K16×Q4** 的 21,504 个 seed 记录（272 numerical-unknown）和已有 R5 静态响应；**新增 rollout 0，新增表示/probe 0，GPU 0**。

已有 Round 3 匹配模型只测试共享 state 面板上的 controller 留出。本次补充相同的小型核岭 critic：R0 输入 physical state，R5 额外输入当前 state 的已有 controller 响应；16 个输出对应共同 K16。外层同时留出整个 controller 和四分之一 states，共 56 折，保证每个 controller/state 有一次预测，其标签和该 physical state 在其他 controller 下的标签都未参与该折训练。内层只在源 controller 和源 states 上选带宽/正则化及归一化。没有根据目标标签挑选模型。

这些数据此前已经打开，因此本次是严格划分下的探索性重分析，不能称为独立新数据确认。η 是固定 K16 的离散输出，不检验连续 η 或跨场景泛化。

## 1. Controller 之间是否存在可迁移的行为规律，能否解释 Q 差异？

**有可迁移的行为预测信息；完整闭环机制解释仍未验证。**

在相同模型和源标签下，加入已有 controller 响应后，held-controller/held-state Q MSE 从 **0.08190 降为 0.05879**；**13/14** 个 held controller 改善。把每个 state 的 Q 曲线均值扣除后，η-centered MSE 仍从 **0.06111 降为 0.04701**，说明学到的差异不全是 controller 整体成功率偏移。

进一步固定训练好的 R5 模型，只把目标 controller 描述替换成每个源 controller 在同一 physical state 的描述，逐 donor 评分后平均。错配描述的平均 MSE 为 **0.11773**，正确描述为 **0.05879**，14/14 个 controller 都是正确描述更好。这说明预测确实使用了与目标 controller 对应的行为信息；该对照不证明某一个响应分量构成因果机制。

边界必须保留：P1 缓存保存 first action 和终止摘要，没有完整闭环轨迹。模型输入是无 outcome 的静态响应，未使用 episode length、终止结果或最终距离。上述结果支持“静态 controller 行为与闭环 Q 存在可迁移统计关系”，**不能宣称已经解释了闭环反馈如何移动完整 success basin**。本批只采样 K16，也没有识别连续 basin 边界。

## 2. 利用已见 controller 数据，模型能否预测未见 controller 的 Q？

**在这个已有面板上可以改善预测，但准确度与决策相关误差仍有限。**

| 指标 | Controller-blind R0 | Controller-conditioned R5 |
|---|---:|---:|
| Q4 中点 MSE | 0.08190 | **0.05879** |
| MSE 的 numerical-unknown 区间 | [0.08065,0.08488] | **[0.05765,0.06165]** |
| 同 state 内 K16 Spearman | 0.40594 | **0.44272** |
| η-centered MSE | 0.06111 | **0.04701** |
| 相对源固定 η 的 advantage MSE | 0.21901 | 0.21868 |

Q MSE 的平均减少为 **0.02312**，共享未知标签赋值下的差值区间 **[0.02245,0.02378]**；按 controller 重抽样的描述性 95% 区间 **[0.00987,0.03921]**。这不是所选 checkpoint 库以外的总体置信保证。

关键限制是：虽然整体 Q 和 η 排序有所改善，预测“某个 η 是否比源固定 η 更好”的误差几乎没有改善，**0.21901→0.21868**。因此不能由 Q MSE 推出正确的 intervention selection。各 controller 的概率校准分箱、完整预测和内层选模记录均已保存。

## 3. 学到的规律是否真正改善未见 controller 上的选择？

**没有。本次条件 critic 的选择明显差于两种基线。**

所有选择都由冻结预测在同一 K16 上 argmax 得到；源固定 η 仅由各折源训练标签选择，结果均为原 K16 的 η₁₀。直接复用对应的真实 rollout 标签评价，未重跑 episode。

| 选择方法 | 平均经验成功率的 numerical-unknown 区间 |
|---|---:|
| 源域固定 η | **[60.19%,61.16%]** |
| Controller-blind critic | **[60.12%,61.09%]** |
| Controller-conditioned critic | [55.65%,56.70%] |

条件 critic 对 blind 的配对收益为 **[−4.54,−4.32]pp**，对固定 η 为 **[−4.61,−4.39]pp**；后者保守端点的 controller bootstrap 区间为 **[−6.99,−2.46]pp**。只有 **3/14** 个 controller 的固定 η 比较下界非负，不能由少数改善掩盖整体退化。相对目标 Q4 empirical oracle 的遗憾区间：R0 **[0.13393,0.16146]**，R5 **[0.17857,0.20610]**。该 oracle 是高噪声 hindsight 参照，不是真实总体最优。

| Held Flow（沿用 P1 名称缩写） | 固定 η 成功下界 | Blind 成功下界 | Conditioned 成功下界 |
|---|---:|---:|---:|
| v4 wide plus-w | 0.646 | 0.646 | 0.604 |
| base u | 0.240 | 0.240 | 0.240 |
| v4 | 0.344 | 0.344 | 0.281 |
| v6 plus-w | 0.312 | 0.312 | 0.250 |
| v3 plus-w | 0.906 | 0.906 | 0.906 |
| v4 targeted plus-w | 0.469 | 0.469 | 0.427 |
| v3 | 0.771 | 0.771 | 0.760 |
| v2 | 0.396 | 0.396 | 0.385 |
| v10 local | 0.656 | 0.656 | 0.667 |
| v5 plus-w | 0.385 | 0.385 | 0.302 |
| v7 local | 0.990 | 0.979 | 0.906 |
| v4 plus-w | 0.417 | 0.417 | 0.271 |
| v8 local | 0.990 | 0.990 | 0.979 |
| v9 local | 0.906 | 0.906 | 0.812 |

因此三个问题的判定分别是：**行为规律的预测关联有证据，闭环机制解释 PARTIAL；未见 controller 的 Q 可学习性获得有限面板支持；提高选择成功率的 claim REJECTED（当前模型/面板），TRANSFER_GATE_FAILED。** 这不是所有可能 critic 都不可学习的证明。当前直接证据指向：学到 Q 的一部分变化，尚未准确学到相对固定 η 的决策收益。

复现与证据：[冻结测试协议](../diagnostics/part3_three_questions_20261012/PROTOCOL.md)、[完整结果](../diagnostics/part3_three_questions_20261012/results.json)、[逐折划分/配置](../diagnostics/part3_three_questions_20261012/folds.json)、[预测](../diagnostics/part3_three_questions_20261012/predictions.npz)、[运行脚本](../diagnostics/part3_three_questions_20261012/run.py)、[核验](../diagnostics/part3_three_questions_20261012/verification.json)、[运行记录](../diagnostics/part3_three_questions_20261012/RUN_LOG.md)。

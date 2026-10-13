# P2′ 数值 unknown 的估计规则补充

本补充在 P2′ 聚合结果和读取新 outcome 前冻结，不修改原预注册的样本、模型、候选集、决策门和多重比较规则。原因：原方案要求 unknown 不当作失败并报告上下界，但分析实现需要将其形式化。

对每个 state/controller/η 在独立评价 seed4–7 的标签，记成功数 S、明确失败数 F、数值 unknown 数 U。可观测成功率为 S/(S+F)，若 S+F=0 则记 missing，绝不以 0 代替；全 seed 的保守下界为 S/4，上界为 (S+U)/4。主表同时报告可观测成功率、unknown 比例和上下界。选择器与固定 η 的配对差异以下界差（selector S/4 减 baseline (S+U)/4）及上界差（selector (S+U)/4 减 baseline S/4）呈现；严谨正收益必须在最坏情况下的下界差的预注册同时置信区间下界 >0。目标全部 K16 固定 η 的 simultaneous 比较同样将每个固定 η 的 unknown 计入其最有利上界。

训练 NLL 只用明确成功/失败标签。选择最佳固定 η 时，不使用 unknown 作为失败；若 unknown 使候选排序不确定，则记录排名不确定性。此补充只收紧结果解释，不增加 rollout，不通过筛除 unknown 寻找收益。

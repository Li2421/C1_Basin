# P2′ 成对 unknown 界的分析修正

首轮 P2′ 统计已完成并保存在 `P2PRIME_RESULTS_UNCOUPLED_UNKNOWN.json`。审计发现最坏 unknown 界把 selector 与 baseline 在**同一个 state、同一个 η**上的未知 rollout 当成独立事件：当两者选择相同 η 时，差异其实精确为零，该实现却扣除 baseline 的 unknown 比例。此规则仍是保守下界，但与预注册的配对设计不一致，也会使 R0 与相同预测的错配描述出现人工负界。

只读重算用相同 seed4–7 标签、相同 5,000 个 state 成组 bootstrap 抽样与相同 Bonferroni 门槛；对同一 state/η 的 matched baseline unknown 精确抵消，对不同 η 仍使用 selector 成功下界减 baseline 成功上界。全 K16 固定比较逐候选做这种配对抵消后再取最不利候选，未重新选 state/η、未重训、未生成 rollout。首轮 JSON 保留；最终 `P2PRIME_RESULTS.json` 给出修正后的紧致最坏界并报告结论是否变化。

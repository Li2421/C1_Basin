# P2′ 选择收益的强判读补充

此补充登记于 P2′ 任何新标签前，保留原 `PREREGISTRATION_P2PRIME.md` 的全部分割、模型和报告指标。三种可用Controller表示（R3、R4_action、二者拼接）×两个held placement共有6个主要机会；因此另报每项配对比较的Bonferroni familywise区间（每机会双侧置信水平99.17%），95% CI仍按原预注册展示。

把结论分成两层：①“零目标标签的source-prior transfer”要求相对源合并固定、R4最近源固定、R0及错配描述，均在familywise区间下为正。②更强的“state-conditioned η selection确实超过场景级固定η”还须相对目标全部K16 hindsight最优固定η的 max-over-K16 simultaneous区间下界>0；否则只报告为source-prior迁移或INCONCLUSIVE，不能借 Q MSE 宣称解决主目标。数值unknown的最坏上下界同时报告。目标 hindsight 固定η只作评估上界，绝不进入模型训练/校准。

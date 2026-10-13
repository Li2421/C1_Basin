# E4 冻结补充（读取本任务 factor outcome 前）

E0 已通过，E1 不触发；E4 作为次要的零目标标签/少样本场景级 η regime 审计，
不取代 Ring state-conditioned 主结论。复用上游 `data/factors.npz`：12 个物理配置，
各 2 个 horizon × 16 state × 2 controller × 4 η × 2 seed；同一物理配置的
两种 horizon 同时留出。源配置可用全部 state；目标配置仅 state 0–7 用于
1/2/4-shot 校准、state 8–15 作评估。控制器分开预测，η 集合固定 K4。

比较：(i) 源均值固定 η，(ii) raw 几何参数的 leave-physical-config-out
岭回归，(iii) 上游预定义的预测时可计算机制代理坐标岭回归，(iv) 目标
1/2/4-state η prior 校准及原始/机制预测+offset；目标测试结果只作评估。
岭回归对源训练 fold 内按列标准化，常数列保留0，截距不惩罚，固定 L2=1，
不根据目标结果调参。每个目标 state 的两个 seed 均用于校准；对 0–7 的
校准池按 RNG 20261011 产生20个排列，使用每个排列的首 k 个 state。
offset 权重为 2k/(2k+4)，先验为源零目标预测；同样权重应用于 prior-only。
报告 target 8–15 闭环成功、相对目标 held-out hindsight fixed η 的 regret，
及各方法配对差；以12个物理配置聚类 bootstrap 2000次。

跨拓扑测试额外留出整个 topology，训练另一 topology，除此之外不调参。
代理 capacity、time margin、detour 分别是原先 factor_data.py 的几何归一
通道宽度、距离/最大速度余量、最短绕行比；它们不是经过测量的吞吐量或
联合可解性定理。若跨拓扑收益不稳，机制不变性判为 INCONCLUSIVE。

本补充的时间/SHA 收据另存。不得覆盖上游 factor 数据或现有校准结果。

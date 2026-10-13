# Stage 0：既有证据的探索性审计

本阶段零新增 rollout、零重新拟合 critic。识别真实 Flow 之间 controller→η regime 的覆盖及表示可辨识性；不进行新泛化确认，不据此筛选 state 或修改锚点。

复用 round4 Stage A/B 的 14 个 Flow、共享128个训练初态、平衡随机8-head Q4 标签，以及原静态特征和冻结模型/逐seed预测。未知=-1、未请求=-2，二者不得编码成已知失败。经验最佳η沿用已有128训练state的观测有效标签均值，作为探索性regime身份。每η仅64个state，arm分配不同，置信区间按128个共享物理state重采样5000次、保留原head请求mask。给出未知上下界；最佳η与η10、η13差的保守区间，并对14个Flow×2个比较作max-T校正。经验最佳η由同一标签选出，其差值不作无偏确认结论。

R3/R4_action/R3+R4空间采用128个初态上的controller特征均值。每次留一，只用13个源controller及其初态计算特征尺度，预测四类regime。NN、L2=1的多项logistic（截距不惩罚；不调参），与源多数类比较；类目排序固定[2,4,10,13]，平票取最小索引。14个controller统计量仅描述性，不推断独立训练run数量为14。

折1/4分别报告三个空间的最近邻及冻结Stage A/B controller-prior η（3fitseed），直接读取历史无目标标签prior预测。必要折regret：终点评价上的 target固定η参照 − model自身无目标标签controller-prior，加上 model-prior − per-state选择；两项相加等于 target固定η − per-state选择。目标固定η只由历史discovery seed选取。未知用上下界，另报相同η取消的配对保守差。所有这些是已打开state上的 DEVELOPMENT/EXPLORATORY。

只写本轮新目录；不修改历史数据库标签、checkpoint或报告。阶段不设决策门。

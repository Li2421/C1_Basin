# 旧 Ring oracle–critic 7 个 miss 的回顾审计

此审计只解释已查看的旧测试，不参与 K、eta=0、数据或重训决定；这些决定已经由 train/dev 冻结。

7 个状态均存在 16/16 的 generator proposal，但 critic 选择的 proposal 未达到 15/16：

| 旧状态 | oracle | critic 选择 | critic 实际成功数 | 现象 |
|---|---|---|---:|---|
| 000896 | sample_0 | sample_3 | 14/16 | 高 Q 之间的严格阈值误排 |
| 000905 | sample_2/sample_3 | sample_0 | 9/16 | robust/non-robust 混淆 |
| 000910 | sample_0 | sample_3 | 3/16 | 严重高估低 Q proposal |
| 000926 | sample_0 | sample_1 | 14/16 | 高 Q 之间的严格阈值误排 |
| 000927 | sample_0 | sample_2 | 12/16 | robust/non-robust 混淆 |
| 000937 | sample_3 | sample_2 | 5/16 | 严重高估低 Q proposal |
| 000938 | sample_0/sample_1 | mean | 14/16 | 多个近满分候选中的阈值误排 |

其中 3/7 是 16/16 对 14/16 的窄间隔错误，4/7 是更明显的概率/排序外推错误。旧 miss 不是 proposal 缺失，因为 oracle 候选已经存在；直接原因是冻结 critic 在这些未见状态上高估了次优 proposal。

train/dev 的 finite-set top-1 accuracy 为 1.0、regret 为 0，故预注册的 R4 触发条件未满足。全标签仍有 449 个 high-score/low-Q 点，说明边界或稀疏支持区的全局校准是潜在限制；但不能据此使用旧测试案例重训。

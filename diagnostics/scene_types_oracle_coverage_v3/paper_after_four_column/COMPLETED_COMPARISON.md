# 已执行完论文配置与四列族：相同17场景比较

Identical17 geometry-selected scenes and state0..7. C1 nominal controllers each136 cases. Paper fixed profiles and C1 fixed first candidate are deployed policies; Oracle64 is a per-case finite-search envelope, not equal-budget policy performance.

表中时间均为各自成功样本条件均值，不能直接解释为配对速度优势。共同成功样本的配对时间、仅一方成功数、全队列截断时间与逐UID见JSON。

|C1控制器/策略|成功/136|成功率|成功平均时间（秒）|全队列截断均时（秒）|
|---|---:|---:|---:|---:|
|centralized_sampling_fixed_first_point|53|39.0%|16.00|33.33|
|centralized_sampling_oracle64|102|75.0%|21.79|28.87|
|gradient_mpc_fixed_first_point|58|42.6%|16.91|32.94|
|gradient_mpc_oracle64|109|80.1%|21.84|27.53|

|论文固定配置|成功/136|成功平均时间（秒）|全队列截断均时（秒）|出现未处理异常的局数|单机初筛通过/17|
|---|---:|---:|---:|---:|---:|
|Wang 2017 完整死锁消解 reference_default|35|18.01|36.84|119|7|
|Wang 2017 完整死锁消解 v1_selected|32|16.19|36.63|115|6|
|Wang 2017 一致扰动 reference_default|35|18.01|36.84|119|7|
|Wang 2017 一致扰动 v1_selected|32|16.19|36.63|115|6|
|Zhang 2025 自适应风险死锁规避 reference_default|10|26.83|39.84|118|1|
|Zhang 2025 自适应风险死锁规避 v1_selected|9|25.66|39.80|118|1|
|Zhang 2025 死锁感知控制 reference_default/v1_selected|18|13.47|37.87|126|6|
|Lee 2025 旋转木马 reference_default|32|19.48|37.15|0|7|
|Lee 2025 旋转木马 v1_selected|22|15.95|38.09|0|8|
|Chen 2023 动态优先级＋右手规则＋安全走廊 reference_default/v1_selected|5|9.20|39.32|51|7|

- Paper exceptions use affected-robot one-step stop and recovery; solver/action errors and poor single-agent diagnostics prevent attributing low success to the original paper mechanisms.

- Success-conditional means, common-success paired means, and full-cohort truncated means are distinct. Earlier successful Oracle candidates are retained rather than picking fastest.

- The C1 fixed first point has C2=0; it does not establish necessity of all four columns.

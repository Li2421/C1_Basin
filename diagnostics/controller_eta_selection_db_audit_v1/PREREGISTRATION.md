# State-conditioned η selection：数据库优先审计规则

冻结范围：先完成 E0，再根据 E0 决定 E1/E2/E3 或 E4；E5 只审计证据。
本文件冻结于读取本任务任何新计算的 outcome 之前。时间及 SHA256 在
`preregistration_receipt.json`；不是第三方时间戳。

## 数据与纳入

只读 `shared_rollout_db/rollout.sqlite`；不写共享 SQLite、不生成 rollout、
不训练模型来完成 E0。核实此前 57,223 个至少16条有效 seed 的 cell 数，
同时检查该阈值以下的覆盖与其他遗漏面板。有效记录必须 `conflict_quarantined=0`
且 `numerical_failure=0`，同一 cell 内 seed_key 唯一；unknown、缺失和
aggregate-only 不作二值标签。每个 controller UID 下单独分析；不同 UID
不合并，即使名字相同。核查 controller 的 scenario、checkpoint、horizon、
projection、success/rng/conditioning、placement，以及 state 的 scenario、
physical/provenance 和 identity quality。非 true-t0 或无法核对执行语义的
面板单列为发现，不进入主决策。共享 η 身份按 canonical float64 η UID。

在同一 controller 中，按 state 可用 η 的 **完整交集** 建立候选面板；
同一 state 的每个 η 需要可对齐的16个共同有效 seed 次序。优先从
seed_key JSON 的 future_index（或明确的相同字段）确定编号，并检查
future_root/namespace 是否在比较中兼容；不能解释的 seed_key 只作探索性。
当一个 controller 下有多个同样完整的大面板时，同时报告完整 η 交集、
状态/η 覆盖矩阵与预先已发表候选池，防止通过缩小 η 池产生收益。
不删除已知表现最好的通用固定 η。所有扫描到的候选和排除理由留机器索引。

## E0 指标与检验

主分析用16个共同 seed 按 future_index 偶/奇分成8/8，两方向交叉拟合取平均。
敏感性分析做四组固定随机8/8对分（RNG 20261011），不择优报告。
每方向，A半为每个 state 选经验成功最多 η，也从 A半的所有 state 均值
选择一个固定 η；B半仅用于评估。η 并列以 canonical UID 字典序确定。
按物理 state 成组配对 bootstrap 2000次（RNG 20261011），保留 controller、
η、seed 对齐；报告百分点收益及95% percentile CI。这个交叉拟合
state-specific oracle 是高容量参考选择器，**不是部署时可用的 selector，
也不是真实/连续 η oracle**。另列一个看了全标签的乐观上界，不参与门槛。

定义 `H_xfit` 为上述 state-specific oracle − 固定 η 的 held-out 成功差。
定义 `U=1−固定 η held-out 成功率`；`I` 为全部候选 η 在全部有效 seed
上都无成功的 state 比例。`U−I` 是可观察剩余上界诊断，不保证有可学
selector。鲁棒代理把半数 seed 中至少7/8成功看作成功，同时报告 Q16
下B15=≥15/16、最强固定 η 覆盖率、每个 state 是否存在合格η、全部η
失败比例。用固定顺序的贪心 set-cover 计算覆盖所有可行 state 所需η数；
候选数≤20时另算精确最小数作审计。报告不同 state 的最优η、稳定唯一
偏好数、并列数、controller间 η 排序变化；后者只对同一物理state、η
交集和可比执行环境比较，不能把不同地图/controller嵌套当因果效应。

**主门槛（保持原定）：** 至少12个 state；`H_xfit≥0.05` 且 state bootstrap
95% CI 下界>0（平均成功或鲁棒代理任一）；`U−I≥0.05`；可行 state
的最小η覆盖数≥2。任何只在事后筛η、筛state、换split后通过的结果
为探索性，不标作确认。至少16seed/格是精度条件，不能据此推断B15
选择效果的统计CI为窄。若主门槛通过，须在独立state/seed E2确认。

## 决策后续

E0 无合格面板时，E1先检查Toy w0.41 isotropic、16个广初态、K8 η
和历史所有有效 seed。已存标签足以达成计划精度则0新增rollout；否则
先冻结缺失 cell/seed 列表、语义、时间/CPU成本和不可替代性，才可能
执行最小Q16确认。不扩大/删减候选池寻求正结果。E1仍不通过则进入E4，
结论限定为已审计场景/controller/η池与seed政策。

只有E0/E1出现主门槛合格者才进入E2。独立state与独立seed、完整包含
该环境最强通用固定η的候选池，重复同一判据；先用缓存，确有缺口
才评估最小新增。E2独立确认通过才进入E3的MANY/FEW/EXTRAP正式迁移。
不存在合格E2时，E3标为条件不满足，不能虚构已执行结果。

E4优先复用已保存的 `controller_state_geometry_transfer_v1` factor标签、
四场景旧标签及few-shot预测。比较 source-only固定η、预测时可获得
的原始几何/物理机制坐标、1/2/4目标state校准，严格按目标配置与
物理state分离；目标success只用于最终评估或明确的few-shot校准。
报告 source-only 跨配置/拓扑与校准后的不同证据等级；不把拓扑代理
叫作精确吞吐capacity。E5只读检查连续η点数和高精度标签覆盖，不由
K4/K8推断连续 basin。

未纳入Git的历史诊断/PoC资产只读；本任务新建独立目录。有限实验的
统计假设、负结果、数值未知与范围限制同等保存。纯Q审计按AGENTS免视频。

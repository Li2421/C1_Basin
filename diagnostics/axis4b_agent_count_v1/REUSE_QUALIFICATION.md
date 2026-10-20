# 完整轨迹的精确派生

这是主评估开始前冻结的执行优化，不改变物理、控制器、安全层、成功判定、能力门、选择算法或统计终点。当前 pilot 使用原 worker，既有 case 与 journal 不修改。源/目标主评估仍先登记全部九臂的独立身份、封存 preflight，再仅补 genuinely_missing。

时间表臂在切换前与同 state、同 future、同初始 η 的固定臂执行相同代码。若固定臂已在切换前结束，整条时间表轨迹由现有证据确定：success/collision 的已完成步数 ≤ switch，或 numerical_failure 的已完成步数与异常 step 都严格 < switch。后一种保留 unknown，绝不通过重跑前缀消除异常。timeout、切换点或切换后的异常，以及不满足兼容性证明的轨迹均不派生。需要切换后控制的 episode 从原始 t0 执行冻结 runtime，不使用新的续跑实现。

派生保留源轨迹路径与 SHA、原始事件、首动作、步数、误差与证据等级；新 raw 显式记录 `prefix_source` 的来源 UID、raw/trace 哈希、未到达的切换时刻与派生代码 SHA。新动作身份单独经默认 journal 和官方 merger 登记。它不是不同 η 的数据库 EXACT_REUSE 命中，而是经执行等价证明得到的标签。`new_physical_steps=0`，`deterministic_reused_steps` 记录避免重复的原轨迹步数。该字段在此表示给定 state/RNG 后的执行确定性，不表示 Flow 控制器跨 future 确定。

P2 future1 仍按原冻结实现由同臂 future0 派生，两类复用分别验证。`prefix_audit.py` 检查全部时间表派生证明；原冻结 `review.py` 检查所有 raw/trace、物理积分、DB、journal 与交付一致性。原始 analyze/statistics/selection 不变：逻辑 episode 仍按原设计计数、unknown 给出上下界、state 成簇保持臂间配对；逻辑 unknown 数不等于独立物理异常次数。

`performance_v2.py` 仅从实际仿真的 episode 计算物理吞吐、步耗时，排除以上两类派生，不把同一轨迹重复当作计时样本。另见 `TIMING_LIMITATIONS.md`。新的分析入口先执行额外派生审计，再调用原冻结分析、交付与复核。原冻结文件不改，附加收据链接主完成收据与派生审计。

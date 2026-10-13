# P1 真实 Flow controller 普查预注册

本文件在任何本轮 P1 新 rollout outcome 之前登记。P1 使用历史已打开的开发 state，是 **DEVELOPMENT pilot**。P0 已完成，P2 的标签分配/验证规则见独立 P0 supplement，不从 P1 反选。

## 样本、执行身份与预算

候选为历史 `FLOW_CHECKPOINT_INDEX.json` 的14个 Ring checkpoint。每个先只读检查 checkpoint SHA、训练/环境 manifest 指纹、观测维度、归一化、10步 Flow、4 agent×2动作、H700 物理常数。global checkpoint 采用其训练 manifest 的 global observation/动作坐标，local checkpoint 采用 local chart；物理plant、安全投影、FF积分、η基底、成功判据及seed规则一致。差异不能用改写 checkpoint、源代码或改变场景物理参数来掩盖。具体execution profile必须包含 checkpoint/manifest/adapter/FF/safety/评价代码hash、JAX x64。静态首步以及新旧native v10同一程序逐字节校验后才可批量运行。兼容失败者保留原因，不计为被检验controller。

样本选择只使用历史 native v10 outcome/冻结 Q512 预测：在已打开256个开发state里，以 seed0–7 的 native fixedη13 与 Q512所选η的成功率差排序，选正差最大的12个为 adaptive-demand；其余12个按 native fixedη13 的旧成功率四分位分层，用固定rng 2026101205、每层3个、不与前12重叠。遇并列按state UID。写定24个state UID及历史评分/hash；之后不得按任何新 controller outcome修改。每个兼容controller×24state×完整K16×seed0–3，共每controller1536次；最多14controller即21504个请求，**新 rollout上限25000**。seed0–1与2–3互为发现/评价，结果双向平均。K16含η13。数据库同语义同seed记录先精确复用，数值未知保留未知不重跑；实际新数按preflight确认。每worker单CPU，最多24并发。

## 静态表示

R3为现有field compact64；R4_action在 t0 和固定预定义 probe state 上查询η0投影动作与中央差分 Jacobian。禁止env.step、未来标签或轨迹；probe由起点位置/速度与朝向目标的确定性偏移生成，不能按P1 outcome选择。静态失败也记录，不能为提高结果删controller。所有静态查询使用与rollout相同jax x64。

## 指标与门

成功率是闭环 Q4 的两半交叉拟合评价；unknown 不当失败，主要报告下/上界。固定η由发现半的全state均值选，tie先取原生η13再按η索引；state η亦以发现半值选，tie按该controller发现半全局prior再按索引。目标B自身fixedη是可达基准，只用于headroom/遗憾分析，不作为部署zero-shot输入。对A→B，报告B自身fixedη减A的fixedη在B评价半成功率；state层用B自身state oracle减A的state oracle。双方向、两seed半平均。额外报告候选池Q均值、controller×η与state×controller×η交互能量，用独立半交叉积减弱噪声偏差。Q level与η排序单列。

门A：至少8个兼容controller的交叉拟合最佳固定η成功率在[.40,.95]。门B：在这些合格controller的所有有向不同controller对上，global或state层的median cross-controller regret≥.05；对global/state两种median用state成组bootstrap5000次的Bonferroni simultaneous 95%区间，两者之一**保守unknown下界的CI下界>0**。门A与门B必须同时通过才能进入P2，否则进入P2′。另报告每controller自身state oracle headroom和全K16 hindsight fixed敏感性，不靠删减强η制造收益。不得把oracle当作已训练的selector。

两层门只检验family是否提供可辨识的Controller–η差异，不宣称held-controller critic已泛化。P1使用旧开发state做设计，因此所有科学结论需由独立新state验证。若兼容controller不足8或preflight后新数超过25000，停止新rollout并登记资源/资格失败；不擅自更改门槛。阶段结束后冻结P2/P2′划分及独立预注册，在其新outcome前登记。

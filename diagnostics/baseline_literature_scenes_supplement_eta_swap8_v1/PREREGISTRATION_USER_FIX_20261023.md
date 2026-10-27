# 用户修正预注册补充

本补充优先于原说明中仅移动障碍约束与静态L布局两项。用户明确指出原说明设计错误，给定最坏情况法向约束、γ选择门、走廊/最大臂长规则；不是看结果调参。原控制器算法/超参数、机器人state、seed、障碍路径、物理、成功/碰撞判据、95%能力门、修正和Rc规则均不变。原结果保留标“原约束/原布局（作废）”，不进主分析。

移动公式和全区间推导、六个γ完整迎面扫描、10000快照前提分层、静态几何LP/实际直线净空在USER_FIX_DERIVATION_20261023及对应JSON。已选γ=.3、ε=.001001；R=1.1m、L=.2726464281937122m。这些在首条修订研究rollout前冻结；不按能力门/后续结果再调。

历史检查：原static/moving能力门49152条因约束/布局改变不能回答新问题；使用新scenario/controller/state/rollout UID，保留机器人RNG不随版本名改变。free能力门24576条直接精确复用；free主线及筛选用原factory/engine/执行指纹，不重跑已存在精确证据。旧三场景筛选/评估在原身份继续。8条free完整逐位回放验证保持，物理step和移动路径函数直接继承。

后续顺序：接管但不停止正在跑的old_eval→原free_main（11328请求，原语义）→修订static/moving能力门（2×48×32×8×2=49152，官方preflight扣精确复用）→PRIMARY资格表（free原43；其余按新结果95%下界门）→PRIMARY三场景主线（free11328精确复用；两修订版48 baseline+3×合格实例，每例32state×2seed）→两静态版本16固定/6配对训练、冻结与评估→moving转移free赢家→保存轨迹诊断/5000配对bootstrap/DB与源码审计/报告。

每批精确请求数在CASES/官方PREFLIGHT/REQUEST_LEDGER中封存。未知和求解器数值错误不重跑；数学无解的速度放松/安全minimax照原代码执行并记录。理论安全界以g>=0及约束满足为前提，回退违反时不冒充全局保证。gamma选择仅单元门，布局选择仅几何，不读取标签。

旧qualification和原报告保留，CURRENT分析资格用QUALIFICATION_PRIMARY_20261023；primary main实际case不含原两版旧UID。单进程官方merger/default journals、全目录读写边界和每批前后stat规则不变。用户总CPU≤18（含管理）/GPU≤2；当前旧评估作业没有收到任何停止信号，控制平面改用driver_user_fix_20261023。

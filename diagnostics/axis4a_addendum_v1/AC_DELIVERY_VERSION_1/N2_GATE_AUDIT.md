门的执行正确，N=2 Flow 在相向任务上能力不足。本结论限于原16个相向pilot与原三臂，门仍为20%，18.75%不四舍五入成20%，没有重判门或新增rollout。

| 臂 | 成功 | 碰撞 | timeout | unknown | 成功state索引 |
|---|---|---|---|---|---|
| safety_baseline | 1/16 (6.25%) | 0 | 15 | 0 | [11] |
| F10 | 2/16 (12.50%) | 0 | 14 | 0 | [0, 10] |
| S13t2 | 3/16 (18.75%) | 0 | 13 | 0 | [0, 7, 10] |

逐条state_uid、original_uid、终止、步数、raw路径及hash见N2_GATE_EPISODES.csv与JSON；三个臂的成功state可重叠，不把不同臂当独立state。

冻结checkpoint 7ea73d275ba3b928922cd68da24bf8b0c316c098d4e455050828acf8c0ecd6e5 的选择依据明确为非相向navigation四类各12/12，时间释放even12/12、odd11/12，冻结前未打开相向测试。训练数据与选择/验收分开：含特权示范不代表相向闭环控制已通过。后续相向frozen_eval与replication的原记录也在本审计引用，不能把非相向验收当相向成功保证。本轮pilot所有机器人同时对向争用同一Gap，且使用预注册R10精确η菜单、相应TT链和新state/RNG；与历史非相向/单向/分时放行任务不同。

修正的是专家复现检查器：旧actions是float32存储，旧checker先提升为float64再与新原生64位动作比较；新checker把新动作按旧artifact dtype序列化后比较。未放宽容差、未改变物理或执行精度、未重跑gate。历史专家原生float64首动作未存储，不能宣称那些未记录字节已验证。完整float64位置、终止与步数在修复前已一致。Flow primitive在修复前后一直16/16通过，本次只读重新逐字节核对16条首动作/完整位置/终止/步数，包含N2的4条。冻结runtime/controllers及全部gate原始执行hash未改变。

“实际路径与历史一致”仅限可一一对应的16条Flow→单安全投影→native历史复现路径。新TT修正菜单的完整历史轨迹本来不存在，不能把该门扩写成全部pilot/全部TT轨迹逐位复现；TT组合顺序另有原UNIT_TESTS证据。这一证据范围限制不是发现执行或判定错误。原门只按known成功比例判定、unknown给界；N2三臂无unknown，最大3/16，结论无需改门。

检查器完整diff、各源路径/sha、未变执行hash与历史任务统计见N2_GATE_AUDIT.json。

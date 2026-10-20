本轮安全与修正动作由逐字复制的 pipeline.py::decentralized_action 计算，固定传入 kind='local'。没有替换已有模块函数、修改投影器、物理、事件或成功标准。episode_frozen.py 保持逐字副本，用于 Round28 的16条逐位复现门；新增评估使用本轮 worker.py 封装臂调度、日志、官方精确身份和journal，而不是声称整份旧episode文件直接作为新worker入口。

各臂的动作路径：

| 臂全名 | 动作来源 |
|---|---|
| 只跑控制器、不加安全层 | 冻结nominal直接输出 |
| 只加安全层、四个固定修正/时间表臂 | 原样decentralized_action，原样LocalSafety、local_basis与两次投影 |
| 原20候选公平右手规则 | 原样fair_rhr.py::FairRuntime |
| 新4候选局部邻居右手规则 | 新子类NeighborRightRuntime，原样LocalSafety与两次投影 |

新旧episode均从同一runtime.make_env(state)真t=0重置开始，每步使用同一key_for(state,seed,t)，在同一switch_step切换相同精确系数，动作后调用同一原生env.step；原生done才停止，最多原配置700步。新worker的state_index是design按states逐项枚举确定，与按state_uid查询同一记录等价。成功由同一summary.collision_free_success、termination='success'且无异常共同决定。unknown保留，不重跑；没有将timeout改为deadlock。

旧episode累计的slow_counts只影响kind为rhr或rhr_f10的旧诊断分支；本轮固定kind='local'，该参数不参与动作表达式。公平右手规则使用原FairRuntime的私有Trigger；新增邻居规则使用预注册的当前局部位置。新worker不记录旧episode的所有诊断字段，但仍完整保存positions、velocities、actions、goals和原生summary，并由不改定义的Round30g纯函数重算机制证据。错误回合的日志步数不用于任务成功标准。

源码、协议、checkpoint和各阶段提交前preflight均有哈希收据。32条授权历史重放逐位通过；A*两个seed的动作、位置、速度将在主评估结束后按实际字节核对。不得因新增日志或critic编码而重复执行已有兼容标签。

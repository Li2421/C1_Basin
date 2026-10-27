
2026-10-23T04:04:34.365764+00:00 第二部分独立开工审计1540源文件；新几何/组件门、32条SI逐位回放和strace通过。提前提交的回放批28322已停，保留29条；完整stat通过后28338仅恢复3条。此为调度程序失序，不倒补审计时间；未发现新的越界写入。H已封存不写。

2026-10-23T05:18UTC 能力门73728请求已合库，unknown0；free43/static12/moving0合格，移动碰撞4448/24576。按冻结95%门保留结果，不改场景/控制/门；moving修正评估缺口明确。旧固定η训练5888请求，734精确复用+2历史unknown保留，仅补5152；当前执行中。

2026-10-23T05:23UTC 证据收尾代码检查：跨阶段raw查找原先遍历*_CASES会误读REPLAY_CASES；改为索引实际不可变*/raw/roll_*.json，5个已完成不同终止结果的跨阶段精确lookup通过。postflight只遍历已注册主case清单，避免把MISSING或回放清单当研究批次。未改冻结执行源码、任何raw/trace、注册UID或标签；新研究未重跑。补充类型等权诊断率输出，分类及bootstrap数学内核不变；无合格moving实例的4a候选明确不作能力门支持的死锁证据。

2026-10-23T05:25UTC 后续新建队列细分到单个动作，seed0/1仍在同组；当前old_fixed_train封存队列保持原样。原算法、case UID/RNG/物理、默认journals和官方preflight完全不变。纯调度改动减轻单个state长尾串行多η的影响，见SCHEDULING_ACTION_GROUPS_20261023.json。

2026-10-23T06:12UTC 用户询问剩余时长及moving correction可用性。仅重算保存证据：moving单机24576请求中19052成功、4448碰撞、1076超时；全部碰撞仅移动障碍，constraint_violation/restoration/speed_excess均0；这支持动态安全层时间设定的缺口，不是correction有效或无效的实验。未放宽能力门，未补跑未合格correction。基于旧η13 eval耗时和本轮吞吐，剩余粗估3–6小时；八机N8尚未测时，需后续校准，见RUNTIME_FORECAST_20261023_0610.json。

2026-10-23T06:15UTC 用户追问碰撞是否安全过滤之后及原因。重算一条保存的MPPI4/state10/solo5/seed0轨迹：两次投影、η0，frame83当前QP余量+.04719，但真实移动hdot+γh=-.11947；中心距.351659→.343752，真实门.35m，重叠6.248mm。膨胀层.3626的正不变前提已失去，当前位置静止CBF未计障碍接近，单帧膨胀不能自动保持后续膨胀余量。详见MOVING_COLLISION_CAUSE.md及EXPLANATION.json。仅保存证据核查，未新跑、未改冻结源码或规则。

2026-10-23T06:31:42.667424+00:00 用户修正：这是用户指出说明设计错误后的修正，不是看结果调参。仅改变moving障碍约束（速度上界最坏情况半空间，去除一步半径膨胀，ε=.01配对d_safe）和static L布局（指定半径范围、最大臂长、名义走廊与既定扰动state直线可达）。旧结果保留标“原约束/原布局（作废）”，不进主分析；旧场景粗筛和free语义/数据不变、运行不中断；碰撞判据、成功标准、95%能力门都不变。

2026-10-23T06:57UTC USER_FIX_READY已封存：γ=.3，6×164×200迎面扫描全过；10000随机快照8844满足前提、1156不满足前提保留，失败/unknown0。R=1.1,L=.2726464281937122是名义走廊+32state实际机器人净空同时约束的几何最大值。8类型free完整逐位回放全过；step/mover path直接继承，原冻结源码所有SHA一致。修订能力门49152官方missing已封存；先运行5分钟单shard28418（postguard通过），现继续单shard28419占用空出的1份额，与16个old_eval worker并行，管理1总18。新coordinator在regular revised capability前只USR1辅助新能力门worker并恢复部分claim；旧/free不发停止信号，UID/raw/unknown保留。

2026-10-23T11:57:08.616141+00:00 用户新增：第三阶段完成后追加寻找最佳correction，包含固定参数与切换时间表的搜索；先历史证据复用并冻结搜索/验证协议，当前16固定+6配对及moving转移主分析不变。仅登记后续任务，尚未启动额外搜索，详见FOLLOWUP_BEST_CORRECTION_REQUEST_20261023.json。

2026-10-23T11:57:08.616141+00:00 排名辅助脚本缺少np显式导入导致new_fixed_train完成后的pairs规划NameError；仅补import numpy as np。两scene各16候选排名与独立标量均值校验通过；不改冻结科学模块/UID/raw，不重复实验。原DRIVER_ERROR归档driver_actions/DRIVER_ERROR_20261023_1932_preserved.json，恢复同一driver入口。

2026-10-23T12:00:01.822294+00:00 用户要求继续并盯进程：已启动仅观察的20秒进程监控，核对driver/资源守护、stage新raw计数、Slurm全用户占用与异常标志，写PROCESS_WATCH_STATUS/INCIDENTS；不自动重跑或改科学参数。管理进程通过local_cpu_guard.pin共享1个管理CPU。当前new_pair_train 10worker，其他用户12CPU和1pending，资源约束保持。

2026-10-23T12:03:00.701947+00:00 将已授权的第三阶段后最佳correction搜索指针加入最终HANDOFF生成器，明确part2 COMPLETE不等于后续搜索完成；仅交接元数据，不改科学实现或结果。

2026-10-23T12:08:57.584841+00:00 持续观察确认new_pair_train1344完成并合库（615success/729timeout/0collision/0unknown），批前后stat均通过。new_eval在冻结free eta9/eta15_to_9及static eta0/eta0_to_3后才preflight/submission：5632逻辑请求、2816精确复用、2816新补，job28486十shard已产出raw。其他用户12CPU+1pending，依capacity使用10worker和1动态管理核心；监控和协调器健康。
